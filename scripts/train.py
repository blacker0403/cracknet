import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.crack500 import Crack500
from datasets.transforms import TrainTransform
from datasets.manifest import (FOLDER_DIGEST_PREFIX, folder_samples, folder_digest, load_manifests,
                               audit, manifest_digest, verify_split_lists, verify_manifest_paths)
from models.cracktoponet import build_model
from losses.combined import CombinedLoss
from utils.config import load_config, validate, save_config, fingerprint
from utils.seed import seed_everything, seed_worker
from utils.logging_utils import environment, dump_json, get_device
from utils.checkpoint import save_checkpoint, read_checkpoint, resume_checkpoint, initialize_from_checkpoint
from utils.complexity import complexity
from utils.evaluation import evaluate
from utils.visualization import training_curves
from utils.artifacts import TrainingArtifacts, dataset_profile
from utils.metrics import pixel_scores
from utils.resume_policy import check_resume_config, approved_checkpoint_config, record_resume_change


def checked_data(cfg, checkpoint=None):
    d = cfg['data']
    if not cfg.get('multidomain') and (checkpoint is None or checkpoint['manifest_hash'].startswith(FOLDER_DIGEST_PREFIX)):
        samples = folder_samples(d)
        return samples, folder_digest(d, samples)

    # Keep the original data signature for accepted legacy checkpoints and LODO runs.
    manifests = load_manifests(d['manifest_dir'])
    verify_manifest_paths(d['root'], manifests)
    if cfg.get('multidomain'):
        from datasets.multidomain import validate_fold
        validate_fold(cfg, manifests, verify_sources=True)
    if d.get('split_lists'):
        verify_split_lists(d['root'], manifests, d['split_lists'], d.get('split_path_map'))
    report = audit(d['root'], manifests, d['label_map'], d['require_source_ids'])
    report.update(dataset_version=d['version'], synthetic=d['synthetic'], source_id_regex=d['source_id_regex'],
                  source_provenance='Configured IDs; official source provenance is not independently verified by this tool.')
    dump_json(Path(cfg['output_dir']) / 'data_audit.json', report)
    # Training diagnostics use only train/val data; test model outputs are frozen separately.
    profile = {**report, 'samples': [s for s in report['samples'] if s['split'] in {'train', 'val'}]}
    dataset_profile(profile, Path(cfg['output_dir']))
    if not report['valid']:
        raise ValueError(f'Dataset audit failed: {report["errors"][:5]}. See data_audit.json. No training/test permitted.')
    return manifests, fingerprint({'manifests': manifest_digest(manifests), 'content': report['content_digest']})


def optimizer_and_scheduler(model, cfg):
    t = cfg['train']
    optimizer = torch.optim.RMSprop if t.get('optimizer', 'adamw') == 'rmsprop' else torch.optim.AdamW
    opt = optimizer(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    scheduler = (torch.optim.lr_scheduler.MultiStepLR(opt, milestones=t['lr_milestones'], gamma=t['lr_gamma'])
                 if t.get('scheduler', 'cosine') == 'multistep' else
                 torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t['epochs']))
    return opt, scheduler


def train(cfg, resume=None, stop_after_epoch=None, init_checkpoint=None, allow_resume_changes=False, reset_early_stopping=False):
    if resume and init_checkpoint:
        raise ValueError('Choose --resume or --init-checkpoint, not both.')
    if reset_early_stopping and not (resume and allow_resume_changes):
        raise ValueError('--reset-early-stopping requires --resume and --allow-resume-changes.')
    validate(cfg)
    if cfg.get('multidomain') and init_checkpoint:
        raise ValueError('LODO starts from the shared ImageNet encoder; arbitrary crack checkpoints may contain the held-out domain.')
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if not resume and (out / 'last.pth').exists():
        raise FileExistsError(f'{out}/last.pth exists; use --resume or a new output directory.')
    if (out / 'frozen.json').exists():
        raise ValueError('Frozen experiment cannot be trained further; choose a new output directory.')
    torch.set_num_threads(cfg['train']['cpu_threads'])
    seed_everything(cfg['seed'], cfg['train']['deterministic'])
    if cfg['train'].get('matmul_precision'):
        torch.set_float32_matmul_precision(cfg['train']['matmul_precision'])
    state = read_checkpoint(resume) if resume else None
    samples, digest = checked_data(cfg, state)
    print('[data] ' + ', '.join(f'{split}={len(rows)}' for split, rows in samples.items()), flush=True)
    device = get_device(cfg['train']['device'])
    if state:
        if state['manifest_hash'] != digest:
            raise ValueError('Dataset files changed since training. Resume with the original dataset.')
        if not approved_checkpoint_config(state['config'], cfg):
            check_resume_config(state['config'], cfg, allow_resume_changes)
            record_resume_change(state['config'], cfg, resume, state['epoch'], state.get('bad_epochs', 0), reset_early_stopping)
        elif reset_early_stopping:
            record_resume_change(state['config'], cfg, resume, state['epoch'], state.get('bad_epochs', 0), True)
    model = build_model(cfg['model'], initialize=not bool(state or init_checkpoint)).to(device)
    if init_checkpoint:
        record = initialize_from_checkpoint(model, init_checkpoint, cfg)
        dump_json(out / 'initialization.json', record)
        print(f'Model-only initialization: {record["checkpoint"]}; optimizer/scheduler/epoch are new.', flush=True)
    criterion = CombinedLoss(cfg['loss']).to(device)
    opt, scheduler = optimizer_and_scheduler(model, cfg)
    amp = cfg['train']['amp'] and device.type == 'cuda'
    amp_dtype = getattr(torch, cfg['train'].get('amp_dtype', 'float16'))
    if amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('This CUDA device does not support the configured bfloat16 AMP.')
    scaler = torch.amp.GradScaler('cuda', enabled=amp and amp_dtype == torch.float16)
    generator = torch.Generator().manual_seed(cfg['seed'])
    d = cfg['data']
    train_set = Crack500(d['root'], samples['train'], d['label_map'],
                        TrainTransform(d['crop_size'], d['foreground_crop_probability'], d['augment'],
                                       reflect_mask_padding=d.get('reflect_mask_padding', False)))
    val_set = Crack500(d['root'], samples['val'], d['label_map'])
    sampler = None
    if cfg.get('multidomain'):
        from datasets.multidomain import domain_sampler, domain_probabilities
        epoch_samples = cfg['train']['steps_per_epoch'] * cfg['train']['batch_size'] * cfg['train']['accumulation_steps']
        sampler = domain_sampler(samples['train'], cfg['multidomain']['sampling'], epoch_samples, generator)
        dump_json(out / 'sampling.json', {'method': cfg['multidomain']['sampling'], 'samples_per_epoch': epoch_samples,
                  'optimizer_steps_per_epoch': cfg['train']['steps_per_epoch'],
                  'domain_probabilities': domain_probabilities(samples['train'], cfg['multidomain']['sampling']),
                  'within_domain': 'uniform images with replacement'})
    epoch_samples = len(sampler) if sampler is not None else len(train_set)
    loader_options = {'prefetch_factor': d.get('prefetch_factor', 2)} if d['workers'] else {}
    loader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=sampler is None, sampler=sampler,
                        num_workers=d['workers'], worker_init_fn=seed_worker, generator=generator,
                        pin_memory=device.type == 'cuda', persistent_workers=False, **loader_options)
    save_config(cfg, out / 'config.yaml')
    if state and allow_resume_changes and (out / 'planned_config.yaml').is_file():
        save_config(cfg, out / 'planned_config.yaml')
    dump_json(out / 'environment.json', {**environment(), 'effective_batch_size': cfg['train']['batch_size'] * cfg['train']['accumulation_steps'],
                                        'amp_effective': amp, 'epoch_boundary_resume': True})
    (out / 'model.txt').write_text(str(model))
    dump_json(out / 'complexity.json', complexity(model, (1, 3, d['crop_size'], d['crop_size']), device))
    start_epoch, best, bad_epochs = (resume_checkpoint(state, model, opt, scheduler, scaler, generator) if state else (0, -1.0, 0))
    if reset_early_stopping:
        bad_epochs = 0
    if start_epoch >= cfg['train']['epochs']:
        raise ValueError('Checkpoint has already completed configured epochs.')
    history_path = out / 'history.csv'
    history = pd.read_csv(history_path).to_dict('records') if state and history_path.exists() else []
    if state and len(history) > start_epoch:
        # A previous interrupt may have occurred after history export but before last.pth.
        pd.DataFrame(history).to_csv(out / f'history_uncommitted_{len(history)}.csv', index=False)
        history = history[:start_epoch]
        pd.DataFrame(history).to_csv(history_path, index=False)
    if state and len(history) != start_epoch:
        raise ValueError('History length differs from resume epoch; use matching run artifacts.')
    patience = cfg['train']['early_stopping_patience']
    if state and patience and bad_epochs >= patience:
        save_config(cfg, out / 'args.yaml')
        dump_json(out / 'training_status.json', {'complete': True, 'last_epoch': start_epoch - 1, 'best_iou': best,
            'stopped_early': True, 'stop_reason': 'restored patience already exhausted', 'bad_epochs': bad_epochs,
            'early_stopping_patience': patience, 'optimizer_steps': len(history) * cfg['train'].get('steps_per_epoch', (len(loader) + cfg['train']['accumulation_steps'] - 1) // cfg['train']['accumulation_steps'])})
        print(f'[early stopping] restored bad_epochs={bad_epochs} >= patience={patience}; no additional update.', flush=True)
        return out / 'best_iou.pth'
    writer = SummaryWriter(str(out / 'tensorboard'), purge_step=start_epoch + 1 if state else None)
    artifacts = TrainingArtifacts(out, cfg, writer)
    completed = False
    dump_json(out / 'training_status.json', {'complete': False, 'next_epoch': start_epoch})
    try:
        for epoch in range(start_epoch, cfg['train']['epochs']):
            begin = time.perf_counter()
            print(f'[train] epoch {epoch + 1}/{cfg["train"]["epochs"]}; bad_epochs={bad_epochs}; patience={patience}', flush=True)
            model.train()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            total, n_samples, skipped = 0.0, 0, 0
            component_sums = {key: 0.0 for key in ['bce', 'dice', 'cldice', 'edge']}
            train_counts, gradient_norms = np.zeros(4, np.int64), []
            opt.zero_grad(set_to_none=True)
            accumulation = cfg['train']['accumulation_steps']
            lr_used = opt.param_groups[0]['lr']
            wait_seconds, previous_batch_end = 0.0, time.perf_counter()
            # Each accumulated group is normalized by its actual sample count,
            # including the final partial group and a final short minibatch.
            for step, batch in enumerate(loader):
                wait_seconds += time.perf_counter() - previous_batch_end
                artifacts.train_batch(batch, step, epoch)
                group_start = (step // accumulation) * accumulation
                group_end = min(group_start + accumulation, len(loader))
                group_samples = min(epoch_samples, group_end * cfg['train']['batch_size']) - group_start * cfg['train']['batch_size']
                x, y = batch['image'].to(device, non_blocking=True), batch['mask'].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
                    output = model(x)
                    loss, terms = criterion(output, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite train loss at epoch={epoch}, step={step}')
                scaler.scale(loss * x.shape[0] / group_samples).backward()
                if step + 1 == group_end:
                    scaler.unscale_(opt)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['clip_grad'])
                    gradient_norms.append(float(norm))
                    if step + 1 == len(loader):
                        artifacts.gradients(model, epoch)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                total += float(loss.detach()) * x.shape[0]
                n_samples += x.shape[0]
                skipped += terms['cldice_skipped']
                for key in component_sums:
                    component_sums[key] += terms[key] * x.shape[0]
                with torch.no_grad():
                    logits = output['logits'] if isinstance(output, dict) else output
                    pred, gt = logits.float().sigmoid() >= cfg['evaluation']['threshold'], y > 0.5
                    counts = torch.stack([(pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum(), (~pred & ~gt).sum()])
                    train_counts += counts.cpu().numpy()
                if (step + 1) % cfg['train'].get('log_every_steps', 20) == 0 or step + 1 == len(loader):
                    print(f'[train] epoch {epoch + 1} batch {step + 1}/{len(loader)}; loader_wait={wait_seconds:.1f}s', flush=True)
                previous_batch_end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - begin
            validation_begin = time.perf_counter()
            detailed = (epoch + 1) % cfg['evaluation'].get('train_detailed_every', 1) == 0 or epoch + 1 == cfg['train']['epochs']
            print(f'[validation] epoch {epoch + 1}; full source val; detailed_metrics={detailed}', flush=True)
            curves, previews = {}, []
            val, val_rows = evaluate(model, val_set, cfg, device, criterion=criterion,
                                    visualizer=artifacts.collect_preview(previews), curve_collector=curves,
                                    detailed=detailed, progress=True)
            validation_seconds = time.perf_counter() - validation_begin
            selection_iou = val['domain_mean']['iou'] if cfg.get('multidomain') else val['micro']['iou']
            improved = selection_iou > best
            best = max(best, selection_iou)
            bad_epochs = 0 if improved else bad_epochs + 1
            patience = cfg['train']['early_stopping_patience']
            final_epoch = epoch + 1 == cfg['train']['epochs'] or (patience and bad_epochs >= patience)
            scheduler.step()
            row = {'epoch': epoch, 'train_loss': total / n_samples, 'val_loss': val['val_loss'],
                   **val['micro'], 'lr': lr_used, 'seconds': time.perf_counter() - begin,
                   **{f'macro_{k}': v for k, v in val['macro'].items()},
                   **{f'train_{k}_loss': v / n_samples for k, v in component_sums.items()},
                   **{f'val_{k}_loss': v for k, v in val['loss_components'].items()},
                   **{f'loss_weight_{k}': cfg['loss'][k] for k in component_sums},
                   **{f'train_crop_{k}': v for k, v in pixel_scores(train_counts, cfg['evaluation']['empty_score']).items()
                      if k in ['iou', 'dice', 'precision', 'recall']},
                   **artifacts.parameters(model),
                   'grad_norm': float(np.mean(gradient_norms)), 'grad_norm_max': max(gradient_norms),
                   'train_cldice_skipped': skipped, 'val_cldice_skipped': val['cldice_skipped'],
                   'gpu_peak_mb': torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == 'cuda' else 0,
                   'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                   'loader_wait_seconds': wait_seconds, 'validation_cpu_metric_seconds': val.get('cpu_metric_seconds', 0),
                   'bad_epochs': bad_epochs, 'early_stopping_patience': patience}
            history.append(row)
            if cfg.get('multidomain'):
                row.update(domain_mean_iou=selection_iou, domain_mean_dice=val['domain_mean']['dice'],
                           domain_worst_iou=val['domain_worst_iou'],
                           **{f'domain/{name}/{key}': result['micro'][key] for name, result in val['domains'].items()
                              for key in ['iou', 'dice', 'precision', 'recall']})
            artifact_begin = time.perf_counter()
            print('[artifacts] saving checkpoints, metrics and scheduled plots (CPU/I/O).', flush=True)
            artifacts.write_epoch(history, val, val_rows, curves, previews, model, improved, final=bool(final_epoch))
            dump_json(out / 'validation_last.json', val)
            if improved:
                dump_json(out / 'validation_best.json', val)
                save_checkpoint(out / 'best_iou.pth', model, opt, scheduler, scaler, epoch, best, cfg, digest, generator, bad_epochs)
            save_checkpoint(out / 'last.pth', model, opt, scheduler, scaler, epoch, best, cfg, digest, generator, bad_epochs)
            options = cfg.get('logging', {})
            if options.get('plots', True) and (epoch == 0 or (epoch + 1) % options.get('plot_every', 5) == 0 or final_epoch):
                training_curves(pd.DataFrame(history), out / 'curves.png')
            artifact_seconds = time.perf_counter() - artifact_begin
            timing = {'epoch': epoch + 1, 'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                      'artifact_seconds': artifact_seconds, 'loader_wait_seconds': wait_seconds,
                      'epoch_seconds': time.perf_counter() - begin}
            pd.DataFrame([timing]).to_csv(out / 'timing.csv', index=False, mode='a', header=not (out / 'timing.csv').exists())
            print(f'Epoch {epoch + 1}/{cfg["train"]["epochs"]} loss={row["train_loss"]:.5f} selection_iou={selection_iou:.5f} '
                  f'train={train_seconds:.1f}s val={validation_seconds:.1f}s artifacts={artifact_seconds:.1f}s bad_epochs={bad_epochs}/{patience}', flush=True)
            if final_epoch:
                completed = True
                if patience and bad_epochs >= patience:
                    print(f'Early stopping: validation IoU did not improve for {bad_epochs} epochs; '
                          f'best_iou={best:.5f}. Using best_iou.pth.', flush=True)
                break
            if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
                break  # Deliberate interruption for epoch-boundary resume verification.
    finally:
        writer.close()
    dump_json(out / 'training_status.json', {'complete': completed, 'last_epoch': epoch, 'best_iou': best,
              'stopped_early': bool(completed and epoch + 1 < cfg['train']['epochs']),
              'early_stopping_patience': patience, 'bad_epochs': bad_epochs,
              'optimizer_steps': len(history) * cfg['train'].get('steps_per_epoch', (len(loader) + accumulation - 1) // accumulation)})
    return out / 'best_iou.pth'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--resume')
    initialization.add_argument('--init-checkpoint', help='Load compatible model weights only for a new fine-tuning experiment')
    parser.add_argument('--stop-after-epoch', type=int)
    parser.add_argument('--allow-resume-changes', action='store_true', help='Audit stopping/diagnostic-only configuration changes; keep all learning state')
    parser.add_argument('--reset-early-stopping', action='store_true', help='Explicitly restart the patience counter on resume, with an audit event')
    args = parser.parse_args()
    train(load_config(args.config), args.resume, args.stop_after_epoch, args.init_checkpoint,
          args.allow_resume_changes, args.reset_early_stopping)


if __name__ == '__main__':
    main()
