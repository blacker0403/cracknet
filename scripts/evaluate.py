import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.crack500 import Crack500
from datasets.manifest import read_image
from models.cracktoponet import build_model
from scripts.train import checked_data
from utils.checkpoint import read_checkpoint
from utils.config import load_config, validate, fingerprint, run_signature
from utils.evaluation import evaluate
from utils.logging_utils import dump_json, get_device
from utils.visualization import prediction_panel
from utils.inference import sliding_window
from utils.robustness import corruption_transform
from utils.artifacts import save_diagnostics
from utils.resume_policy import approved_checkpoint_config


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_model(checkpoint, cfg, device):
    state = read_checkpoint(checkpoint)
    if not approved_checkpoint_config(state['config'], cfg):
        raise ValueError('Evaluation config must match checkpoint; use the saved run/config.yaml.')
    model = build_model(cfg['model'], initialize=False).to(device)
    model.load_state_dict(state['model'])
    return model.eval(), state


def freeze(cfg, checkpoint):
    out = Path(cfg['output_dir'])
    status = json.loads((out / 'training_status.json').read_text())
    if not status['complete']:
        raise ValueError('Finish the configured training budget before freezing.')
    if Path(checkpoint).resolve() != (out / 'best_iou.pth').resolve():
        raise ValueError('Freeze the validation-IoU selected best_iou.pth only.')
    if (out / 'frozen.json').exists():
        raise FileExistsError('frozen.json already exists; freeze is immutable for this run.')
    device = get_device(cfg['train']['device'])
    model, state = load_model(checkpoint, cfg, device)
    samples, digest = checked_data(cfg, state)
    if state['manifest_hash'] != digest:
        raise ValueError('Dataset changed since training.')
    d = cfg['data']
    dataset = Crack500(d['root'], samples['val'], d['label_map'])
    result, _ = evaluate(model, dataset, cfg, device, search=True)
    dump_json(out / 'threshold_validation.json', result)
    record = {'checkpoint_sha256': file_digest(checkpoint), 'manifest_hash': digest,
              'config_hash': fingerprint(cfg), 'threshold': result['threshold_search']['selected'],
              'checkpoint_config_hash': fingerprint(state['config']), 'config': cfg,
              'selection_split': 'val', 'checkpoint_selection': 'source-domain mean IoU' if cfg.get('multidomain') else 'validation foreground IoU',
              'threshold_objective': 'source-domain mean Dice' if cfg.get('multidomain') else 'validation micro F1', 'synthetic': d['synthetic']}
    if cfg.get('multidomain'):
        record.update(held_out=cfg['multidomain']['held_out'], sampling=cfg['multidomain']['sampling'],
                      source_domains=sorted(result['domains']))
    dump_json(out / 'frozen.json', record)
    return record


def run_evaluation(cfg, checkpoint, split='val', corruption=None, baseline_checkpoint=None):
    validate(cfg)
    torch.set_num_threads(cfg['train']['cpu_threads'])
    out = Path(cfg['output_dir'])
    device = get_device(cfg['train']['device'])
    model, state = load_model(checkpoint, cfg, device)
    samples, digest = checked_data(cfg, state)
    if state['manifest_hash'] != digest:
        raise ValueError('Dataset files changed since training.')
    frozen = None
    if split == 'test':
        path = out / 'frozen.json'
        if not path.exists():
            raise ValueError('Test requires --freeze first, after architecture/hyperparameters are selected on validation.')
        frozen = json.loads(path.read_text())
        if (frozen['checkpoint_sha256'] != file_digest(checkpoint) or frozen['manifest_hash'] != digest
                or frozen['config_hash'] != fingerprint(cfg)):
            raise ValueError('Frozen model/data/config fingerprint mismatch.')
    if corruption and split != 'test':
        raise ValueError('Robustness corruptions are test-only and cannot select a model.')
    threshold = frozen['threshold'] if frozen else cfg['evaluation']['threshold']
    d = cfg['data']
    dataset = Crack500(d['root'], samples[split], d['label_map'])
    transform = corruption_transform(corruption, cfg['seed']) if corruption else None
    curve_data = {}
    result, rows = evaluate(model, dataset, cfg, device, threshold=threshold, search=split == 'val',
                            corruption=transform, curve_collector=curve_data)
    if split == 'val':
        # Report the searched threshold separately; fixed .5 metrics remain auditable.
        selected = result['threshold_search']['selected']
        selected_result, rows = evaluate(model, dataset, cfg, device, threshold=selected, curve_collector=curve_data)
        selected_result['threshold_search'] = result['threshold_search']
        result = selected_result
        threshold = selected
    result.update(split=split, synthetic=d['synthetic'], experiment=cfg['experiment'], seed=cfg['seed'],
                  manifest_hash=digest, checkpoint_epoch=state['epoch'], checkpoint_sha256=file_digest(checkpoint),
                  corruption=corruption, dataset_version=d['version'], protocol_hash=run_signature(cfg))
    if cfg.get('multidomain'):
        result.update(held_out=cfg['multidomain']['held_out'], sampling=cfg['multidomain']['sampling'],
                      evaluation_scope='unseen-domain original test' if split == 'test' else 'source-domain validation')
    stem = f'{split}_{corruption}' if corruption else split
    dump_json(out / f'{stem}_metrics.json', result)
    pd.DataFrame(rows).to_csv(out / f'{stem}_per_image.csv', index=False)
    save_diagnostics(out / 'diagnostics' / stem, result, curve_data, title=f'{stem}, threshold={threshold}')
    if 'threshold_search' in result:
        pd.DataFrame(result['threshold_search']['table']).to_csv(out / 'diagnostics' / stem / 'threshold_search.csv', index=False)
    # Include best and worst IoU samples, deterministically; retain source IDs.
    ranked = sorted(rows, key=lambda r: (r['iou'], r['id']))
    count = cfg['evaluation']['visualization_count']
    selected_ids = {r['id'] for r in (ranked[:(count + 1) // 2] + ranked[-count // 2:])} if count else set()
    baseline_model, baseline_threshold = None, None
    if baseline_checkpoint:
        bc = read_checkpoint(baseline_checkpoint)['config']
        bf = json.loads((Path(bc['output_dir']) / 'frozen.json').read_text())
        if bf['checkpoint_sha256'] != file_digest(baseline_checkpoint) or bf['manifest_hash'] != digest:
            raise ValueError('Baseline comparison requires matching data and frozen checkpoint.')
        if bc['evaluation'] != cfg['evaluation']:
            raise ValueError('Baseline comparison requires identical evaluation protocol.')
        baseline_model, _ = load_model(baseline_checkpoint, bc, device)
        baseline_threshold = bf['threshold']
    for index, row in enumerate(dataset.rows):
        if row['id'] not in selected_ids:
            continue
        sample = dataset[index]
        x = transform(sample['image'], sample['id']) if transform else sample['image']
        pred = sliding_window(model, x, cfg['evaluation']['window'], cfg['evaluation']['overlap'], device, cfg['train']['amp'],
                              cfg['train'].get('amp_dtype', 'float16'), cfg['evaluation'].get('window_batch_size', 1))
        baseline = None
        if baseline_model is not None:
            baseline = sliding_window(baseline_model, x, cfg['evaluation']['window'], cfg['evaluation']['overlap'], device,
                                      bc['train']['amp'], bc['train'].get('amp_dtype', 'float16'),
                                      bc['evaluation'].get('window_batch_size', 1))['probability'].numpy() >= baseline_threshold
        image = read_image(Path(d['root']) / row['image'])
        if transform:
            from datasets.transforms import MEAN, STD
            image = np.clip((x.numpy().transpose(1, 2, 0) * STD + MEAN) * 255, 0, 255).astype(np.uint8)
        edge = pred['edge_probability'].numpy() if pred['edge_probability'] is not None else None
        prediction_panel(image, sample['mask'][0].numpy(), pred['probability'].numpy(), threshold,
                         sample['id'], out / 'visualizations' / stem / f'{sample["id"]}.png', edge, baseline)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--freeze', action='store_true', help='Freeze selected architecture, checkpoint and val threshold before test')
    parser.add_argument('--corruption', choices=['brightness', 'contrast', 'noise', 'blur'])
    parser.add_argument('--baseline-checkpoint')
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(cfg['train']['cpu_threads'])
    if args.freeze:
        if args.split != 'val' or args.corruption:
            parser.error('--freeze calibrates validation only; omit --split test and --corruption')
        print(json.dumps(freeze(cfg, args.checkpoint), indent=2))
    else:
        print(json.dumps(run_evaluation(cfg, args.checkpoint, args.split, args.corruption, args.baseline_checkpoint), indent=2))


if __name__ == '__main__':
    main()
