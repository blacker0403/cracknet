"""Independent YOLO-style output conventions for binary semantic segmentation."""
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets.transforms import MEAN, STD
from utils.config import save_config
from utils.logging_utils import dump_json
from utils.visualization import save_figure


def results_table(history):
    frame = pd.DataFrame(history)
    aliases = {'train/loss': 'train_loss', 'val/loss': 'val_loss', 'lr/pg0': 'lr',
               'train/grad_norm_preclip': 'grad_norm', 'train/grad_norm_max_preclip': 'grad_norm_max',
               'system/gpu_peak_mb': 'gpu_peak_mb', 'system/epoch_seconds': 'seconds',
               'train/cldice_skipped': 'train_cldice_skipped', 'val/cldice_skipped': 'val_cldice_skipped'}
    aliases.update({f'system/{key}': key for key in ['train_seconds', 'validation_seconds', 'loader_wait_seconds', 'validation_cpu_metric_seconds']})
    for name in ['iou', 'dice', 'precision', 'recall', 'cldice', 'boundary_f1', 'pixel_accuracy',
                 'two_class_miou', 'auprc_hist_ap', 'skeleton_coverage']:
        aliases[f'metrics/{name}'] = name
        aliases[f'macro/{name}'] = f'macro_{name}'
    for name in ['bce', 'dice', 'cldice', 'edge']:
        aliases[f'train/{name}_loss'] = f'train_{name}_loss'
        aliases[f'val/{name}_loss'] = f'val_{name}_loss'
        aliases[f'loss_weight/{name}'] = f'loss_weight_{name}'
    for name in ['iou', 'dice', 'precision', 'recall']:
        aliases[f'train_crop/{name}'] = f'train_crop_{name}'
    for name in ['pred_components', 'gt_components', 'fragmentation_proxy']:
        aliases[f'macro/{name}'] = f'macro_{name}'
    out = pd.DataFrame({'epoch': frame['epoch'] + 1, 'time': frame['seconds'].cumsum()})
    for column, source in aliases.items():
        out[column] = pd.to_numeric(frame[source], errors='coerce') if source in frame else np.nan
    for column in frame:
        if column.startswith(('parameters/', 'domain/', 'domain_mean_', 'domain_worst_')):
            out[column] = frame[column]
    return out


def results_dashboard(history, path):
    frame = pd.DataFrame(history)
    panels = [('Total loss', ['train_loss', 'val_loss']), ('BCE loss', ['train_bce_loss', 'val_bce_loss']),
              ('Dice loss', ['train_dice_loss', 'val_dice_loss']), ('clDice loss', ['train_cldice_loss', 'val_cldice_loss']),
              ('Edge loss', ['train_edge_loss', 'val_edge_loss']),
              ('Validation domain mean / worst IoU', ['domain_mean_iou', 'domain_worst_iou']) if 'domain_mean_iou' in frame else
              ('Validation foreground IoU', ['iou', 'macro_iou']),
              ('Validation Dice/F1', ['domain_mean_dice', 'dice'] if 'domain_mean_dice' in frame else ['dice', 'macro_dice']),
              ('Validation precision', ['precision', 'macro_precision']),
              ('Validation recall', ['recall', 'macro_recall']), ('Validation hard clDice', ['cldice', 'macro_cldice']),
              ('Validation Boundary F1', ['boundary_f1', 'macro_boundary_f1']), ('Validation histogram AP', ['auprc_hist_ap']),
              ('Learning rate', ['lr']), ('Gradient norm before clip', ['grad_norm', 'grad_norm_max']),
              ('CUDA peak allocated MiB', ['gpu_peak_mb']), ('Train + validation seconds', ['seconds'])]
    fig, axes = plt.subplots(4, 4, figsize=(18, 13), constrained_layout=True)
    for ax, (title, keys) in zip(axes.flat, panels):
        for key, marker in zip(keys, ['o', 'x']):
            if key in frame and frame[key].notna().any():
                ax.plot(frame['epoch'] + 1, frame[key], marker=marker, markersize=3, label=key)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.2)
        if title.startswith('Validation'):
            ax.set_ylim(0, 1)
        if ax.lines:
            ax.legend(fontsize=7)
    save_figure(fig, path)


def parameter_dashboard(history, path):
    frame = pd.DataFrame(history)
    keys = [key for key in frame if key.startswith('parameters/')]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for key in keys:
        ax.plot(frame['epoch'] + 1, frame[key], marker='o', markersize=3, label=key)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learned scalar (alpha bounded to [0, 0.5])')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    save_figure(fig, path)


def save_diagnostics(directory, result, collector, render=True, title='Validation'):
    """Binned PR numeric data and native-scale confusion counts, never object mAP."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dump_json(directory / 'metrics.json', result)
    curves = collector['curves']
    pd.DataFrame(curves).to_csv(directory / 'curves.csv', index=False)
    np.savez_compressed(directory / 'curves.npz', **curves,
                        positive_histogram=collector['positive_histogram'],
                        negative_histogram=collector['negative_histogram'])
    tp, fp, fn, tn = result['confusion_tp_fp_fn_tn']
    matrix = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    pd.DataFrame(matrix, index=['GT_background', 'GT_crack'],
                 columns=['pred_background', 'pred_crack']).to_csv(directory / 'confusion_matrix.csv')
    if not render:
        return
    plots = [('PR_curve.png', curves['recall'], curves['precision'], 'Recall', 'Precision'),
             ('F1_curve.png', curves['threshold'], curves['f1'], 'Pixel probability threshold (binned)', 'Pixel F1'),
             ('P_curve.png', curves['threshold'], curves['precision'], 'Pixel probability threshold (binned)', 'Precision'),
             ('R_curve.png', curves['threshold'], curves['recall'], 'Pixel probability threshold (binned)', 'Recall')]
    for name, x, y, xlabel, ylabel in plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, color='#0072B2')
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1), title=title + ' — pixel ' + name.removesuffix('.png'))
        ax.grid(alpha=0.2)
        if name == 'PR_curve.png':
            ax.text(0.02, 0.04, f'Histogram AP={result["micro"]["auprc_hist_ap"]:.4f}', transform=ax.transAxes)
        save_figure(fig, directory / name)
    for normalized in [False, True]:
        values = matrix / np.maximum(matrix.sum(1, keepdims=True), 1) if normalized else matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized else max(1, int(matrix.max())))
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f'{values[row, col]:.3f}' if normalized else str(values[row, col]),
                        ha='center', va='center', color='black', bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'})
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Background', 'Crack'], yticklabels=['Background', 'Crack'],
               xlabel='Predicted pixel class', ylabel='GT pixel class',
               title=title + (' — GT-row normalized' if normalized else ' — pixel counts'))
        save_figure(fig, directory / ('confusion_matrix_normalized.png' if normalized else 'confusion_matrix.png'))


def mosaic(samples, path, prediction=False):
    if not samples:
        return
    columns = min(2, len(samples))
    fig, axes = plt.subplots(math.ceil(len(samples) / columns), columns, figsize=(columns * 5, math.ceil(len(samples) / columns) * 4), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, sample in zip(axes.flat, samples):
        image = sample['image']
        image = np.clip((image.numpy().transpose(1, 2, 0) * STD + MEAN), 0, 1)
        mask = sample['probability'] >= sample['threshold'] if prediction else sample['mask']
        ax.imshow(image)
        overlay = np.zeros((*mask.shape, 4))
        overlay[np.asarray(mask, bool)] = [0.90, 0.30, 0.10, 0.55] if prediction else [0.0, 0.60, 0.80, 0.55]
        ax.imshow(overlay)
        ax.set_title(sample['id'] + (' — prediction' if prediction else ' — GT'), fontsize=9)
    save_figure(fig, path)


def preview_arrays(samples, path):
    if not samples:
        return
    arrays = {}
    for i, sample in enumerate(samples):
        arrays.update({f'id_{i}': np.asarray(sample['id']), f'image_normalized_{i}': sample['image'].numpy(),
                       f'mask_{i}': sample['mask'], f'probability_{i}': sample['probability'],
                       f'threshold_{i}': np.asarray(sample['threshold'])})
        if sample['edge_probability'] is not None:
            arrays[f'edge_probability_{i}'] = sample['edge_probability']
    np.savez_compressed(path, **arrays)


def dataset_profile(report, directory):
    directory = Path(directory)
    frame = pd.DataFrame(report.get('samples', []))
    if frame.empty:
        return
    frame.to_csv(directory / 'dataset_stats.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    counts = frame.groupby('split').size()
    axes[0].bar(counts.index, counts.values, color='#0072B2')
    axes[0].set_title('Image counts (configured splits)')
    for split, group in frame.groupby('split'):
        valid = group['foreground_fraction'].dropna()
        if not valid.empty:
            axes[1].hist(valid, bins=np.linspace(0, 1, 21), histtype='step', label=split)
        axes[2].scatter(group['width'], group['height'], label=split, alpha=0.5, marker='x')
    axes[1].set(xlabel='Foreground fraction (valid masks only)', ylabel='Images', xlim=(0, 1))
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No valid mask encoding; foreground unknown', ha='center', transform=axes[1].transAxes)
    axes[2].set(xlabel='Stored raster width', ylabel='Stored raster height')
    axes[2].legend()
    save_figure(fig, directory / 'labels.jpg')


class TrainingArtifacts:
    def __init__(self, directory, cfg, writer):
        self.out, self.cfg, self.writer = Path(directory), cfg, writer
        self.options = {'plots': True, 'plot_every': 5, 'train_batches': 3, 'val_samples': 4,
                        'histogram_every': 10, **cfg.get('logging', {})}
        save_config(cfg, self.out / 'args.yaml')
        dump_json(self.out / 'results_metadata.json', {
            'task': 'binary semantic segmentation; foreground=crack', 'epoch_base': 1,
            'reference': 'Ultralytics v8.3.0 artifact conventions; independently implemented',
            'metrics': 'Validation micro and macro pixel metrics; not instance/box mAP',
            'checkpoint_selection': ('Source-domain mean IoU' if cfg.get('multidomain') else 'Validation foreground IoU')
                                    + f', fixed threshold {cfg["evaluation"]["threshold"]}',
            'early_stopping_patience': cfg['train']['early_stopping_patience'],
            'train_detailed_validation_every': cfg['evaluation'].get('train_detailed_every', 1),
            'unmeasured_diagnostics': 'null/empty CSV fields on lightweight validation epochs',
            'curve_definition': 'Quantized pixel probability PR/F1, same histogram bins as AUPRC',
            'losses': 'Raw terms; loss_weight/* gives the multipliers. Disabled clDice/edge terms are 0, not measurements.',
            'time': 'Cumulative train+validation seconds; excludes end-of-epoch checkpoint/plot export',
            'preview_rule': 'First validation IDs, fixed across epochs; no test predictions during training',
            'histograms': 'Representative first/last weight tensors; gradients after unscale and clipping',
            'normalization': {'mean': MEAN.tolist(), 'std': STD.tolist()},
            'cpu_gpu_memory': 'gpu_peak_mb=0 for CPU; not host RAM usage', 'synthetic': cfg['data']['synthetic']})

    def render_epoch(self, epoch):
        return self.options['plots'] and (epoch == 0 or (epoch + 1) % self.options['plot_every'] == 0
                                          or epoch + 1 == self.cfg['train']['epochs'])

    def train_batch(self, batch, step, epoch):
        if not self.options['plots'] or epoch != 0 or step >= self.options['train_batches']:
            return
        samples = [{'id': sid, 'image': x, 'mask': y[0].numpy()} for sid, x, y in
                   list(zip(batch['id'], batch['image'], batch['mask']))[:4]]
        mosaic(samples, self.out / f'train_batch{step}.jpg')
        np.savez_compressed(self.out / f'train_batch{step}.npz',
                            image_normalized=batch['image'][:4].numpy(), mask=batch['mask'][:4].numpy(),
                            ids=np.asarray(batch['id'][:4]))
        dump_json(self.out / f'train_batch{step}.json', {'epoch': 1, 'batch_index': step,
                  'ids': [s['id'] for s in samples], 'shape': list(batch['image'].shape),
                  'foreground_fraction': [float(s['mask'].mean()) for s in samples], 'augmented_training_crop': True})

    def collect_preview(self, samples):
        def collect(sample, output, scores):
            if len(samples) >= self.options['val_samples']:
                return
            samples.append({'id': sample['id'], 'image': sample['image'], 'mask': sample['mask'][0].numpy(),
                            'probability': output['probability'].numpy(),
                            'edge_probability': output['edge_probability'].numpy() if output['edge_probability'] is not None else None,
                            'threshold': self.cfg['evaluation']['threshold'], 'scores': scores})
        return collect

    @staticmethod
    def parameters(model):
        scalars = {}
        for name, module in model.named_modules():
            for key in ['gamma', 'alpha']:
                value = getattr(module, key, None)
                if value is not None:
                    scalars[f'parameters/{name}/{key}'] = float(value.detach())
        return scalars

    @staticmethod
    def representative_weights(model):
        weights = [(name, p) for name, p in model.named_parameters() if p.ndim > 1]
        return [weights[0], weights[-1]] if weights else []

    def histograms_due(self, epoch):
        every = self.options['histogram_every']
        return every > 0 and (epoch == 0 or (epoch + 1) % every == 0 or epoch + 1 == self.cfg['train']['epochs'])

    def gradients(self, model, epoch):
        if self.histograms_due(epoch):
            for name, parameter in self.representative_weights(model):
                if parameter.grad is not None:
                    self.writer.add_histogram(f'gradients_post_clip/{name}', parameter.grad.detach().float().cpu(), epoch + 1)

    def write_epoch(self, history, result, rows, curves, previews, model, improved, final=False):
        epoch = history[-1]['epoch']
        pd.DataFrame(history).to_csv(self.out / 'history.csv', index=False)
        table = results_table(history)
        table.to_csv(self.out / 'results.csv', index=False)
        for key, value in table.iloc[-1].items():
            if key != 'epoch' and np.isfinite(value):
                self.writer.add_scalar(key, value, epoch + 1)
        if self.histograms_due(epoch):
            for name, parameter in self.representative_weights(model):
                self.writer.add_histogram(f'weights/{name}', parameter.detach().float().cpu(), epoch + 1)
        self.writer.flush()
        render = self.render_epoch(epoch) or ((final or (improved and self.options.get('render_on_improvement', True))) and self.options['plots'])
        epoch_dir = self.out / 'validation' / f'epoch_{epoch + 1:03d}'
        save_diagnostics(epoch_dir, result, curves, render, f'Validation epoch {epoch + 1}, threshold={result["threshold"]}')
        pd.DataFrame(rows).to_csv(epoch_dir / 'per_image.csv', index=False)
        for name in ['curves.csv', 'curves.npz', 'confusion_matrix.csv']:
            shutil.copyfile(epoch_dir / name, self.out / name)
        if improved:
            best_dir = self.out / 'validation' / 'best'
            best_dir.mkdir(parents=True, exist_ok=True)
            for path in epoch_dir.iterdir():
                shutil.copyfile(path, best_dir / path.name)
            preview_arrays(previews, best_dir / 'preview_arrays.npz')
        if render:
            results_dashboard(history, self.out / 'results.png')
            parameter_dashboard(history, self.out / 'parameters.png')
            for path in epoch_dir.glob('*.png'):
                shutil.copyfile(path, self.out / path.name)
            mosaic(previews, self.out / 'val_batch0_labels.jpg')
            mosaic(previews, self.out / 'val_batch0_pred.jpg', prediction=True)
            preview_arrays(previews, self.out / 'val_preview_arrays.npz')
            dump_json(self.out / 'last_plot.json', {'epoch': epoch + 1, 'split': 'val', 'ids': [s['id'] for s in previews]})
            for i, sample in enumerate(previews):
                image = np.clip(sample['image'].numpy() * STD[:, None, None] + MEAN[:, None, None], 0, 1)
                self.writer.add_image(f'val/sample{i}/image', image, epoch + 1)
                self.writer.add_image(f'val/sample{i}/GT', sample['mask'][None], epoch + 1)
                self.writer.add_image(f'val/sample{i}/probability', sample['probability'][None], epoch + 1)
                if sample['edge_probability'] is not None:
                    self.writer.add_image(f'val/sample{i}/boundary', sample['edge_probability'][None], epoch + 1)
            self.writer.flush()
