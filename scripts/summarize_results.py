import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.visualization import save_figure


def summarize(root, output, split='val'):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, complexity, robustness = [], [], []
    for path in sorted(root.rglob(f'{split}_metrics.json')):
        result = json.loads(path.read_text())
        rows.append({'experiment': result['experiment'], 'seed': result['seed'], 'split': result['split'],
                     'synthetic': result['synthetic'], 'dataset_version': result['dataset_version'],
                     'protocol_hash': result['protocol_hash'],
                     'manifest_hash': result['manifest_hash'], 'threshold': result['threshold'],
                     **{f'micro_{k}': v for k, v in result['micro'].items()},
                     **{f'macro_{k}': v for k, v in result['macro'].items()}, 'run_dir': str(path.parent)})
        b = path.parent / 'benchmark.json'
        if b.exists():
            value = json.loads(b.read_text())
            complexity.append({'experiment': result['experiment'], 'seed': result['seed'],
                               **{k: value[k] for k in ['parameters', 'conv_linear_macs', 'latency_ms_median', 'fps_median', 'device', 'amp', 'cpu_threads']},
                               'input_shape': str(value['input_shape']), 'scope': value['scope']})
        if split == 'test':
            for corruption in ['brightness', 'contrast', 'noise', 'blur']:
                p = path.parent / f'test_{corruption}_metrics.json'
                if p.exists():
                    r = json.loads(p.read_text())
                    robustness.append({'experiment': result['experiment'], 'seed': result['seed'], 'corruption': corruption,
                                       'clean_iou': result['micro']['iou'], 'corrupted_iou': r['micro']['iou'],
                                       'iou_drop': result['micro']['iou'] - r['micro']['iou'],
                                       'cldice_drop': result['micro']['cldice'] - r['micro']['cldice']})
    columns = ['experiment', 'seed', 'split', 'synthetic', 'dataset_version', 'manifest_hash', 'threshold',
               'micro_iou', 'micro_dice', 'micro_precision', 'micro_recall', 'micro_cldice', 'micro_boundary_f1', 'run_dir']
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    frame.to_csv(output / 'ablation_results.csv', index=False)
    frame[frame['experiment'].str.startswith('S_')].to_csv(output / 'loss_sensitivity.csv', index=False)
    pd.DataFrame(complexity, columns=['experiment', 'seed', 'parameters', 'conv_linear_macs', 'latency_ms_median',
                                    'fps_median', 'device', 'amp', 'cpu_threads', 'input_shape', 'scope']).to_csv(output / 'model_complexity.csv', index=False)
    if robustness:
        pd.DataFrame(robustness).to_csv(output / 'robustness_results.csv', index=False)
    text = [f'# {split} results', '', 'Synthetic and real results are separate. std uses ddof=1; n=1 has no std.', '',
            '| experiment | synthetic | n | IoU mean +/- std | Dice mean +/- std | clDice mean +/- std |',
            '|---|---|---:|---|---|---|']
    if not frame.empty:
        # Never pool different datasets, split protocols, or synthetic/real evidence.
        groups = frame.groupby(['experiment', 'synthetic', 'dataset_version', 'manifest_hash', 'protocol_hash'], dropna=False)
        grouped = []
        for (name, synthetic, version, digest, protocol_hash), group in groups:
            if group['seed'].duplicated().any():
                raise ValueError(f'Duplicate seed evidence for {name}; choose a narrower results root.')
            cells = []
            for metric in ['micro_iou', 'micro_dice', 'micro_cldice']:
                cells.append(f'{group[metric].mean():.4f} +/- {group[metric].std():.4f}' if len(group) > 1 else f'{group[metric].mean():.4f} (std N/A)')
            text.append(f'| {name} | {synthetic} | {len(group)} | ' + ' | '.join(cells) + ' |')
            grouped.append({'label': f'{name}\nsynthetic={synthetic}\n{version}', 'mean': group['micro_iou'].mean(),
                            'std': group['micro_iou'].std() if len(group) > 1 else 0, 'values': group['micro_iou'].tolist()})
        fig, ax = plt.subplots(figsize=(max(9, len(grouped) * 0.65), 5))
        x = np.arange(len(grouped))
        ax.errorbar(x, [g['mean'] for g in grouped], yerr=[g['std'] for g in grouped], fmt='o', capsize=4, color='#0072B2')
        for i, g in enumerate(grouped):
            ax.scatter(np.repeat(i, len(g['values'])), g['values'], marker='x', color='#D55E00')
        ax.set_xticks(x, [g['label'] for g in grouped], rotation=60, ha='right')
        ax.set_ylabel('Foreground IoU (points: individual seeds; bars: sample std)')
        save_figure(fig, output / 'ablation_seed_distribution.png')
    failures = list(root.rglob('failure.json'))
    text += ['', f'Recorded failures: {len(failures)}. See failure JSONs; they are not discarded.']
    text += [f'- {p}: {json.loads(p.read_text())["traceback"].splitlines()[-1]}' for p in failures]
    (output / 'ablation_results.md').write_text('\n'.join(text) + '\n')
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    args = parser.parse_args()
    frame = summarize(args.root, args.output, args.split)
    print(f'Summarized {len(frame)} runs; no missing results were invented.')


if __name__ == '__main__':
    main()
