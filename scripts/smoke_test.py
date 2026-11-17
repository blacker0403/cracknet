"""Synthetic integration fixture. Never evidence for H1-H5 or Crack500 performance."""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from utils.config import load_config, merge, save_config
from utils.logging_utils import dump_json
from scripts.train import train
from scripts.evaluate import freeze, run_evaluation
from scripts.benchmark import run as benchmark_run
from scripts.summarize_results import summarize


def synthetic_dataset(root):
    root = Path(root)
    for split_index, (split, count) in enumerate([('train', 5), ('val', 3), ('test', 3)]):
        for index in range(count):
            rng = np.random.default_rng(split_index * 100 + index)
            h, w = (71, 93) if index % 2 == 0 else (83, 109)
            image = rng.integers(100, 160, (h, w, 3), dtype=np.uint8)
            mask = np.zeros((h, w), np.uint8)
            if index != 0:
                points = np.array([[8, 15], [w // 3, h // 2], [w // 2, h // 3], [w - 8, h - 10]], np.int32)
                points[:, 1] += split_index * 2
                cv2.polylines(mask, [points], False, 255, 1 + index % 2)
                image[mask > 0] = 35
            for role, data in [('images', image), ('masks', mask)]:
                path = root / split / role / f'synthetic_{split}_{index}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/smoke')
    parser.add_argument('--verify-existing', action='store_true', help='Re-evaluate existing checkpoints without retraining')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    data_root = root / 'synthetic_data'
    base = load_config('configs/baseline.yaml')
    base = merge(base, {'data': {'root': str(data_root),
        'version': 'synthetic_smoke_only', 'synthetic': True, 'crop_size': 64, 'mask_pattern': '{relative_stem}.png',
        'splits': {s: {'images': f'{s}/images', 'masks': f'{s}/masks'} for s in ['train', 'val', 'test']},
        'image_glob': '**/*'},
        'model': {'pretrained': False}, 'train': {'device': 'cpu', 'epochs': 1, 'batch_size': 2,
        'accumulation_steps': 2, 'amp': False, 'cpu_threads': 2},
        'evaluation': {'window': 64, 'overlap': 16, 'visualization_count': 2}})
    if not args.verify_existing:
        synthetic_dataset(data_root)
    for name, override in [('E0', {}), ('E8', {'model': {'context': 'mdcc', 'fusion': 'bggf', 'boundary_head': True},
                                               'loss': {'cldice': 0.2, 'edge': 0.1}, 'train': {'epochs': 2}})]:
        run_dir = root / f'{name}_seed42'
        if args.verify_existing:
            cfg_path = run_dir / 'config.yaml'
            cfg = load_config(cfg_path)
            checkpoint = run_dir / 'best_iou.pth'
        else:
            cfg = merge(base, override)
            cfg.update(experiment=name, output_dir=str(run_dir))
            cfg_path = root / f'{name}.yaml'
            save_config(cfg, cfg_path)
            if name == 'E8':
                train(cfg, stop_after_epoch=1)
                checkpoint = train(cfg, resume=run_dir / 'last.pth')
            else:
                checkpoint = train(cfg)
        run_evaluation(cfg, checkpoint, 'val')
        if not args.verify_existing:
            freeze(cfg, checkpoint)
        baseline = root / 'E0_seed42' / 'best_iou.pth' if name == 'E8' else None
        run_evaluation(cfg, checkpoint, 'test', baseline_checkpoint=baseline)
        for corruption in ['brightness', 'contrast', 'noise', 'blur']:
            run_evaluation(cfg, checkpoint, 'test', corruption)
        benchmark_run(cfg, checkpoint, warmup=1, repeats=3)
        subprocess.run([sys.executable, '-m', 'scripts.predict', '--config', str(cfg_path), '--checkpoint', str(checkpoint),
                        '--input', str(data_root / 'test' / 'images'), '--output', str(root / f'{name}_predictions')], check=True)
    summarize(root, root / 'summary', 'test')
    dump_json(root / 'smoke_status.json', {'passed': True, 'synthetic_only': True,
        'verified': ['dataset load', 'baseline train', 'full model train', 'epoch-boundary resume', 'validation threshold search',
                     'freeze', 'native test', 'baseline comparison panels', '4 robustness copies', 'predict', 'benchmark', 'summary'],
        'real_crack500_training': False})
    print(f'Synthetic smoke PASSED. Artifacts: {root}. No real Crack500 performance conclusion.')


if __name__ == '__main__':
    main()
