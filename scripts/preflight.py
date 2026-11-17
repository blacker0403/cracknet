"""Report image counts, device and weights for the selected configuration."""
import argparse
import json
from pathlib import Path

import torch
from torchvision.models import ResNet34_Weights, ResNet101_Weights

from datasets.inspect_dataset import inspect
from datasets.manifest import folder_samples, load_manifests, audit, verify_split_lists, verify_manifest_paths
from utils.config import load_config, validate
from utils.logging_utils import environment, dump_json, get_device
from utils.artifacts import dataset_profile


def preflight(cfg, rescan=False, require_cuda=False, paths_only=False):
    validate(cfg)
    d = cfg['data']
    if not cfg.get('multidomain'):
        counts = {split: len(rows) for split, rows in folder_samples(d).items()}
        if paths_only:
            return {'data_paths_valid': True, 'data_root': d['root'], 'counts': counts,
                    'limitation': 'Image folder counts. Images and masks are decoded when samples are loaded.'}
        report = {'valid': True, 'errors': [], 'source_audit': 'folder mode',
                  'splits': {split: {'count': count} for split, count in counts.items()}}
    else:
        verify_manifest_paths(d['root'], {})
        if paths_only:
            manifests = load_manifests(d['manifest_dir'])
            verify_manifest_paths(d['root'], manifests)
            return {'data_paths_valid': True, 'data_root': d['root'],
                    'counts': {s: len(rows) for s, rows in manifests.items()},
                    'limitation': 'File existence only; no image decoding or CUDA check.'}
        if rescan or not all((Path(d['manifest_dir']) / f'{s}.jsonl').is_file() for s in ('train', 'val', 'test')):
            report = inspect(cfg)
        else:
            manifests = load_manifests(d['manifest_dir'])
            verify_manifest_paths(d['root'], manifests)
            if d.get('split_lists'):
                verify_split_lists(d['root'], manifests, d['split_lists'], d.get('split_path_map'))
            report = audit(d['root'], manifests, d['label_map'], d['require_source_ids'])
            dump_json(Path(d['manifest_dir']) / 'audit.json', report)
            dataset_profile(report, Path(d['manifest_dir']))
    m = cfg['model']
    encoder_weights = ResNet101_Weights if m['name'] == 'linknet101' else ResNet34_Weights
    weights = Path(m['weights_file']) if m.get('weights_file') else Path(m['weights_dir']) / Path(encoder_weights.IMAGENET1K_V1.url).name
    weights_ready = not (m['pretrained'] or m.get('weights_file')) or weights.is_file()
    device = get_device(cfg['train']['device'])
    cuda_ready = torch.cuda.is_available()
    device_ready = (device.type != 'cuda' or cuda_ready) and (not require_cuda or cuda_ready)
    result = {'ready_for_training': bool(report['valid'] and weights_ready and device_ready),
              'data_valid': report['valid'], 'data_errors': report['errors'][:10],
              'counts': {s: v['count'] for s, v in report['splits'].items()}, 'source_audit': report['source_audit'],
              'device': str(device), 'require_cuda': require_cuda, 'cuda_available': cuda_ready,
              'pretrained_weights_ready': weights_ready, 'weights_path': str(weights),
              'effective_batch_size': cfg['train']['batch_size'] * cfg['train']['accumulation_steps'],
              'environment': environment(), 'limitation': 'Readiness is not a full-batch CUDA memory stress test.'}
    dump_json(Path(cfg['output_dir']) / 'preflight.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--rescan', action='store_true', help='Rescan a legacy multi-domain dataset')
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--paths-only', action='store_true', help='List image counts without checking the device or weights')
    args = parser.parse_args()
    try:
        result = preflight(load_config(args.config), args.rescan, args.require_cuda, args.paths_only)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f'Preflight failed: {exc}\n')
    print(json.dumps(result, indent=2))
    if not result['data_paths_valid' if args.paths_only else 'ready_for_training']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
