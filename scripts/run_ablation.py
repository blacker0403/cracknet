import argparse
import json
from pathlib import Path
import traceback
import yaml

from utils.config import load_config, merge, validate, save_config, fingerprint
from utils.logging_utils import dump_json
from scripts.train import train
from scripts.evaluate import run_evaluation
from scripts.benchmark import run as benchmark_run


def experiment_definitions(matrix):
    definitions = dict(matrix['experiments'])
    s = matrix.get('sensitivity', {})
    for key in ['cldice', 'edge', 'skeleton_iterations']:
        for value in s.get(key, []):
            definitions[f'S_{key}_{value}'] = merge(s['structure'], {'loss': {key: value}})
    return definitions


def protocol(cfg):
    return {'data': cfg['data'], 'train': cfg['train'], 'evaluation': cfg['evaluation'],
            'encoder': {k: cfg['model'][k] for k in ['pretrained', 'weights_file', 'norm']},
            'pos_weight': cfg['loss']['pos_weight']}


def select_e12(root):
    root = Path(root)
    candidates = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']
    scores, protocols = {}, []
    for name in candidates:
        run = root / f'{name}_seed42'
        result = json.loads((run / 'val_metrics.json').read_text())
        if result['split'] != 'val' or result['seed'] != 42:
            raise ValueError('E12 selection requires completed development validation (seed 42).')
        cfg = load_config(run / 'config.yaml')
        protocols.append(fingerprint(protocol(cfg)))
        scores[name] = result['micro']['iou']
    if len(set(protocols)) != 1:
        raise ValueError('Development experiments used different training/data/evaluation protocols.')
    selection = {'selection_split': 'val', 'development_seed': 42,
                 'best_single': max(candidates[:3], key=lambda n: scores[n]),
                 'best_combination': max(candidates[3:], key=lambda n: scores[n]),
                 'validation_iou': scores, 'tie_rule': 'first experiment in E order',
                 'note': 'E12 is a comparison of these two existing designs, not a new module or test-based selection.'}
    dump_json(root / 'E12_selection.json', selection)
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', default='configs/ablation.yaml')
    parser.add_argument('--base-config', help='Override base, e.g. local/synthetic configuration')
    parser.add_argument('--output-root', help='Use a separate result root for a dataset protocol')
    parser.add_argument('--experiments', default='E0', help='Comma-separated IDs, core, sensitivity, or all')
    parser.add_argument('--stage', choices=['development', 'formal'], default='development')
    parser.add_argument('--seeds', help='Comma-separated seeds; defaults to stage seeds from YAML')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--select-e12', action='store_true')
    args = parser.parse_args()
    matrix_path = Path(args.matrix).resolve()
    matrix = yaml.safe_load(matrix_path.read_text())
    base = load_config(args.base_config or matrix_path.parent / matrix['base'])
    if (base['model']['context'] != 'none' or base['model']['fusion'] not in {'concat', 'add'}
            or base['model']['boundary_head'] or base['loss']['cldice'] or base['loss']['edge']):
        raise ValueError('Ablation base must be the plain Baseline without optional modules/losses.')
    output_root = (Path(args.output_root).resolve() if args.output_root else
                   Path(base['output_dir']).parent if args.base_config else
                   (matrix_path.parent / matrix['output_root']).resolve())
    definitions = experiment_definitions(matrix)
    if args.select_e12:
        print(json.dumps(select_e12(output_root), indent=2))
        return
    names = (list(definitions) if args.experiments == 'all' else [f'E{i}' for i in range(11)] if args.experiments == 'core'
             else [n for n in definitions if n.startswith('S_')] if args.experiments == 'sensitivity' else args.experiments.split(','))
    if base['model']['name'] == 'linknet101' and names != ['E0']:
        raise ValueError('LinkNet baseline_v2 supports --experiments E0 only; the core matrix is for ResUNet34.')
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else matrix[f'{args.stage}_seeds']
    if args.stage == 'formal':
        selection = json.loads((output_root / 'E12_selection.json').read_text())
        required = {'E0', selection['best_single'], 'E7', 'E8', 'E9', 'E10'}
        if not required <= set(names) or not {42, 123, 3407} <= set(seeds):
            raise ValueError(f'Formal stage requires {sorted(required)} and seeds 42,123,3407.')
    for name in names:
        if name not in definitions:
            raise ValueError(f'Unknown experiment: {name}. E12 uses --select-e12 after E1-E8 validation.')
    for seed in seeds:
        for name in names:
            definition = definitions[name]
            if 'alias_of' in definition:
                dump_json(output_root / f'{name}_seed{seed}' / 'alias.json',
                          {'alias_of': f'{definition["alias_of"]}_seed{seed}', 'reason': 'E11 and E5 are identical in the requested matrix; not independent evidence.'})
                continue
            cfg = merge(base, definition)
            cfg.update(experiment=name, seed=seed, output_dir=str(output_root / f'{name}_seed{seed}'))
            validate(cfg)
            if protocol(cfg) != protocol(base):
                raise ValueError(f'{name} changes the shared experimental protocol.')
            out = Path(cfg['output_dir'])
            out.mkdir(parents=True, exist_ok=True)
            if args.dry_run:
                save_config(cfg, out / 'planned_config.yaml')
                dump_json(out / 'planned.json', {'stage': args.stage, 'executed': False, 'config_hash': fingerprint(cfg)})
                print(f'PLANNED {name} seed={seed}: {out}')
                continue
            try:
                resume = out / 'last.pth' if args.resume and (out / 'last.pth').exists() else None
                status_path = out / 'training_status.json'
                complete = status_path.exists() and json.loads(status_path.read_text())['complete']
                if complete and args.resume:
                    saved_cfg = load_config(out / 'config.yaml')
                    if saved_cfg != cfg:
                        raise ValueError('Existing completed run config differs.')
                    checkpoint = out / 'best_iou.pth'
                else:
                    checkpoint = train(cfg, resume)
                run_evaluation(cfg, checkpoint, 'val')
                benchmark_run(cfg, checkpoint)
                dump_json(out / 'execution.json', {'stage': args.stage, 'complete': True, 'test_executed': False})
            except Exception:
                dump_json(out / 'failure.json', {'stage': args.stage, 'traceback': traceback.format_exc()})
                raise


if __name__ == '__main__':
    main()
