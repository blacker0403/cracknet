import argparse
import json
from pathlib import Path

from datasets.manifest import scan_split, audit, write_manifests, load_manifests, read_image, read_mask, verify_split_lists
from utils.config import load_config


def inspect(cfg, existing=False):
    d = cfg['data']
    manifests = load_manifests(d['manifest_dir']) if existing or d.get('fixed_manifests', False) else {
        split: scan_split(d['root'], spec, d['mask_pattern'], d['image_glob'], d['source_id_regex'])
        for split, spec in d['splits'].items()}
    if d.get('split_lists'):
        verify_split_lists(d['root'], manifests, d['split_lists'], d.get('split_path_map'))
    report = audit(d['root'], manifests, d['label_map'], d['require_source_ids'])
    report.update(dataset_version=d['version'], synthetic=d['synthetic'], source_id_regex=d['source_id_regex'],
                  source_provenance='Configured IDs; official source provenance is not independently verified by this tool.')
    output = Path(d['manifest_dir'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    from utils.artifacts import dataset_profile
    dataset_profile(report, output)
    # Save invalid manifests for diagnosis too; train independently re-audits and refuses them.
    write_manifests(manifests, output)
    from utils.visualization import dataset_preview
    for split, rows in manifests.items():
        for row in rows[:3]:
            try:
                dataset_preview(read_image(Path(d['root']) / row['image']),
                    read_mask(Path(d['root']) / row['mask'], d['label_map']), row['id'],
                    output / 'samples' / split / f'{row["id"]}.png')
            except ValueError:
                pass  # full reason already appears in audit.errors
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--existing-manifests', action='store_true')
    args = parser.parse_args()
    report = inspect(load_config(args.config), args.existing_manifests)
    print(json.dumps({k: v for k, v in report.items() if k not in {'errors', 'samples'}}, indent=2))
    if not report['valid']:
        print('\n'.join(report['errors'][:20]))
        raise SystemExit(f'Audit FAILED: {len(report["errors"])} errors. See manifest_dir/audit.json.')


if __name__ == '__main__':
    main()
