"""Explicit pairing, decoded-content auditing, source-group leakage checks."""
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
FOLDER_DIGEST_PREFIX = 'folders-v1:'
MANIFEST_EXAMPLE = {'id': 'sample01', 'image': 'train/images/sample01.jpg',
                    'mask': 'train/masks/sample01.png', 'source_id': 'original01'}


def verify_manifest_paths(root, manifests):
    """Check deployment paths before decoding thousands of missing images."""
    root = Path(root).resolve()
    if not root.is_dir():
        available = sorted(p.name for p in root.parent.iterdir() if p.is_dir()) if root.parent.is_dir() else []
        raise ValueError(f'Dataset root missing: {root}. Directories in {root.parent}: {available}. '
                         'Linux paths are case-sensitive; check data.root and the extraction directory.')
    missing = [str(root / row[role]) for rows in manifests.values() for row in rows
               for role in ['image', 'mask'] if not (root / row[role]).is_file()]
    if missing:
        raise ValueError(f'{len(missing)} missing image/mask files under {root}; examples: {missing[:3]}. '
                         'Check directory casing and extraction depth before starting a GPU job.')


def read_image(path):
    # Annotations align with stored raster pixels, not a viewer's EXIF rotation.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f'Unreadable image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path, label_map=None):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f'Missing or unreadable mask: {path}')
    if mask.ndim != 2:
        raise ValueError(f'Expected readable single-channel mask: {path}')
    values = set(np.unique(mask).tolist())
    if label_map is None:
        if not (values <= {0, 1} or values <= {0, 255}):
            raise ValueError(f'Invalid mask values {sorted(values)[:30]} in {path}; set explicit data.label_map.')
        return (mask > 0).astype(np.uint8)
    mapping = {int(k): int(v) for k, v in label_map.items()}
    if not values <= mapping.keys() or not set(mapping.values()) <= {0, 1}:
        raise ValueError(f'Incomplete/nonbinary label_map for {path}: {values - mapping.keys()}')
    out = np.zeros(mask.shape, np.uint8)
    for key, value in mapping.items():
        out[mask == key] = value
    return out


def scan_split(root, split_cfg, pattern, image_glob='**/*', source_regex=None, *, check_pairs=True):
    root = Path(root).resolve()
    image_dir, mask_dir = root / split_cfg['images'], root / split_cfg['masks']
    images = sorted(p for p in image_dir.glob(image_glob) if p.suffix.lower() in EXTENSIONS and p.is_file())
    if not images:
        raise ValueError(f'No images in {image_dir} matching {image_glob!r}.')
    rows, used_masks = [], set()
    for image in images:
        rel = image.relative_to(image_dir)
        mask = mask_dir / pattern.format(stem=image.stem, name=image.name,
                                         relative_stem=str(rel.with_suffix('')), suffix=image.suffix)
        if check_pairs:
            if not mask.is_file():
                raise ValueError(f'Missing pair: {image} -> {mask}. Set mask_pattern or supply JSONL: {MANIFEST_EXAMPLE}')
            if mask.resolve() in used_masks:
                raise ValueError(f'Multiple images matched one mask: {mask}')
            used_masks.add(mask.resolve())
        match = re.fullmatch(source_regex, str(rel.with_suffix(''))) if source_regex else None
        if source_regex and (match is None or 'source' not in match.groupdict()):
            raise ValueError(f'source_id_regex must full-match and contain (?P<source>...): {rel}')
        rows.append({'id': rel.with_suffix('').as_posix(), 'image': image.relative_to(root).as_posix(),
                     'mask': mask.relative_to(root).as_posix(), 'source_id': match['source'] if match else None})
    if check_pairs:
        candidates = {p.resolve() for p in mask_dir.rglob('*') if p.suffix.lower() in EXTENSIONS}
        if mask_dir.resolve() == image_dir.resolve():
            candidates -= {p.resolve() for p in images}
        extra = candidates - used_masks
        if extra:
            raise ValueError(f'{len(extra)} unpaired masks, example: {next(iter(extra))}')
    return rows


def folder_samples(data):
    return {split: scan_split(data['root'], data['splits'][split], data['mask_pattern'],
                              data['image_glob'], check_pairs=False)
            for split in ('train', 'val', 'test')}


def folder_digest(data, samples):
    """Fingerprint folder entries using metadata, without opening sample files."""
    root = Path(data['root'])
    directories = {root / spec[role] for spec in data['splits'].values() for role in ('images', 'masks')}
    entries = []
    for directory in sorted(directories):
        for path in sorted(directory.rglob('*')):
            if path.suffix.lower() in EXTENSIONS and path.is_file():
                info = path.stat()
                entries.append((path.relative_to(root).as_posix(), info.st_size, info.st_mtime_ns))
    data_hash = hashlib.sha256(json.dumps([samples, entries], sort_keys=True).encode()).hexdigest()
    return FOLDER_DIGEST_PREFIX + data_hash


def verify_split_lists(root, manifests, split_lists, path_map=None):
    """Verify explicit image/mask pairs against supplied split text files."""
    root, path_map = Path(root), path_map or {}
    for split, filename in split_lists.items():
        pairs = []
        for line in (root / filename).read_text().splitlines():
            if not line.strip():
                continue
            tokens = line.split()
            if len(tokens) != 2:
                raise ValueError(f'{filename}: expected image_path mask_path, got {line!r}')
            mapped = []
            for token in tokens:
                path = Path(token)
                path = Path(path_map.get(path.parts[0], path.parts[0])).joinpath(*path.parts[1:])
                mapped.append(str((root / path).resolve()))
            pairs.append(tuple(mapped))
        actual = {(str((root / row['image']).resolve()), str((root / row['mask']).resolve())) for row in manifests[split]}
        if len(set(pairs)) != len(pairs) or set(pairs) != actual:
            raise ValueError(f'{filename}: scanned pairs differ from provided {split} split list.')


def content_hash(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def audit(root, manifests, label_map=None, require_source_ids=False):
    """Never accept duplicate paths/content, including duplicates within a split."""
    root = Path(root).resolve()
    report = {'splits': {}, 'samples': [], 'errors': [], 'source_audit': 'complete',
              'limitations': ['Exact decoded hashes do not detect near duplicates.']}
    paths, hashes, sources, content_records = {}, {}, {}, []
    for split, rows in manifests.items():
        if not rows:
            report['errors'].append(f'Empty split: {split}')
        values, sizes, fractions, ids = set(), set(), [], set()
        for row in rows:
            sample = {'split': split, 'id': row['id'], 'source_id': row.get('source_id'),
                      'height': None, 'width': None, 'foreground_fraction': None, 'mask_valid': False}
            if row['id'] in ids:
                report['errors'].append(f'Duplicate sample ID in {split}: {row["id"]}')
            ids.add(row['id'])
            for role in ['image', 'mask']:
                path = (root / row[role]).resolve()
                if path in paths:
                    report['errors'].append(f'Duplicate path: {path} ({paths[path]} / {split})')
                paths[path] = split
            try:
                image = read_image(root / row['image'])
                sample.update(height=image.shape[0], width=image.shape[1])
                sizes.add(tuple(image.shape[:2]))
                key = content_hash(image)
                if key in hashes:
                    report['errors'].append(f'Duplicate image content: {hashes[key]} / {split}:{row["id"]}')
                hashes[key] = f'{split}:{row["id"]}'
                raw = cv2.imread(str(root / row['mask']), cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    values.update(np.unique(raw).tolist())
                    content_records.append((split, row['id'], key, content_hash(raw)))
                mask = read_mask(root / row['mask'], label_map)
                if image.shape[:2] != mask.shape:
                    raise ValueError(f'Image/mask size mismatch: {row["id"]}')
                fractions.append(float(mask.mean()))
                sample.update(foreground_fraction=float(mask.mean()), mask_valid=True)
            except ValueError as exc:
                report['errors'].append(str(exc))
            report['samples'].append(sample)
            source = row.get('source_id')
            if not source:
                report['source_audit'] = 'incomplete: source IDs unavailable'
            elif source in sources and sources[source] != split:
                report['errors'].append(f'Source leakage: {source} ({sources[source]} / {split})')
            else:
                sources[source] = split
        report['splits'][split] = {'count': len(rows), 'sizes_hw': sorted(sizes),
                                  'raw_mask_values': sorted(values),
                                  'foreground_mean': float(np.mean(fractions)) if fractions else None,
                                  'foreground_min': min(fractions) if fractions else None,
                                  'foreground_max': max(fractions) if fractions else None}
    if 'incomplete' in report['source_audit']:
        report['limitations'].append('Original-image leakage cannot be excluded without verified source IDs.')
        if require_source_ids:
            report['errors'].append('Source IDs required but missing.')
    report['valid'] = not report['errors']
    report['content_digest'] = hashlib.sha256(json.dumps(content_records, sort_keys=True).encode()).hexdigest()
    return report


def write_manifests(manifests, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for split, rows in manifests.items():
        (directory / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


def load_manifests(directory):
    return {split: [json.loads(line) for line in (Path(directory) / f'{split}.jsonl').read_text().splitlines() if line.strip()]
            for split in ['train', 'val', 'test']}


def manifest_digest(manifests):
    return hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()
