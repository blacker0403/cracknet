"""Domain probabilities and immutable, source-disjoint LODO manifests."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler

from datasets.manifest import manifest_digest

DOMAINS = ['CRACK500', 'CrackSeg', 'DeepCrack', 'Forest', 'YCD']
METHODS = ['natural', 'sqrt']


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def domain_probabilities(rows, method):
    if method not in METHODS or not rows:
        raise ValueError('Require nonempty training rows and sampling=natural or sqrt.')
    counts = Counter(row['domain'] for row in rows)
    mass = {name: n ** (1.0 if method == 'natural' else .5) for name, n in sorted(counts.items())}
    return {name: value / sum(mass.values()) for name, value in mass.items()}


def domain_sampler(rows, method, num_samples, generator):
    counts = Counter(row['domain'] for row in rows)
    probabilities = domain_probabilities(rows, method)
    weights = [probabilities[row['domain']] / counts[row['domain']] for row in rows]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples,
                                 replacement=True, generator=generator)


def validate_fold(cfg, manifests, verify_sources=False):
    directory, root = Path(cfg['data']['manifest_dir']), Path(cfg['data']['root'])
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    receipt = json.loads((directory.parent.parent / 'preparation.json').read_text(encoding='utf-8'))
    m = cfg['multidomain']
    heldout = m['held_out']
    domains = m['domains']
    if (protocol != receipt['folds'].get(heldout) or protocol['held_out'] != heldout or receipt['domains'] != domains
            or protocol['manifest_hash'] != manifest_digest(manifests)):
        raise ValueError('LODO protocol/config/manifest mismatch; prepare again into a new directory.')
    inventory = directory.parent.parent / 'inventory.jsonl'
    if protocol['inventory_sha256'] != file_digest(inventory) or protocol['inventory_sha256'] != receipt['inventory_sha256']:
        raise ValueError('Preparation inventory fingerprint mismatch.')
    groups = {}
    for split, rows in manifests.items():
        expected = {heldout} if split == 'test' else set(domains) - {heldout}
        if {r['domain'] for r in rows} != expected:
            raise ValueError(f'{split}: missing source domain or held-out-domain contamination.')
        groups[split] = {r['source_id'] for r in rows}
    if any(groups[a] & groups[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Source/duplicate group crosses LODO splits.')
    if verify_sources:
        for line in inventory.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            for path, digest in record['files'].items():
                if file_digest(root / path) != digest:
                    raise ValueError(f'Prepared data/source changed: {path}')
    return protocol
