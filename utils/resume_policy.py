"""Explicitly audit stopping/diagnostic changes while retaining learning state."""
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets.multidomain import file_digest
from utils.config import fingerprint
from utils.logging_utils import dump_json

MUTABLE = {
    'train.early_stopping_patience', 'train.log_every_steps',
    'evaluation.window_batch_size', 'evaluation.train_detailed_every', 'evaluation.progress_every',
    'logging.plot_every', 'logging.histogram_every', 'logging.plots', 'logging.train_batches',
    'logging.val_samples', 'logging.render_on_improvement', 'data.prefetch_factor',
}


def config_changes(source, target, prefix=''):
    changes = []
    for key in sorted(source.keys() | target.keys()):
        name = f'{prefix}.{key}' if prefix else key
        a, b = source.get(key), target.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changes.extend(config_changes(a, b, name))
        elif a != b:
            changes.append({'field': name, 'before': a, 'after': b})
    return changes


def check_resume_config(source, target, allow_changes=False):
    changes = config_changes(source, target)
    if not changes:
        return []
    if not allow_changes:
        raise ValueError('Resume configuration differs; --allow-resume-changes permits audited stopping/diagnostic fields only.')
    forbidden = [row['field'] for row in changes if row['field'] not in MUTABLE]
    if forbidden:
        raise ValueError(f'Resume changes not permitted for learning/data fields: {forbidden}')
    return changes


def approved_checkpoint_config(source, target):
    if fingerprint(source) == fingerprint(target):
        return True
    try:
        check_resume_config(source, target, allow_changes=True)
    except ValueError:
        return False
    path = Path(target['output_dir']) / 'resume_changes.json'
    if not path.is_file():
        return False
    events = json.loads(path.read_text(encoding='utf-8'))['events']
    reachable = {fingerprint(source)}
    for event in events:
        if event['from_config_hash'] in reachable:
            reachable.add(event['to_config_hash'])
    return fingerprint(target) in reachable


def record_resume_change(source, target, checkpoint, epoch, bad_epochs, reset):
    changes = check_resume_config(source, target, allow_changes=True)
    path = Path(target['output_dir']) / 'resume_changes.json'
    record = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'events': []}
    record['events'].append({'utc': datetime.now(timezone.utc).isoformat(),
        'from_config_hash': fingerprint(source), 'to_config_hash': fingerprint(target),
        'changes': changes, 'checkpoint': str(Path(checkpoint).resolve()),
        'checkpoint_sha256': file_digest(checkpoint), 'last_completed_epoch': epoch + 1,
        'bad_epochs_before': bad_epochs, 'bad_epochs_reset': bool(reset),
        'state_preserved': ['model', 'optimizer', 'scheduler', 'scaler', 'rng', 'loader_rng', 'best_iou']})
    temporary = path.with_suffix('.tmp')
    dump_json(temporary, record)
    temporary.replace(path)
