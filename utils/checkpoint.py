"""Epoch-boundary resume. Checkpoints include trusted local Python/NumPy RNG objects."""
import hashlib
import os
from pathlib import Path
import torch
from utils.seed import rng_state, restore_rng


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_iou, cfg,
                    manifest_hash, generator, bad_epochs=0):
    path = Path(path)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'scaler': scaler.state_dict(), 'epoch': epoch, 'best_iou': best_iou, 'config': cfg,
             'seed': cfg['seed'], 'manifest_hash': manifest_hash, 'rng': rng_state(),
             'loader_rng': generator.get_state(), 'bad_epochs': bad_epochs}
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def read_checkpoint(path):
    # Only load checkpoints produced by this project / from a trusted source.
    return torch.load(path, map_location='cpu', weights_only=False)


def initialize_from_checkpoint(model, path, cfg):
    """Load model weights only; the caller creates fresh optimizer/scheduler state."""
    path = Path(path)
    state = read_checkpoint(path)
    ignored = {'pretrained', 'weights_dir', 'weights_file'}
    source = {k: v for k, v in state['config']['model'].items() if k not in ignored}
    target = {k: v for k, v in cfg['model'].items() if k not in ignored}
    if source != target:
        raise ValueError('Initial checkpoint architecture differs from the target model configuration.')
    model.load_state_dict(state['model'], strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'checkpoint': str(path.resolve()), 'checkpoint_sha256': digest.hexdigest(),
            'source_epoch': state.get('epoch'), 'source_best_iou': state.get('best_iou'),
            'source_experiment': state['config'].get('experiment'),
            'model_weights_only': True, 'optimizer_restored': False, 'scheduler_restored': False,
            'epoch_reset': True, 'best_metric_reset': True}


def resume_checkpoint(state, model, optimizer, scheduler, scaler, generator):
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    generator.set_state(state['loader_rng'])
    restore_rng(state['rng'])
    return state['epoch'] + 1, state['best_iou'], state.get('bad_epochs', 0)
