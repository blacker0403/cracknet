"""YAML inheritance; paths are relative to the file which declares them."""
import copy
import hashlib
import json
from pathlib import Path

import yaml


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    parent = cfg.pop('extends', None)
    for section, keys in [(cfg, ['output_dir']), (cfg.get('data', {}), ['root', 'manifest_dir']),
                          (cfg.get('model', {}), ['weights_dir', 'weights_file'])]:
        for key in keys:
            if section.get(key):
                section[key] = str((path.parent / section[key]).resolve())
    return merge(load_config(path.parent / parent), cfg) if parent else cfg


def validate(cfg):
    model, loss, train = cfg['model'], cfg['loss'], cfg['train']
    dropout = model.get('dropout', 0.0)
    if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError('model.dropout must be a number in [0, 1).')
    if dropout and model['name'] != 'resunet34':
        raise ValueError('model.dropout is supported by the internal ResNet34/CrackTopoNet architecture only.')
    if loss['edge'] and not model['boundary_head']:
        raise ValueError('Edge Loss requires model.boundary_head=true; no silent disabling.')
    if model['fusion'] in {'bggf', 'bggf_uncentered', 'edge_concat'} and not model['boundary_head']:
        raise ValueError('Boundary fusion requires a predicted boundary head.')
    if model['name'] == 'deeplabv3plus' and (model['context'] != 'none' or model['fusion'] != 'concat' or model['boundary_head']):
        raise ValueError('DeepLabV3+ is an independent baseline, without internal UNet modules.')
    if model['name'] == 'deeplabv3plus' and model['norm'] != 'batch':
        raise ValueError('Pinned external DeepLab baseline uses BatchNorm; set model.norm=batch.')
    if model['name'] == 'linknet101' and (model['context'] != 'none' or model['fusion'] != 'add'
                                        or model['boundary_head'] or model['norm'] != 'batch'):
        raise ValueError('LinkNet101 requires context=none, fusion=add, boundary_head=false and norm=batch.')
    if train.get('optimizer', 'adamw') not in {'adamw', 'rmsprop'}:
        raise ValueError('Supported optimizers: adamw, rmsprop.')
    if train.get('scheduler', 'cosine') not in {'cosine', 'multistep'}:
        raise ValueError('Supported schedulers: cosine, multistep.')
    if train.get('scheduler') == 'multistep' and (not train.get('lr_milestones')
            or any(not isinstance(n, int) or n < 1 for n in train['lr_milestones'])
            or not 0 < train.get('lr_gamma', 0) < 1):
        raise ValueError('Multistep scheduler requires positive lr_milestones and lr_gamma in (0,1).')
    if train['batch_size'] < 1 or train['accumulation_steps'] < 1 or train['epochs'] < 1:
        raise ValueError('batch_size, accumulation_steps and epochs must be positive.')
    if not isinstance(train['early_stopping_patience'], int) or train['early_stopping_patience'] < 0:
        raise ValueError('early_stopping_patience must be a nonnegative integer (0 disables it).')
    if train.get('log_every_steps', 20) < 1 or cfg['data'].get('prefetch_factor', 2) < 1:
        raise ValueError('log_every_steps and prefetch_factor must be positive.')
    if (cfg['evaluation'].get('window_batch_size', 1) < 1
            or cfg['evaluation'].get('train_detailed_every', 1) < 1
            or cfg['evaluation'].get('progress_every', 100) < 1):
        raise ValueError('Window batch size and validation reporting intervals must be positive.')
    if train.get('amp_dtype', 'float16') not in {'float16', 'bfloat16'}:
        raise ValueError('amp_dtype must be float16 or bfloat16.')
    if cfg.get('multidomain'):
        from datasets.multidomain import METHODS
        m = cfg['multidomain']
        if (m['sampling'] not in METHODS or len(m['domains']) != len(set(m['domains']))
                or len(m['domains']) < 2 or m['held_out'] not in m['domains']):
            raise ValueError('Invalid multi-domain sampling/domains/held_out.')
        if not isinstance(train.get('steps_per_epoch'), int) or train['steps_per_epoch'] < 1:
            raise ValueError('Multi-domain training requires positive optimizer steps_per_epoch.')
    if any(loss[k] < 0 for k in ['bce', 'dice', 'cldice', 'edge']):
        raise ValueError('Loss weights must be nonnegative.')
    if cfg['data']['crop_size'] < 32:
        raise ValueError('ResNet encoder needs crop_size >= 32.')
    if not 0 <= cfg['data']['foreground_crop_probability'] <= 1:
        raise ValueError('foreground_crop_probability must be in [0, 1].')
    if not 0 <= cfg['evaluation']['overlap'] < cfg['evaluation']['window']:
        raise ValueError('Require 0 <= overlap < window.')
    logging = cfg.get('logging', {})
    if logging.get('plot_every', 5) < 1 or any(logging.get(k, 0) < 0 for k in ['train_batches', 'val_samples', 'histogram_every']):
        raise ValueError('logging.plot_every must be positive; sample counts and histogram_every must be nonnegative.')
    return cfg


def run_signature(cfg):
    return fingerprint({k: v for k, v in cfg.items() if k not in {'seed', 'output_dir', 'experiment'}})


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_config(cfg, path):
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
