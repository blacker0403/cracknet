import argparse
import copy
import json
from pathlib import Path
import sys
import time

# Support both `python scripts/predict.py` and `python -m scripts.predict`.
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from datasets.manifest import EXTENSIONS, read_image
from datasets.multidomain import file_digest
from datasets.transforms import image_tensor
from models.cracktoponet import build_model
from utils.checkpoint import read_checkpoint
from utils.config import load_config, fingerprint
from utils.inference import sliding_window
from utils.logging_utils import get_device, dump_json
from utils.prediction_outputs import save_prediction_visualization, write_prediction_summary


def load_prediction_model(checkpoint, cfg, frozen, device):
    state = read_checkpoint(checkpoint)
    recorded = state['config']
    frozen_cfg = frozen.get('config', recorded)
    if (frozen['checkpoint_sha256'] != file_digest(Path(checkpoint)) or
            frozen['config_hash'] != fingerprint(frozen_cfg)
            or frozen.get('checkpoint_config_hash', frozen['config_hash']) != fingerprint(recorded)):
        raise ValueError('Prediction requires the original frozen checkpoint/config.')
    comparable, relocated = copy.deepcopy(frozen_cfg), {}
    for section, keys in [(None, ['output_dir']), ('data', ['root', 'manifest_dir']),
                          ('model', ['weights_dir', 'weights_file'])]:
        old, current = (comparable[section], cfg[section]) if section else (comparable, cfg)
        for key in keys:
            if old.get(key) != current.get(key):
                relocated[f'{section}.{key}' if section else key] = {'trained': old.get(key), 'current': current.get(key)}
                if key in current:
                    old[key] = current[key]
                else:
                    old.pop(key, None)
    if fingerprint(comparable) != fingerprint(cfg):
        raise ValueError('Prediction config differs from frozen training config beyond relocated storage paths.')
    model = build_model(recorded['model'], initialize=False).to(device)
    model.load_state_dict(state['model'], strict=True)
    return model.eval(), relocated


def prediction_inputs(source, dest):
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError('Output must be separate from input, never inside the source directory.')
    # A standard split root also contains GT masks; predict its images only.
    if (source / 'images').is_dir() and (source / 'masks').is_dir():
        source = source / 'images'
    inputs = (sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
              if source.is_dir() else [source] if source.is_file() and source.suffix.lower() in EXTENSIONS else [])
    if not inputs:
        raise ValueError(f'No readable image paths found under {source}.')
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f'Prediction output is not empty: {dest}; use a new directory to preserve existing results.')
    return source, dest, inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--amp-dtype', choices=['float16', 'bfloat16'],
                        help='Override inference autocast precision without changing the frozen training config.')
    args = parser.parse_args()
    source, dest, inputs = prediction_inputs(args.input, args.output)
    cfg = load_config(args.config)
    torch.set_num_threads(cfg['train']['cpu_threads'])
    frozen = json.loads((Path(cfg['output_dir']) / 'frozen.json').read_text())
    device = get_device(cfg['train']['device'])
    model, relocated = load_prediction_model(args.checkpoint, cfg, frozen, device)
    amp_dtype = args.amp_dtype or cfg['train'].get('amp_dtype', 'float16')
    record = {'complete': False, 'input': str(source), 'output': str(dest), 'expected_images': len(inputs),
              'checkpoint_sha256': frozen['checkpoint_sha256'], 'frozen_config_hash': frozen['config_hash'],
              'requested_config_hash': fingerprint(cfg), 'relocated_paths': relocated,
              'threshold': frozen['threshold'], 'threshold_selection_split': frozen['selection_split'],
              'window': cfg['evaluation']['window'], 'overlap': cfg['evaluation']['overlap'],
              'amp': cfg['train']['amp'], 'amp_dtype': amp_dtype,
              'device': str(device), 'torch': str(torch.__version__),
              'summary': 'prediction_summary.csv',
              'probability_interpretation': 'Model class probabilities; calibration and correctness are not measured.',
              'scope': 'image prediction only; no label metrics or threshold search'}
    dump_json(dest / 'prediction.json', record)
    print(f'Predicting {len(inputs)} images from {source}; threshold={frozen["threshold"]}; relocated paths={len(relocated)}', flush=True)
    started = time.perf_counter()
    summaries = []
    for index, path in enumerate(inputs, 1):
        rel = path.relative_to(source) if source.is_dir() else Path(path.name)
        original = read_image(path)
        image = image_tensor(original)
        inference_started = time.perf_counter()
        result = sliding_window(model, image, cfg['evaluation']['window'], cfg['evaluation']['overlap'], device, cfg['train']['amp'],
                                amp_dtype, cfg['evaluation'].get('window_batch_size', 1))
        inference_seconds = time.perf_counter() - inference_started
        # Retain original extension in the output ID to avoid same-stem collisions.
        output = dest / str(rel)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output) + '.probability.npy', result['probability'].numpy())
        cv2.imwrite(str(output) + '.mask.png', (result['probability'].numpy() >= frozen['threshold']).astype(np.uint8) * 255)
        if result['edge_probability'] is not None:
            cv2.imwrite(str(output) + '.edge.png', (result['edge_probability'].numpy() * 255).astype(np.uint8))
        stats = save_prediction_visualization(original, result['probability'].numpy(), frozen['threshold'],
                                            rel.as_posix(), output,
                                            edge=result['edge_probability'].numpy() if result['edge_probability'] is not None else None)
        summaries.append({'id': rel.as_posix(), **stats, 'inference_seconds': inference_seconds})
        if index == 1 or index % 50 == 0 or index == len(inputs):
            print(f'Predicted {index}/{len(inputs)} images', flush=True)
    write_prediction_summary(dest, summaries)
    record.update(complete=True, predicted_images=len(inputs), seconds=time.perf_counter() - started,
                  time_scope='Prediction plus artifact export; per-image inference_seconds excludes visualization.')
    dump_json(dest / 'prediction.json', record)
    print(f'Predicted {len(inputs)} native-resolution images at frozen threshold {frozen["threshold"]}')


if __name__ == '__main__':
    main()
