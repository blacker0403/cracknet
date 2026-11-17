import argparse
import json
from pathlib import Path
import torch
from models.cracktoponet import build_model
from scripts.evaluate import load_model
from utils.config import load_config
from utils.complexity import complexity, benchmark
from utils.logging_utils import get_device, dump_json, environment


def run(cfg, checkpoint=None, warmup=5, repeats=30):
    torch.set_num_threads(cfg['train']['cpu_threads'])
    device = get_device(cfg['train']['device'])
    model = load_model(checkpoint, cfg, device)[0] if checkpoint else build_model(cfg['model'], initialize=False).to(device)
    shape = (1, 3, cfg['data']['crop_size'], cfg['data']['crop_size'])
    result = {**complexity(model, shape, device), **benchmark(model, shape, device, warmup, repeats, cfg['train']['amp']),
              'environment': environment(), 'checkpoint': str(checkpoint) if checkpoint else 'random weights; structural benchmark only'}
    dump_json(Path(cfg['output_dir']) / 'benchmark.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config), args.checkpoint, args.warmup, args.repeats), indent=2))


if __name__ == '__main__':
    main()
