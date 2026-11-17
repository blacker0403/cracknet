"""Random-initialization architecture inspection; never loads data or checkpoints."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from models.cracktoponet import build_model
from utils.config import load_config, validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/tutorial/local_e8.yaml')
    p.add_argument('--height', type=int, default=384)
    p.add_argument('--width', type=int, default=384)
    p.add_argument('--output')
    args = p.parse_args()
    cfg = validate(load_config(args.config))
    torch.set_num_threads(4)
    model = build_model(cfg['model'], initialize=False).eval()
    records = []
    handles = []
    def hook(name):
        def capture(module, inputs, output):
            records.append({'module': name, 'input': [list(x.shape) for x in inputs if isinstance(x, torch.Tensor)],
                            'output': list(output.shape)})
        return capture
    selected = {'encoder.relu', 'encoder.layer1', 'encoder.layer2', 'encoder.layer3', 'encoder.layer4',
                'bottleneck_dropout', 'boundary', 'final', 'head'}
    for name, layer in model.named_modules():
        if name in selected or name.startswith(('contexts.', 'gates.', 'blocks.')) and name.count('.') == 1:
            handles.append(layer.register_forward_hook(hook(name)))
    with torch.no_grad():
        output = model(torch.randn(1, 3, args.height, args.width))
    for handle in handles:
        handle.remove()
    report = {'scope': 'random-initialization structure only; no trained feature interpretation',
              'input': [1, 3, args.height, args.width], 'parameters': sum(p.numel() for p in model.parameters()),
              'model_config': cfg['model'], 'stages': records,
              'output': {k: list(v.shape) if v is not None else None for k, v in output.items()}}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
