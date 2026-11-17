import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import torch


def environment():
    packages = {}
    for name in ['torch', 'torchvision', 'numpy', 'albumentations', 'opencv-python-headless', 'scikit-image',
                 'pandas', 'matplotlib', 'PyYAML', 'tensorboard', 'segmentation-models-pytorch']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
            'cuda_available': torch.cuda.is_available(), 'cuda_build': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def dump_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def get_device(requested):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else torch.device(requested)
