"""Optional pinned independent baseline; failures do not affect the native UNet."""
import importlib.metadata
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights


class DeepLabBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError('Optional baseline requires segmentation-models-pytorch==0.5.0') from exc
        if importlib.metadata.version('segmentation-models-pytorch') != '0.5.0':
            raise RuntimeError('DeepLab reproducibility requires segmentation-models-pytorch==0.5.0')
        self.net = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None,
                                    encoder_output_stride=16, in_channels=3, classes=1, activation=None)
        if cfg.get('weights_file') or cfg['pretrained']:
            # Share the exact torchvision encoder weights with the internal Baseline.
            try:
                state = (torch.load(cfg['weights_file'], map_location='cpu', weights_only=True) if cfg.get('weights_file') else
                         torch.hub.load_state_dict_from_url(ResNet34_Weights.IMAGENET1K_V1.url,
                             model_dir=str(Path(cfg['weights_dir'])), map_location='cpu', check_hash=True))
                self.net.encoder.load_state_dict(state)
            except Exception as exc:
                raise RuntimeError(f'DeepLab ResNet34 weights failed: {exc}; no random-initialization fallback.') from exc

    def forward(self, image):
        h, w = image.shape[-2:]
        padded = F.pad(image, (0, (-w) % 32, 0, (-h) % 32))
        return {'logits': self.net(padded)[..., :h, :w], 'edge_logits': None}
