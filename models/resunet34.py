"""Shared ResNet34 encoder/decoder for every internal ablation."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34, ResNet34_Weights


def norm_layer(channels, kind='batch'):
    if kind == 'batch':
        return nn.BatchNorm2d(channels)
    if kind == 'group':
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f'Unknown normalization: {kind}')


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, norm='batch'):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), norm_layer(cout, norm), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), norm_layer(cout, norm), nn.ReLU(inplace=True))


class ResUNet34(nn.Module):
    def __init__(self, pretrained=True, weights_dir='weights', weights_file=None, norm='batch'):
        super().__init__()
        # Load weights explicitly so failed downloads cannot silently change the experiment.
        self.encoder = resnet34(weights=None)
        if weights_file or pretrained:
            try:
                state = (torch.load(weights_file, map_location='cpu', weights_only=True) if weights_file else
                         torch.hub.load_state_dict_from_url(ResNet34_Weights.IMAGENET1K_V1.url,
                             model_dir=str(Path(weights_dir)), map_location='cpu', check_hash=True))
                self.encoder.load_state_dict(state)
            except Exception as exc:
                raise RuntimeError(f'ResNet34 weights failed: {exc}. Supply model.weights_file or explicitly set pretrained=false.') from exc
        self.encoder.fc = nn.Identity()
        if norm == 'group':
            self._replace_norm(self.encoder)
        self.blocks = nn.ModuleList([ConvBlock(512 + 256, 256, norm), ConvBlock(256 + 128, 128, norm),
                                     ConvBlock(128 + 64, 64, norm), ConvBlock(64 + 64, 32, norm)])
        self.final = ConvBlock(32, 16, norm)
        self.head = nn.Conv2d(16, 1, 1)

    @staticmethod
    def _replace_norm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                layer = norm_layer(child.num_features, 'group')
                with torch.no_grad():
                    layer.weight.copy_(child.weight)
                    layer.bias.copy_(child.bias)
                setattr(module, name, layer)
            else:
                ResUNet34._replace_norm(child)

    def encode(self, image):
        e = self.encoder
        stem = e.relu(e.bn1(e.conv1(image)))
        l1 = e.layer1(e.maxpool(stem))
        l2 = e.layer2(l1)
        l3 = e.layer3(l2)
        l4 = e.layer4(l3)
        return [stem, l1, l2, l3, l4]

    def decode(self, features, image_size):
        d = features[-1]
        for block, skip in zip(self.blocks, reversed(features[:-1])):
            d = F.interpolate(d, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            d = block(torch.cat([skip, d], dim=1))
        return self.head(self.final(F.interpolate(d, size=image_size, mode='bilinear', align_corners=False)))

    def forward(self, image):
        return self.decode(self.encode(image), image.shape[-2:])
