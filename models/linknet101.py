"""ResNet101 with a standard five-stage LinkNet decoder; returns binary logits."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet101, ResNet101_Weights


class DecoderBlock(nn.Sequential):
    def __init__(self, cin, cout):
        hidden = cin // 4
        super().__init__(
            nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class LinkNet101(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = resnet101(weights=None)
        if cfg.get('weights_file') or cfg['pretrained']:
            try:
                state = (torch.load(cfg['weights_file'], map_location='cpu', weights_only=True)
                         if cfg.get('weights_file') else torch.hub.load_state_dict_from_url(
                             ResNet101_Weights.IMAGENET1K_V1.url, model_dir=str(Path(cfg['weights_dir'])),
                             map_location='cpu', check_hash=True))
                self.encoder.load_state_dict(state)
            except Exception as exc:
                raise RuntimeError(f'LinkNet ResNet101 weights failed: {exc}; no random-initialization fallback.') from exc
        self.encoder.fc = nn.Identity()
        self.decoder = nn.ModuleList([DecoderBlock(cin, cout) for cin, cout in
                                     [(2048, 1024), (1024, 512), (512, 256), (256, 64), (64, 32)]])
        self.head = nn.Conv2d(32, 1, 1)
        for layer in self.decoder.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image):
        h, w = image.shape[-2:]
        x = F.pad(image, (0, (-w) % 32, 0, (-h) % 32))
        e = self.encoder
        stem = e.relu(e.bn1(e.conv1(x)))
        l1 = e.layer1(e.maxpool(stem))
        l2 = e.layer2(l1)
        l3 = e.layer3(l2)
        d = e.layer4(l3)
        for block, skip in zip(self.decoder[:4], [l3, l2, l1, stem]):
            d = block(d) + skip
        d = self.decoder[4](d)
        return {'logits': self.head(d)[..., :h, :w], 'edge_logits': None}
