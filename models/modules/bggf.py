import math
import torch
from torch import nn
from torch.nn import functional as F


class BGGF(nn.Module):
    def __init__(self, skip_channels, decoder_channels, centered=True):
        super().__init__()
        hidden = max(8, skip_channels // 2)
        self.ws = nn.Conv2d(skip_channels, hidden, 1)
        self.wd = nn.Conv2d(decoder_channels, hidden, 1)
        self.we = nn.Conv2d(1, hidden, 1)
        self.wa = nn.Conv2d(hidden, 1, 1)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(0.2 / 0.8)))
        self.centered = centered

    @property
    def alpha(self):
        return 0.5 * self.alpha_logit.sigmoid()

    def forward(self, skip, decoder, edge_probability):
        size = skip.shape[-2:]
        decoder = F.interpolate(decoder, size=size, mode='bilinear', align_corners=False)
        edge_probability = F.interpolate(edge_probability, size=size, mode='bilinear', align_corners=False)
        attention = self.wa(F.relu(self.ws(skip) + self.wd(decoder) + self.we(edge_probability))).sigmoid()
        modulation = 2 * attention - 1 if self.centered else attention
        return skip * (1 + self.alpha * modulation)


class AttentionGate(nn.Module):
    def __init__(self, skip_channels, decoder_channels):
        super().__init__()
        hidden = max(8, skip_channels // 2)
        self.ws, self.wd = nn.Conv2d(skip_channels, hidden, 1), nn.Conv2d(decoder_channels, hidden, 1)
        self.wa = nn.Conv2d(hidden, 1, 1)

    def forward(self, skip, decoder):
        decoder = F.interpolate(decoder, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return skip * self.wa(F.relu(self.ws(skip) + self.wd(decoder))).sigmoid()


class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, max(4, channels // 16), 1),
                                 nn.ReLU(), nn.Conv2d(max(4, channels // 16), channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(channels, max(4, channels // 16), 1), nn.ReLU(),
                                 nn.Conv2d(max(4, channels // 16), channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        x = x * (self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))).sigmoid()
        return x * self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)).sigmoid()
