from torch import nn
import torch
from models.resunet34 import norm_layer


class MDCC(nn.Module):
    """Candidate directional/context combination, not an originality claim."""
    def __init__(self, channels, variant='mdcc', width_ratio=0.25, norm='batch'):
        super().__init__()
        width = max(8, int(channels * width_ratio))
        recipes = {
            'local': [(3, 3, 1)],
            'directional': [(1, 7, 1), (7, 1, 1)],
            'dilated': [(3, 3, 1), (3, 3, 2)],
            'mdcc': [(1, 1, 1), (3, 3, 1), (1, 7, 1), (7, 1, 1), (3, 3, 2)],
        }
        if variant not in recipes:
            raise ValueError(f'Unknown context variant: {variant}')
        self.branches = nn.ModuleList()
        for kh, kw, dilation in recipes[variant]:
            layers = [nn.Conv2d(channels, width, 1, bias=False)]
            if (kh, kw) != (1, 1):
                layers.append(nn.Conv2d(width, width, (kh, kw), padding=(kh // 2 * dilation, kw // 2 * dilation),
                                        dilation=dilation, groups=width, bias=False))
            self.branches.append(nn.Sequential(*layers))
        self.fuse = nn.Sequential(nn.Conv2d(width * len(self.branches), channels, 1, bias=False),
                                  norm_layer(channels, norm), nn.GELU())
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.gamma * self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class ASPP(nn.Module):
    """ASPP control: 1x1, dilated 3x3 at 6/12/18, image pooling and projection."""
    def __init__(self, channels, width_ratio=0.25, norm='batch'):
        super().__init__()
        width = max(8, int(channels * width_ratio))
        self.branches = nn.ModuleList([nn.Conv2d(channels, width, 1, bias=False)] +
            [nn.Conv2d(channels, width, 3, padding=d, dilation=d, bias=False) for d in [6, 12, 18]])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, width, 1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(width * 5, channels, 1, bias=False), norm_layer(channels, norm), nn.GELU())
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        pooled = nn.functional.interpolate(self.pool(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x + self.gamma * self.fuse(torch.cat([b(x) for b in self.branches] + [pooled], dim=1))
