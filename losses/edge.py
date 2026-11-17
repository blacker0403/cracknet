from torch import nn
from torch.nn import functional as F


def edge_target(mask, kernel=3):
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError('edge_kernel must be odd and >=3')
    pad = kernel // 2
    padded = F.pad(mask.float(), (pad, pad, pad, pad), mode='replicate')
    return F.max_pool2d(padded, kernel, 1) + F.max_pool2d(-padded, kernel, 1)


class EdgeLoss(nn.Module):
    def __init__(self, kernel=3):
        super().__init__()
        self.kernel = kernel

    def forward(self, logits, target):
        # Upsample predicted logits, retaining thin native-resolution edge labels.
        logits = F.interpolate(logits.float(), size=target.shape[-2:], mode='bilinear', align_corners=False)
        return F.binary_cross_entropy_with_logits(logits, edge_target(target, self.kernel))
