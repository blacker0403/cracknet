"""2-D soft morphology following Shit et al., CVPR 2021 (see docs/references.md)."""
import torch
from torch import nn
from torch.nn import functional as F


def soft_erode(image):
    return torch.minimum(-F.max_pool2d(-image, (3, 1), 1, (1, 0)),
                         -F.max_pool2d(-image, (1, 3), 1, (0, 1)))


def soft_dilate(image):
    return F.max_pool2d(image, 3, 1, 1)


def soft_open(image):
    return soft_dilate(soft_erode(image))


def soft_skeletonize(image, iterations=10):
    skeleton = F.relu(image - soft_open(image))
    for _ in range(iterations):
        image = soft_erode(image)
        delta = F.relu(image - soft_open(image))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(logits, target, iterations=10, eps=1e-6):
    with torch.autocast(device_type=logits.device.type, enabled=False):
        p, y = logits.float().sigmoid(), target.float()
        valid = y.sum((1, 2, 3)) > 0
        skipped = int((~valid).sum().item())
        if not valid.any():
            return p.sum() * 0, skipped
        p, y = p[valid], y[valid]
        sp, sy = soft_skeletonize(p, iterations), soft_skeletonize(y, iterations)
        dims = (1, 2, 3)
        precision = ((sp * y).sum(dims) + eps) / (sp.sum(dims) + eps)
        recall = ((sy * p).sum(dims) + eps) / (sy.sum(dims) + eps)
        loss = 1 - 2 * precision * recall / (precision + recall).clamp_min(eps)
        return loss.mean(), skipped


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iterations=10):
        super().__init__()
        self.iterations, self.skipped = iterations, 0

    def forward(self, logits, target):
        loss, self.skipped = soft_cldice_loss(logits, target, self.iterations)
        return loss
