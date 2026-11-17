import torch
from torch import nn


class BCELoss(nn.BCEWithLogitsLoss):
    def __init__(self, pos_weight=None):
        super().__init__(pos_weight=torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None)


class SoftDiceLoss(nn.Module):
    """Sample-wise mean by default; empty GT retained, smooth=1e-6."""
    def __init__(self, reduction='sample', eps=1e-6):
        super().__init__()
        if reduction not in {'sample', 'batch'}:
            raise ValueError('dice_reduction must be sample or batch')
        self.reduction, self.eps = reduction, eps

    def forward(self, logits, target):
        p, y = logits.float().sigmoid(), target.float()
        dims = (1, 2, 3) if self.reduction == 'sample' else (0, 1, 2, 3)
        return (1 - (2 * (p * y).sum(dims) + self.eps) / (p.sum(dims) + y.sum(dims) + self.eps)).mean()
