from torch import nn
from losses.bce_dice import BCELoss, SoftDiceLoss
from losses.cldice import SoftCLDiceLoss
from losses.edge import EdgeLoss


class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bce = BCELoss(cfg['pos_weight'])
        self.dice = SoftDiceLoss(cfg['dice_reduction'])
        self.cldice = SoftCLDiceLoss(cfg['skeleton_iterations'])
        self.edge = EdgeLoss(cfg['edge_kernel'])

    def forward(self, output, target):
        logits = output['logits'] if isinstance(output, dict) else output
        terms = {'bce': self.bce(logits.float(), target.float()), 'dice': self.dice(logits, target)}
        terms['cldice'] = self.cldice(logits, target) if self.cfg['cldice'] else logits.sum() * 0
        if self.cfg['edge']:
            if not isinstance(output, dict) or output.get('edge_logits') is None:
                raise ValueError('Edge Loss requires predicted edge_logits.')
            terms['edge'] = self.edge(output['edge_logits'], target)
        else:
            terms['edge'] = logits.sum() * 0
        total = sum(self.cfg[k] * v for k, v in terms.items())
        return total, {**{k: float(v.detach()) for k, v in terms.items()},
                       'cldice_skipped': self.cldice.skipped if self.cfg['cldice'] else 0}
