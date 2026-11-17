from torch import nn
from torch.nn import functional as F
import torch

from models.resunet34 import ResUNet34, ConvBlock
from models.modules.mdcc import MDCC, ASPP
from models.modules.bggf import BGGF, AttentionGate, SE, CBAM
from models.modules.boundary_head import BoundaryHead


class CrackTopoNet(ResUNet34):
    def __init__(self, cfg):
        super().__init__(cfg['pretrained'], cfg['weights_dir'], cfg.get('weights_file'), cfg['norm'])
        self.bottleneck_dropout = nn.Dropout2d(p=cfg.get('dropout', 0.0))
        self.contexts, self.gates = nn.ModuleDict(), nn.ModuleDict()
        self.fusion, self.scales = cfg['fusion'], cfg['fusion_scales']
        if cfg['context'] != 'none':
            for layer in cfg['context_layers']:
                if layer not in [1, 2, 3, 4]:
                    raise ValueError('context_layers must be from [1,2,3,4]')
                channels = [64, 128, 256, 512][layer - 1]
                self.contexts[str(layer)] = (ASPP(channels, cfg['context_width_ratio'], cfg['norm']) if cfg['context'] == 'aspp'
                    else MDCC(channels, cfg['context'], cfg['context_width_ratio'], cfg['norm']))
        self.boundary = BoundaryHead(64, cfg['norm']) if cfg['boundary_head'] else None
        for index, (scale, sc, dc) in enumerate(zip([16, 8, 4, 2], [256, 128, 64, 64], [512, 256, 128, 64])):
            if scale not in self.scales:
                continue
            if self.fusion in {'bggf', 'bggf_uncentered'}:
                self.gates[str(index)] = BGGF(sc, dc, centered=self.fusion == 'bggf')
            elif self.fusion == 'attention':
                self.gates[str(index)] = AttentionGate(sc, dc)
            elif self.fusion in {'se', 'cbam'}:
                self.gates[str(index)] = (SE if self.fusion == 'se' else CBAM)(sc)
            elif self.fusion == 'edge_concat':
                self.blocks[index] = ConvBlock(sc + dc + 1, [256, 128, 64, 32][index], cfg['norm'])
            elif self.fusion != 'concat':
                raise ValueError(f'Unknown fusion: {self.fusion}')

    def encode(self, image):
        e = self.encoder
        stem = e.relu(e.bn1(e.conv1(image)))
        features = [stem]
        x = e.maxpool(stem)
        for layer, block in enumerate([e.layer1, e.layer2, e.layer3, e.layer4], start=1):
            x = block(x)
            if str(layer) in self.contexts:
                x = self.contexts[str(layer)](x)
            features.append(x)
        return features

    def forward(self, image):
        features = self.encode(image)
        edge_logits = self.boundary(features[1]) if self.boundary is not None else None
        edge = edge_logits.sigmoid() if edge_logits is not None else None
        d = self.bottleneck_dropout(features[-1])
        for index, (block, skip, scale) in enumerate(zip(self.blocks, reversed(features[:-1]), [16, 8, 4, 2])):
            d = F.interpolate(d, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            if str(index) in self.gates:
                gate = self.gates[str(index)]
                skip = gate(skip, d, edge) if isinstance(gate, BGGF) else gate(skip, d) if isinstance(gate, AttentionGate) else gate(skip)
            parts = [skip, d]
            if self.fusion == 'edge_concat' and scale in self.scales:
                parts.append(F.interpolate(edge, size=skip.shape[-2:], mode='bilinear', align_corners=False))
            d = block(torch.cat(parts, dim=1))
        logits = self.head(self.final(F.interpolate(d, size=image.shape[-2:], mode='bilinear', align_corners=False)))
        return {'logits': logits, 'edge_logits': edge_logits}


def build_model(cfg, initialize=True):
    cfg = dict(cfg)
    if not initialize:  # Checkpoint already contains complete encoder weights; never redownload.
        cfg.update(pretrained=False, weights_file=None)
    if cfg['name'] == 'deeplabv3plus':
        from models.deeplab_baseline import DeepLabBaseline
        return DeepLabBaseline(cfg)
    if cfg['name'] == 'linknet101':
        from models.linknet101 import LinkNet101
        return LinkNet101(cfg)
    if cfg['name'] != 'resunet34':
        raise ValueError(f'Unknown network: {cfg["name"]}')
    return CrackTopoNet(cfg)
