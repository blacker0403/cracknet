from torch import nn
from models.resunet34 import norm_layer


class BoundaryHead(nn.Sequential):
    def __init__(self, channels=64, norm='batch'):
        super().__init__(nn.Conv2d(channels, 32, 3, padding=1, bias=False), norm_layer(32, norm),
                         nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))
