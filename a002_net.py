from torch import nn
import torch
from torch.nn import init


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(31, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(),

            nn.Linear(20, 11)
        )

    def forward(self, x):
        return self.net(x)


def weight_init(layer):
    if isinstance(layer, torch.nn.Linear):
        init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            init.zeros_(layer.bias)
