from torch import nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return z
