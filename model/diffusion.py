import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_schedule()

    def forward(self, z):
        return z

    def register_schedule(self, timesteps=64):
        b_t = torch.linspace(0, 0.5, timesteps, dtype=torch.float64)
        beta_t = (b_t[1:] - b_t[:-1]) / (0.5 - b_t[:-1])
        k_t = torch.cumprod(1 - beta_t, dim=0)
        one = torch.ones(1)
        k_t = torch.cat([one, k_t])

        self.register_buffer('b_t', b_t)
        self.register_buffer('beta_t', beta_t)
        self.register_buffer('k_t', k_t)
