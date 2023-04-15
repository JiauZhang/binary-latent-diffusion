import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        z_q = torch.bernoulli(z)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q

# swish
def nonlinearity(x):
    return x*torch.sigmoid(x)

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=1,
            padding='same',
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=2,
            padding='same',
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, dropout, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding='same',
        )
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding='same',
        )
        if self.in_channels != self.out_channels:
            self.shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding='same',
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x+h

class DownBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

class VQModel(nn.Module):
    def __init__(
        self, latent_channels=3, in_channels=3, out_channels=3, vq_embed_dim=None,
    ):
        super().__init__()

        self.encoder = Encoder()
        vq_embed_dim = vq_embed_dim if vq_embed_dim else latent_channels
        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.quantize = VectorQuantizer()
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)
        self.decoder = Decoder()

    def forward(self, input):
        h = self.encoder(input)
        h = self.quant_conv(h)
        quant = self.quantize(h)
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
