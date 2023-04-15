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
            padding=1,
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, dropout=0.0, out_channels=None):
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

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0,
        )
        self.k = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0,
        )
        self.v = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0,
        )
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0,
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)     # [b, hw, c]
        k = k.reshape(b, c, h*w)   # [b, c, hw]
        w_ = torch.bmm(q, k)       # [b, hw, hw]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # [b, hw, hw] (first hw of k, second of q)
        h_ = torch.bmm(v, w_)      # [b, c, hw] (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)

        return x + h_

class Encoder(nn.Module):
    def __init__(
        self, in_channels, base_ch, num_res_blocks, latent_channels, ch_mult=(1,2,4,8)
    ):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(
            in_channels, base_ch, kernel_size=3, stride=1, padding='same',
        ))
        in_channels = base_ch
        for mul in ch_mult:
            out_channels = base_ch * mul
            for _ in range(num_res_blocks):
                layers.append(ResnetBlock(in_channels, out_channels=out_channels))
                in_channels = out_channels
            layers.append(Downsample(in_channels))
        layers.append(ResnetBlock(in_channels))
        layers.append(AttnBlock(in_channels))
        layers.append(ResnetBlock(in_channels))
        layers.append(Normalize(in_channels))
        layers.append(nn.Conv2d(
            in_channels, latent_channels, kernel_size=3, stride=1, padding='same',
        ))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        latent = self.layers(input)
        return latent

class Decoder(nn.Module):
    def __init__(
        self, out_channels, base_ch, num_res_blocks, latent_channels, ch_mult=(8,4,2,1)
    ):
        super().__init__()
        layers = []
        out_channels_ = out_channels
        in_channels = base_ch * ch_mult[0]
        layers.append(nn.Conv2d(
            latent_channels, in_channels, kernel_size=3, stride=1, padding='same',
        ))
        layers.append(ResnetBlock(in_channels))
        layers.append(AttnBlock(in_channels))
        layers.append(ResnetBlock(in_channels))
        for mul in ch_mult:
            out_channels = base_ch * mul
            for _ in range(num_res_blocks):
                layers.append(ResnetBlock(in_channels, out_channels=out_channels))
                in_channels = out_channels
            layers.append(Upsample(in_channels))
        layers.append(Normalize(in_channels))
        layers.append(nn.Conv2d(
            in_channels, out_channels_, kernel_size=3, stride=1, padding='same',
        ))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        latent = self.layers(input)
        return latent

class VQModel(nn.Module):
    def __init__(
        self, latent_channels=8, in_channels=3, out_channels=3, vq_embed_dim=None,
        base_ch=32, num_res_blocks=2, 
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, base_ch, num_res_blocks, latent_channels)
        vq_embed_dim = vq_embed_dim if vq_embed_dim else latent_channels
        self.quant_conv = nn.Sequential(
            nn.Conv2d(latent_channels, vq_embed_dim, 1),
            nn.Sigmoid(),
        )
        self.quantize = VectorQuantizer()
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)
        self.decoder = Decoder(out_channels, base_ch, num_res_blocks, latent_channels)

    def forward(self, input):
        h = self.encoder(input)
        h = self.quant_conv(h)
        quant = self.quantize(h)
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
