import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_out=None, dim_mult=1):
        super().__init__()
        inner_dim = int(dim_in * dim_mult)
        if dim_out is None: dim_out = dim_in
        self.ff = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, hidden_states):
        return self.ff(hidden_states)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, cross_attention_dim=None, embed_dim=None, num_heads=4, bias=False):
        super().__init__()
        embed_dim = embed_dim if embed_dim else query_dim
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.to_q = nn.Linear(query_dim, embed_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, embed_dim, bias=bias)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.to_out = nn.Linear(embed_dim, query_dim)

    def forward(self, hidden_states, cross_hidden_states=None):
        query = self.to_q(hidden_states)
        if cross_hidden_states is None: cross_hidden_states = hidden_states
        key = self.to_k(cross_hidden_states)
        value = self.to_v(cross_hidden_states)
        attn_output, attn_output_weights = self.mha(query, key, value)
        return attn_output

class AttnBlock(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.attn1 = CrossAttention(dim_in)
        self.ff = FeedForward(dim_in)
        self.ln1 = nn.LayerNorm(dim_in)
        self.ln2 = nn.LayerNorm(dim_in)

    def forward(self, hidden_states):
        skip = hidden_states
        hidden_states = skip + self.attn1(self.ln1(hidden_states))
        skip = hidden_states
        hidden_states = skip + self.ff(self.ln2(hidden_states))
        return hidden_states

class Transformer2D(nn.Module):
    def __init__(self, in_channels, num_layers, embed_dim=None, out_channels=None):
        super().__init__()
        self.register_schedule()
        if out_channels is None: out_channels = in_channels
        if embed_dim is None: embed_dim = in_channels
        self.to_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.ModuleList([
            AttnBlock(embed_dim) for _ in range(num_layers)
        ])
        self.to_out = nn.Conv2d(embed_dim, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, z):
        b, _, h, w = z.shape
        z = self.to_in(z)
        z = z.permute(0, 2, 3, 1).reshape(b, h*w, -1)
        for block in self.blocks:
            z = block(z)
        z = z.permute(0, 2, 1).reshape(b, -1, h, w)
        z = self.to_out(z)
        return z

    def register_schedule(self, timesteps=64):
        b_t = torch.linspace(0, 0.5, timesteps, dtype=torch.float64)
        beta_t = (b_t[1:] - b_t[:-1]) / (0.5 - b_t[:-1])
        zero = torch.zeros(1)
        beta_t = torch.cat([zero, beta_t])
        k_t = torch.cumprod(1 - beta_t, dim=0)

        self.register_buffer('b_t', b_t)
        self.register_buffer('beta_t', beta_t)
        self.register_buffer('k_t', k_t)
