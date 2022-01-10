import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

# helper functions

def exists(val):
    return val is not None

# classes

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Linear(dim * mult, dim)
    )

class ReLA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        causal = True,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        device = x.device
        x = self.norm(x)
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if self.causal:
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device = device).triu_(1).bool()
            sim = sim.masked_fill(mask, 0.)

        # relu attention?
        attn = F.relu(sim)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ReLATransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        causal = True,
        heads = 8,
        dim_head = 64,
        ff_mult = 4
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ReLA(dim = dim, heads = heads, dim_head = dim_head),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.to_logits(x)
