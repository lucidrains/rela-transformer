import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

# classes

class GatedRMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps = 1e-8
    ):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.to_gate = nn.Linear(dim, dim, bias = False)
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        normed_x = x / norm.clamp(min = self.eps) * self.g
        return normed_x * self.to_gate(x).sigmoid()

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
        heads = 8,
        num_memory_kv = 0
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        self.norm = GatedRMSNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.mem_k = nn.Parameter(torch.randn(num_memory_kv, inner_dim))
        self.mem_v = nn.Parameter(torch.randn(num_memory_kv, inner_dim))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            GatedRMSNorm(dim)
        )

    def forward(self, x, mask = None):
        b, device = x.shape[0], x.device
        x = self.norm(x)
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        mem_k, mem_v = map(lambda t: repeat(t, 'n d -> b n d', b = b), (self.mem_k, self.mem_v))
        k = torch.cat((mem_k, k), dim = 1)
        v = torch.cat((mem_v, v), dim = 1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = F.relu(sim)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            i, j = attn.shape[-2:]
            causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            attn = attn.masked_fill(causal_mask, 0.)

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
        num_memory_kv = 0,
        no_ff = False,
        ff_mult = 4,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ReLA(dim = dim, heads = heads, dim_head = dim_head, num_memory_kv = num_memory_kv, causal = causal),
                FeedForward(dim = dim, mult = ff_mult) if not no_ff else None
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
            x = attn(x, mask = mask) + x

            if exists(ff):
                x = ff(x) + x

        return self.to_logits(x)
