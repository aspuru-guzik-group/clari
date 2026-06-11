from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

Norm = partial(nn.LayerNorm, eps=1e-5)


class Modulate(nn.Module):

    def __init__(self, dim, dim_cond, bias=True):
        super().__init__()

        self.adaptive = dim_cond is not None
        if not self.adaptive:
            return
        self.scale = nn.Linear(dim_cond, dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)
        if bias:
            self.shift = nn.Linear(dim_cond, dim)
            nn.init.zeros_(self.shift.weight)
            nn.init.zeros_(self.shift.bias)
        else:
            self.shift = None

    def forward(self, x, cond):
        if not self.adaptive:
            return x
        x = self.scale(cond) * x
        if self.shift is not None:
            x = x + self.shift(cond)
        return x


class Conditioned(nn.Module):

    def __init__(self, module, dim, dim_cond):
        super().__init__()

        self.norm = Norm(dim, elementwise_affine=False)
        self.mod1 = Modulate(dim, dim_cond)
        self.module = module
        self.mod2 = Modulate(dim, dim_cond, bias=False)

    def forward(self, x, cond, **kwargs):
        x = self.mod1(self.norm(x), cond)
        x = self.mod2(self.module(x, **kwargs), cond)
        return x


class Transition(nn.Module):

    def __init__(self, dim, dim_out=None, expand=2):
        super().__init__()

        if dim_out is None:
            dim_out = dim
        inner = 8 * round((expand * dim) // 8)  # multiple of 8
        self.up = nn.Linear(dim, inner)
        self.gate = nn.Linear(dim, inner)
        self.down = nn.Linear(inner, dim_out)

    def forward(self, x):
        x = F.silu(self.gate(x)) * self.up(x)
        return self.down(x)


class TransitionStack(nn.Module):

    def __init__(self, dim, expand, depth):
        super().__init__()

        self.norms = nn.ModuleList([Norm(dim) for _ in range(depth)])
        self.stack = nn.ModuleList([Transition(dim, expand=expand) for _ in range(depth)])
        self.post_norm = Norm(dim)

    def forward(self, x):
        for norm, block in zip(self.norms, self.stack, strict=False):
            x = x + block(norm(x))
        return self.post_norm(x)


class Attention(nn.Module):

    def __init__(self, dim, dim_pair, num_heads):
        assert dim % num_heads == 0
        super().__init__()

        self.num_heads = num_heads
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.norm_q = Norm(dim)
        self.norm_k = Norm(dim)
        self.proj_z = nn.Linear(dim_pair, num_heads, bias=False)
        self.proj_g = nn.Linear(dim, dim)
        self.proj_o = nn.Linear(dim, dim)

    def forward(self, x, pair, mask):
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        q = self.norm_q(q)
        k = self.norm_k(k)
        q, k, v = [rearrange(x, "b n (h d) -> b h n d", h=self.num_heads) for x in [q, k, v]]
        z = torch.where(mask, self.proj_z(pair), -torch.inf)
        z = rearrange(z, "b n m h -> b h n m")
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=z)
        o = rearrange(o, "b h n d -> b n (h d)")
        g = self.proj_g(x).sigmoid()
        return self.proj_o(g * o)


class Transformer(nn.Module):

    def __init__(self, dim, dim_pair, dim_cond, num_heads, expand, depth):
        super().__init__()

        Cond = partial(Conditioned, dim=dim, dim_cond=dim_cond)
        self.mixs = nn.ModuleList([Cond(Attention(dim, dim_pair, num_heads)) for _ in range(depth)])
        self.mlps = nn.ModuleList([Cond(Transition(dim, expand=expand)) for _ in range(depth)])
        self.post_norm = Norm(dim, elementwise_affine=False)
        self.post_mod = Modulate(dim, dim_cond)

    def forward(self, x, pair, cond, mask):
        cond = rearrange(cond, "b d -> b 1 d") if (cond is not None) else None
        mask = rearrange(mask, "b n -> b 1 n 1")

        for mix, mlp in zip(self.mixs, self.mlps, strict=False):
            x = x + mix(x, cond=cond, pair=pair, mask=mask)
            x = x + mlp(x, cond=cond)
        return self.post_mod(self.post_norm(x), cond)
