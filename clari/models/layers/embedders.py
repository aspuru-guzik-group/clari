import einops
import numpy as np
import torch
import torch.linalg as LA
import torch.nn as nn
from einops import rearrange


class SinusoidEmbedding(nn.Module):
    def __init__(self, dim, n, wrange, wnum=256, skip=True, div1=False):
        super().__init__()

        periods = np.geomspace(*wrange, num=(wnum // 2))
        scales = 1 / torch.from_numpy(periods).float()
        if div1:
            ms = [0]  # should all be integers
            for s in sorted(scales.tolist()):
                s = max(1, round(s))
                if s <= ms[-1]:
                    s = ms[-1] + 1
                ms.append(s)
            scales = torch.tensor(ms[1:]).float()
        self.register_buffer("freqs", 2 * torch.pi * scales)

        self.proj = nn.Linear(n * wnum, dim, bias=False)
        self.skip = nn.Linear(n, dim, bias=False) if skip else None

    def forward(self, t):
        with torch.autocast("cuda", enabled=False):
            x = self.freqs * t.unsqueeze(-1).float()
            x = einops.rearrange(x, "... n d -> ... (n d)")
            x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = self.proj(x.to(t))
        if self.skip:
            x = x + self.skip(t)
        return x


class VocabEmbedding(nn.Module):
    def __init__(self, vocab, dim, vmax):
        super().__init__()

        m = len(vocab) + 1
        vocab = torch.tensor(vocab).long()
        assert torch.all(vocab >= 0)  # unknown = 0
        self.emb = nn.Embedding(m, dim)

        remap = torch.zeros([vmax + 1], dtype=torch.long)
        remap[vocab] = torch.arange(1, m, dtype=torch.long)
        self.register_buffer("vocab", vocab)
        self.register_buffer("remap", remap)

    def forward(self, x):
        return self.emb(self.remap[x])


class FormulaEmbedding(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()

        self.register_buffer("vocab", torch.tensor(vocab).long())
        self.emb = nn.Linear(len(vocab), dim)

    def forward(self, x, mask):
        matches = (x.unsqueeze(-1) == self.vocab) & mask.unsqueeze(-1)  # (B, N, V)
        f = torch.sum(matches, dim=-2, dtype=torch.float)
        return self.emb(f)


class BinnedEmbedding(nn.Module):
    def __init__(self, dim, bins, binmax):
        super().__init__()

        self.bins = bins
        self.binmax = binmax
        self.emb = nn.Embedding(bins, dim)

    def forward(self, x):
        scale = (self.bins - 1) / self.binmax
        binned = torch.ceil(scale * x).int().clip(min=0, max=(self.bins - 1))
        return self.emb(binned)


class LatticeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.emb_fwd = nn.Linear(9, dim, bias=False)
        self.emb_inv = nn.Linear(9, dim, bias=False)
        self.emb_met = nn.Linear(9, dim, bias=False)
        self.emb_vol = nn.Linear(1, dim, bias=False)

    def forward(self, L):
        with torch.autocast("cuda", enabled=False):
            Linv = LA.inv(L)
        f = [
            self.emb_fwd(rearrange(L, "... i j -> ... (i j)")),
            self.emb_inv(rearrange(Linv, "... i j -> ... (i j)")),
            self.emb_met(rearrange(L.mT @ L, "... i j -> ... (i j)")),
            self.emb_vol(LA.det(L).unsqueeze(-1)),
        ]
        return sum(f) / len(f)
