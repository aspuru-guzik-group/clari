from functools import partial

import torch
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import Tensor

from clari.chem import ATOM_FEATURES, Crystal
from clari.geometry import periodic_pdist
from clari.models.layers import (
    BinnedEmbedding,
    FormulaEmbedding,
    LatticeEmbedding,
    Modulate,
    SinusoidEmbedding,
    Transformer,
    Transition,
    TransitionStack,
    VocabEmbedding,
)


class DiT(nn.Module):

    def __init__(
        self,
        dim: int = 256,
        dim_pair: int = 32,
        dim_cond: int = 512,
        num_heads: int = 8,
        expand: float = 4.0,
        depth: int = 16,
        lattice_nodes: bool = True,
        self_cond: bool = True,
        use_mpa: bool = False,
    ):
        super().__init__()

        assert not use_mpa  # deprecated: kept for checkpoint compatibility
        self.dim = dim
        self.dim_pair = dim_pair
        self.dim_cond = dim_cond
        self.use_lattice_regs = lattice_nodes
        self.self_cond = self_cond

        # Mined most common from training set, everything else set to unknown (0)
        # fmt: off
        common_atoms = [
            0, 1, 3, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19,
            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            42, 44, 45, 46, 47, 50, 51, 52, 53,
            74, 75, 76, 77, 78, 79,
            120, 121, 122  # lattice atom nums
        ]
        # fmt: on

        # Note that 3D coords are in units of (1 / Crystal.COORD_NORM) Å
        CartEmbedder = partial(SinusoidEmbedding, n=3, wrange=(0.001, 15))
        FracEmbedder = partial(SinusoidEmbedding, n=3, wrange=(0.001, 1), div1=True)
        DistEmbedder = partial(BinnedEmbedding, bins=128, binmax=4)

        # Conditioning featurizers
        self.embed_timestep = SinusoidEmbedding(dim_cond, n=1, wrange=(0.001, 5))
        self.embed_lattice = LatticeEmbedding(dim_cond)
        self.embed_formula = FormulaEmbedding(common_atoms[:-3], dim_cond)

        # Sequence featurizers
        self.embed_cart = CartEmbedder(dim)
        self.embed_frac = FracEmbedder(dim)
        self.embed_element = VocabEmbedding(common_atoms, dim, vmax=122)
        self.embed_feats = nn.Linear(ATOM_FEATURES, dim, bias=False)
        self.node_mod = Modulate(dim, dim_cond)

        # Optional featurizers
        if self.self_cond:
            self.embed_self_cond = nn.Embedding(2, dim_cond)
            self.embed_cart_sc = CartEmbedder(dim)

        # Pair featurizers
        self.embed_dist_cart = DistEmbedder(dim_pair)
        self.embed_dist_pdic = DistEmbedder(dim_pair)
        self.embed_bonds = nn.Embedding(32, dim_pair)
        self.pair_mod = Modulate(dim_pair, dim_cond)

        # Main body
        self.stem_node = Transition(dim, dim + 2 * dim_pair, expand=2)
        self.stem_cond = TransitionStack(dim_cond, expand=2, depth=2)
        self.trunk = Transformer(
            dim=dim,
            dim_pair=dim_pair,
            dim_cond=dim_cond,
            num_heads=num_heads,
            expand=expand,
            depth=depth,
        )
        self.head = Transition(dim, 3, expand=2)

        if not self.use_lattice_regs:
            assert not self.self_cond
            self.lattice_proj = nn.Linear(9, dim, bias=False)
            self.lattice_head = Transition(dim, 9, expand=2)

    def crystal_forward(self, x, xsc, t, f: Crystal) -> Tensor:
        bonds = f.bonds + 18  # shift: [-17, 5] -> [1, 23]
        if self.use_lattice_regs:
            pad = (3, 0)
            lat_nums = torch.tensor([120, 121, 122]).to(f.atom_nums)
            lat_nums = repeat(lat_nums, "d -> b d", b=x.shape[0])
            return self(
                x=x,
                t=t,
                xsc=xsc.to(x),
                atom_nums=torch.cat([lat_nums, f.atom_nums], dim=1),
                atom_feats=F.pad(f.atom_feats, (0, 0) + pad, value=0),
                bonds=F.pad(bonds, pad + pad, value=0),
                mask=F.pad(f.mask, pad, value=True),
            )
        else:
            return self(
                x=x,
                t=t,
                xsc=xsc.to(x),
                atom_nums=f.atom_nums,
                atom_feats=f.atom_feats,
                bonds=bonds,
                mask=f.mask,
            )

    def forward(
        self,
        x,  # (B 3+N 3)
        t,  # (B 2)
        xsc,  # (B 3+N 3)
        atom_nums,  # (B 3+N)
        atom_feats,  # (B 3+N D)
        bonds,  # (B 3+N 3+N)
        mask,  # (B 3+N)
    ):
        L = 2 * clip_sigma_min(x[:, :3])
        x = x if self.use_lattice_regs else x[:, 3:]
        with torch.no_grad():
            n0 = 3 if self.use_lattice_regs else 0
            dc = zero_top_left(torch.cdist(x, x), n=n0)
            dp = zero_top_left(periodic_pdist(x, L), n=n0)

        # Embed conditioning features
        cond = [
            self.embed_timestep(t.unsqueeze(-1)),
            self.embed_lattice(0.5 * L),
            self.embed_formula(atom_nums[:, 3:], mask[:, 3:]),
        ]
        if self.self_cond:
            is_sc = torch.isfinite(xsc).all(dim=[1, 2]).long()  # (B)
            cond.append(self.embed_self_cond(is_sc))
        cond = self.stem_cond(sum(cond) / len(cond))

        # Embed sequence features
        h = [
            self.embed_cart(x),
            self.embed_frac(frac(x, L)),
            self.embed_element(atom_nums),
            self.embed_feats(atom_feats),
        ]
        if not self.use_lattice_regs:
            h.append(self.lattice_proj(rearrange(x[:, :3], "b i j -> b 1 (i j)")))
        if self.self_cond:
            assert self.use_lattice_regs
            xsc = torch.nan_to_num(xsc)
            h.append(self.embed_cart_sc(xsc))
        h = F.tanh((0.5 / len(h)) * sum(h))
        h = self.node_mod(h, rearrange(cond, "b d -> b 1 d"))
        h, e1, e2 = self.stem_node(h).split([self.dim, self.dim_pair, self.dim_pair], dim=-1)

        # Embed pair features
        pair = [
            rearrange(e1, "b n d -> b n 1 d"),
            rearrange(e2, "b n d -> b 1 n d"),
            self.embed_dist_cart(dc),
            self.embed_dist_pdic(dp),
            self.embed_bonds(bonds),
        ]
        pair = F.tanh((0.5 / len(pair)) * sum(pair))
        pair = self.pair_mod(pair, rearrange(cond, "b d -> b 1 1 d"))

        # Trunk
        h = self.trunk(h, pair=pair, cond=cond, mask=mask)

        # Output
        mask = mask.unsqueeze(-1)
        v = torch.where(mask, self.head(h), 0)  # for safety
        if not self.use_lattice_regs:
            hL = torch.where(mask, h, 0).sum(dim=-2) / mask.float().sum(dim=-2).clamp(min=1)
            vL = rearrange(self.lattice_head(hL), "b (i j) -> b i j", i=3, j=3)
            v = torch.cat([vL, v], dim=-2)  # (B 3+N 3)
        return v


def clip_sigma_min(L):
    with torch.autocast("cuda", enabled=False):
        U, S, Vh = LA.svd(L.float())
    return (U @ torch.diag_embed(S.clamp(min=0.15)) @ Vh).to(L)


def zero_top_left(d, n):
    d[:, :n, :] = 0
    d[:, :, :n] = 0
    return d


def frac(x, L):
    with torch.autocast("cuda", enabled=False):
        f = einsum(LA.inv(L), x, "... j i, ... n j -> ... n i")
    return f.to(x)
