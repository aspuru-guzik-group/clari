from abc import ABC
from typing import Literal

import torch
import torch.distributions as dist
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from pymatgen.core import Lattice as PmgLattice
from scipy.stats import ortho_group
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from clari.chem import Crystal, distance_lbound
from clari.geometry import zero_com_suffix
from clari.pipelines.utils import bcast_right, masked_mean

TrainTimeDistPreset = Literal["uniform", "lognormal", "logitnormal", "beta", "ramp"]

TRAIN_TDIST_PRESETS: dict[TrainTimeDistPreset, dict] = {
    "uniform": {"shape": "uniform"},
    "lognormal": {"shape": "lognormal", "mu": -1.2, "sigma": 1.2},
    "logitnormal": {"shape": "logitnormal", "mu": 0.0, "sigma": 1.0},
    "beta": {"shape": "beta", "beta0": 1.0, "beta1": 1.8, "eps": 0.0},
    "ramp": {"shape": "ramp"},
}

PriorType = Literal["normal", "lat-normal", "mixture"]


class Interface(ABC):
    def __init__(
        self,
        prior: PriorType = "lat-normal",
        train_tdist: TrainTimeDistPreset | dict = "beta",
    ):
        super().__init__()

        self.prior = prior
        if isinstance(train_tdist, str):
            self.train_tdist = TRAIN_TDIST_PRESETS[train_tdist]
        else:
            self.train_tdist = train_tdist

    def sample_lattice(self, natoms: int, device: torch.device) -> Tensor:
        # Fitted to CSD training set (950k crystals)
        density_mean, density_std = 0.096, 0.015  # atoms/Å³
        angle_mean, angle_std = 91.91, 9.41  # degrees
        len_norm_mean = torch.tensor([0.741, 1.018, 1.470])
        len_norm_chol = torch.linalg.cholesky(
            torch.tensor(
                [
                    [0.022, -0.005, -0.038],
                    [-0.005, 0.024, -0.023],
                    [-0.038, -0.023, 0.134],
                ]
            )
        )

        # Sample atom density → volume
        density = density_mean + density_std * torch.randn(1).item()
        density = max(density, 1e-3)
        volume = natoms / density  # Å³

        # Sample angles (degrees), clip to valid range
        angles = angle_mean + angle_std * torch.randn(3)
        angles = angles.clamp(10.0, 170.0)
        alpha, beta, gamma = angles.tolist()

        # Sample normalised lengths from multivariate Gaussian, then unnormalise
        len_norm = len_norm_mean + len_norm_chol @ torch.randn(3)
        len_norm = len_norm.clamp(min=0.1)
        lengths = len_norm * (volume ** (1 / 3))
        a, b, c = lengths.sort().values.tolist()

        # Build lattice from parameters
        L = torch.tensor(
            PmgLattice.from_parameters(a, b, c, alpha, beta, gamma).matrix,
            dtype=torch.float32,
            device=device,
        )

        # Randomly permute lattice vectors
        L = L[torch.randperm(3, device=device)]

        # Randomly rotate
        R = torch.from_numpy(ortho_group.rvs(3)).to(L)
        return L @ R.T

    def sample_prior(self, C: Crystal) -> Crystal:
        if self.prior == "normal":
            noise = zero_com_suffix(torch.randn_like(C.x), w=C.mask)
        elif self.prior == "lat-normal":
            if C.batched:
                L0 = torch.stack([self.sample_lattice(n, C.device) for n in C.num_atoms.tolist()])
            else:
                L0 = self.sample_lattice(C.num_atoms, C.device)
            noise = Crystal.pack_to_x(L0, Crystal.COORD_NORM * torch.randn_like(C.coords), C.mask)
        else:
            raise ValueError(f"Unknown prior: {self.prior}")
        return C.replace(x=noise)

    def collate_fn(self, batch) -> tuple[Crystal, Crystal]:
        C0s, C1s = [], []
        for C in batch:
            c0 = self.sample_prior(C)
            c1 = C.aligned(c0, on="x")
            C0s.append(c0)
            C1s.append(c1)
        C1s = Crystal.collate(C1s)
        C0s = C1s.replace(x=pad_sequence([C.x for C in C0s], batch_first=True))
        return C0s, C1s

    def sample_t(self, shape: list[int], device: torch.device) -> Tensor:
        p = self.train_tdist
        s = p["shape"]
        if s == "uniform":
            return torch.rand(shape, device=device)
        elif s == "lognormal":
            return torch.exp(torch.randn(shape, device=device) * p["sigma"] + p["mu"])
        elif s == "logitnormal":
            return torch.sigmoid(torch.randn(shape, device=device) * p["sigma"] + p["mu"])
        elif s == "beta":
            beta = dist.Beta(p["beta1"], p["beta0"]).sample(shape)
            unif = dist.Uniform(0, 1).sample(shape)
            return torch.where(torch.rand(shape) < p["eps"], unif, beta).to(device)
        elif s == "ramp":
            u = torch.rand(shape, device=device)
            low = torch.sqrt(3 * u / 4)
            high = (3 * u + 1) / 4
            return torch.where(u <= 1 / 3, low, high)
        else:
            raise ValueError(f"Unknown time distribution: {s}")

    def forward(
        self,
        net: nn.Module,
        xt: Tensor,
        xsc: Tensor | None,
        t: Tensor,
        f: Crystal,
    ) -> Tensor:
        if isinstance(t, float) or (t.ndim == 0):
            t = torch.full([f.batch_size], t).to(xt)
        if xsc is None:
            xsc = torch.full_like(xt, torch.nan)
        pred = net.crystal_forward(x=xt, xsc=xsc, t=t, f=f)
        pred = zero_com_suffix(pred, w=f.mask)
        return pred

    def pred(
        self,
        net: nn.Module,
        xt: Tensor,
        xsc: Tensor | None,
        t: Tensor,
        f: Crystal,
    ) -> Tensor:
        raise NotImplementedError()

    def estimate_x1(self, xt: Tensor, t: Tensor, pred: Tensor) -> Tensor:
        raise NotImplementedError()

    def get_final(
        self,
        net: nn.Module,
        x: Tensor,
        xsc: Tensor | None,
        t: Tensor,
        f: Crystal,
    ) -> Tensor:
        raise NotImplementedError()

    def loss(self, net: nn.Module, batch: tuple[Crystal, Crystal]) -> dict[str, Tensor]:
        raise NotImplementedError()


class SiTInterface(Interface):
    def __init__(
        self,
        prior: PriorType = "normal",
        train_tdist: TrainTimeDistPreset | dict = "uniform",
        train_aux_losses: bool = True,
    ):
        super().__init__(prior=prior, train_tdist=train_tdist)

        self.train_aux_losses = train_aux_losses

    def sample_xt(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        t_ = bcast_right(t, x0)
        return (1 - t_) * x0 + t_ * x1

    def target(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        return x1 - x0

    def pred(
        self,
        net: nn.Module,
        xt: Tensor,
        xsc: Tensor | None,
        t: Tensor,
        f: Crystal,
    ) -> Tensor:
        return self.forward(net=net, xt=xt, xsc=xsc, t=t, f=f)

    def score(self, xt: Tensor, t: Tensor, pred: Tensor, scale: Tensor) -> Tensor:
        t_ = bcast_right(t, xt)
        scale = scale / (1 - t_)
        return scale * (t_ * pred - xt)

    def estimate_x1(self, xt: Tensor, t: Tensor, pred: Tensor) -> Tensor:
        t_ = bcast_right(t, xt)
        return xt + (1.0 - t_) * pred

    def loss(self, net: nn.Module, batch: tuple[Crystal, Crystal]) -> dict[str, Tensor]:
        C0, C1 = batch
        x0, x1 = C0.x, C1.x

        # Interpolate
        t = self.sample_t([C1.batch_size], device=C1.x.device)
        xt = self.sample_xt(x0, x1, t)

        # Self-conditioning
        if net.self_cond:
            with torch.no_grad():
                nsc = C0.batch_size // 2
                fsc = C0.subset(slice(0, nsc))
                out = self.pred(net=net, xt=xt[:nsc], xsc=None, t=t[:nsc], f=fsc)
                xsc = torch.full_like(xt, torch.nan)
                xsc[:nsc] = self.estimate_x1(xt[:nsc], t[:nsc], out)
            xsc = xsc.detach()
        else:
            xsc = None

        # Forward pass
        pred = self.forward(net=net, xt=xt, xsc=xsc, t=t, f=C0)
        true = self.target(x0, x1, t)

        # Losses (flow-matching and aux.)
        losses = F.mse_loss(pred, true, reduction="none")
        loss_lattice = losses[:, :3].mean()
        loss_coord = masked_mean(losses[:, 3:], C0.mask.unsqueeze(-1), dim=[1, 2]).mean()
        if self.train_aux_losses:
            pred_x1 = self.estimate_x1(xt, t, pred)
            loss_vol = self._vol_losses(pred_x1, x1).mean()
            loss_ldd = self._ldd_losses(pred_x1, x1, f=C0).mean()
        else:
            loss_vol = 0.0
            loss_ldd = 0.0
        loss = loss_lattice + loss_coord + loss_vol + loss_ldd

        return {
            "loss": loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
            "loss_vol": loss_vol,
            "loss_ldd": loss_ldd,
        }

    def _vol_losses(self, pred_x1, true_x1):
        pred_volume = LA.det(pred_x1[:, :3]).abs()
        true_volume = LA.det(true_x1[:, :3]).abs()
        ratio = pred_volume / true_volume
        return torch.abs(ratio - 1)

    def _ldd_losses(self, pred_x1, true_x1, f: Crystal):
        ticks = torch.tensor([-1, 0, 1]).to(pred_x1)
        grid = torch.cartesian_prod(ticks, ticks, ticks)  # (27 3)
        pred_dists = self._pcdist(pred_x1, grid)  # (B N N)
        true_dists = self._pcdist(true_x1, grid)  # (B N N)
        cutoff = distance_lbound(f.atom_nums, f.atom_nums, bond_mask=(f.bonds > 0))

        mask_ldd = (
            ((true_dists < 15) | (pred_dists < cutoff))
            & ~torch.eye(true_dists.shape[-1]).to(f.mask)
            & (f.mask.unsqueeze(-1) & f.mask.unsqueeze(-2))
        )
        error_term = F.l1_loss(pred_dists, true_dists, reduction="none")
        clash_term = F.l1_loss(torch.minimum(pred_dists, cutoff), cutoff, reduction="none")
        loss = masked_mean(error_term + 5 * clash_term, mask_ldd, dim=[1, 2])
        return loss / Crystal.COORD_NORM

    def _pcdist(self, x, grid):
        x = Crystal.COORD_NORM * x
        grid = einsum(2 * x[:, :3], grid, "b j i, g j -> b g i")
        grid = rearrange(grid, "b g i -> g b 1 i")
        return torch.cdist(x[:, 3:], grid + x[:, 3:]).amin(dim=0)  # (27 B N N)
