import torch
import torch.linalg as LA
import torch.nn.functional as F
from einops import einsum, rearrange


def zero_com(x, w=None, return_com=False):  # (* N D) (* N)
    if w is None:
        w = torch.ones_like(x[..., :, 0])
    w = w.float()
    w = w / w.sum(dim=-1, keepdim=True)  # (* N)
    com = (x * w.unsqueeze(-1)).sum(dim=-2, keepdim=True)  # (* 1 D)
    x = x - com
    return (x, com) if return_com else x


def zero_com_suffix(x, w=None, start=3):  # (* 3+N D) (* N)
    return torch.cat([x[..., :start, :], zero_com(x[..., start:, :], w=w)], dim=-2)


def rmsd(A, B):  # (* N D) (* N 3)
    return (A - B).square().sum(dim=-1).mean(dim=-1).sqrt()


# Reference:
# https://github.com/Graylab/GeoDock/blob/main/geodock/utils/metrics.py
def kabsch_align(A, B, w=None, sym="so3", return_rot=False):  # (* N D) (* N D) (* N)
    # E(3):  rotation + reflection + translations
    # SE(3): rotation + translations
    # O(3):  rotation + reflection
    # SO(3): rotation
    assert sym in {"e3", "o3", "se3", "so3"}

    if A.shape != B.shape:
        A, B = torch.broadcast_tensors(A, B)
    if w is None:
        w = torch.ones_like(A[..., :, 0]) / A.shape[-2]
    if sym in {"e3", "se3"}:
        A, comA = zero_com(A, w=w, return_com=True)
        B, comB = zero_com(B, w=w, return_com=True)
        t = comB
    else:
        t = 0

    # Covariance matrix and SVD
    # Optimized: (D, N) @ (N, D) -> (D, D) is O(N)
    H = (A * w.unsqueeze(-1)).mT @ B
    U, _, Vt = LA.svd(H.float(), full_matrices=True)
    if sym in {"se3", "so3"}:
        is_rot = LA.det(U).sign() * LA.det(Vt).sign()
        SS = torch.diag_embed(F.pad(is_rot.unsqueeze(-1), (2, 0), value=1.0))
        R = Vt.mT @ SS @ U.mT
    else:
        R = Vt.mT @ U.mT

    return (R, t) if return_rot else (A @ R.mT + t)


def periodic_pdist(x, L):
    ticks = torch.tensor([0, -1, 1]).to(x)
    grid = torch.cartesian_prod(ticks, ticks, ticks)  # (27 3), grid[0] = [0,0,0]
    dx = einsum(L, grid, "b j i, g j -> b g i")  # (B 27 3)
    with torch.autocast("cuda", enabled=False):
        f = einsum(x.float(), LA.inv(L.float()), "b n i, b i j -> b n j") % 1
    x = einsum(f.to(x), L, "b n i, b i j -> b n j")
    x1 = rearrange(x, "b n d -> b n 1 1 d")
    x2 = rearrange(x, "b n d -> b 1 n 1 d")
    dx = rearrange(dx, "b g d -> b 1 1 g d")
    D = LA.vector_norm(x1 - x2 + dx, dim=-1)  # (B N N 27)
    n = torch.arange(x.shape[1])
    D[:, n, n, 0] = torch.inf
    return D.amin(dim=-1)
