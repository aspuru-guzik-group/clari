from __future__ import annotations

import importlib

import torch
from torch import Tensor


def instantiate(obj):
    if not isinstance(obj, dict):
        return obj
    class_path, init_args = obj["class_path"], obj["init_args"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**init_args)


def prefix_keys(pre, d):
    return {f"{pre}{k}": v for k, v in d.items()}


def bcast_right(x: Tensor, y: Tensor) -> Tensor:
    assert y.ndim >= x.ndim
    return x.reshape(x.shape + (1,) * (y.ndim - x.ndim))


def masked_mean(x: Tensor, mask: Tensor, dim: list[int] | None = None) -> Tensor:
    x, mask = torch.broadcast_tensors(x, mask)
    x = torch.where(mask, x, 0.0)
    n = mask.sum(dim=dim, dtype=torch.int64).clip(min=1)
    return x.sum(dim=dim) / n.to(x.dtype)
