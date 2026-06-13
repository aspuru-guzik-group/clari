from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
import tqdm
from torch import Tensor

from clari.chem import Crystal
from clari.geometry import zero_com_suffix
from clari.pipelines.base.interfaces import Interface
from clari.pipelines.utils import bcast_right

SamplingSchedule = Literal["uniform", "log"]

SCHEDULE_PRESETS: dict[SamplingSchedule, dict] = {
    "uniform": {"shape": "uniform"},
    "log": {"shape": "log"},
}


class Sampler(ABC):
    def __init__(
        self,
        num_steps: int = 100,
        schedule: SamplingSchedule | dict = "uniform",
        stochasticity: Literal["none", "t(1-t)", "(1-t)/t"] = "none",
        alpha: float = 0.0,
    ):
        self.num_steps = int(num_steps)
        if isinstance(schedule, str):
            self.schedule_params = SCHEDULE_PRESETS[schedule]
        else:
            self.schedule_params = schedule
        self.stochasticity = stochasticity
        self.alpha = alpha

    @abstractmethod
    def step(
        self,
        interface: Interface,
        net: nn.Module,
        x: Tensor,
        xsc: Tensor | None,
        t_curr: Tensor,
        t_next: Tensor,
        f: Crystal,
        last: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError()

    def sample_t(
        self,
        steps: int,
        *,
        t_start: float | Tensor = 0.0,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> Tensor:
        p = self.schedule_params
        s = p["shape"]

        if s == "uniform":
            t = torch.linspace(0, 1, steps, device=device, dtype=dtype)
        elif s == "log":
            t = 1.0 - torch.logspace(-2, 0, steps, device=device, dtype=dtype).flip(0)
            t = t - torch.min(t)
            t = t / torch.max(t)
        else:
            raise ValueError(f"Sampling schedule {s} not supported.")

        if isinstance(t_start, float):
            t_start = torch.tensor([t_start], device=device, dtype=dtype)
        return t_start + t.unsqueeze(-1) * (1.0 - t_start)

    def g(self, x: Tensor, t: Tensor, last: bool = False) -> Tensor:
        if last or self.stochasticity == "none":
            gt = torch.zeros_like(x)
        elif self.stochasticity == "t(1-t)":
            gt = self.alpha * t * (1 - t)
        elif self.stochasticity == "(1-t)/t":
            gt = self.alpha * (1 - t) / (t + 0.01)
        else:
            raise ValueError(f"Unknown stochasticity: {self.stochasticity}")
        gt = bcast_right(gt, x)
        gt[..., :3, :] = 0.0
        return gt

    def sample(
        self,
        interface: Interface,
        net: nn.Module,
        C: Crystal,
        *,
        sample_prior: bool = True,
        t_start: float | Tensor = 0.0,
        pbar: str | None = None,
        return_trajectory: bool = False,
    ) -> Crystal | Tensor:
        if sample_prior:
            C0 = C.replace(x=torch.zeros_like(C.x))
            C0 = interface.sample_prior(C0)
        else:
            C0 = C

        x = C0.x
        xsc = None
        T = self.num_steps
        traj = [x] if return_trajectory else None

        timegrid = self.sample_t(T + 1, t_start=t_start, device=C.device, dtype=C.x.dtype)
        for i in tqdm.trange(T, desc=pbar, leave=False, disable=(pbar is None)):
            x, xsc = self.step(
                interface=interface,
                net=net,
                x=x,
                xsc=xsc,
                t_curr=timegrid[i],
                t_next=timegrid[i + 1],
                f=C,
                last=(i == T - 1),
            )
            x = zero_com_suffix(x, w=C.mask)
            xsc = zero_com_suffix(xsc, w=C.mask)
            if return_trajectory:
                traj.append(x)

        return torch.stack(traj) if return_trajectory else C.replace(x=x)


class EulerSampler(Sampler):
    def step(
        self,
        interface: Interface,
        net: nn.Module,
        x: Tensor,
        xsc: Tensor | None,
        t_curr: Tensor,
        t_next: Tensor,
        f: Crystal,
        last: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if last and (type(interface).get_final is not Interface.get_final):
            return interface.get_final(net=net, x=x, xsc=xsc, t=t_curr, f=f), None

        gt = self.g(x, t_curr, last=last)
        dt = bcast_right(t_next - t_curr, x)
        v = interface.pred(net=net, xt=x, xsc=xsc, t=t_curr, f=f)
        drift = v + interface.score(xt=x, t=t_curr, pred=v, scale=gt)
        noise = torch.sqrt(2 * gt * dt) * zero_com_suffix(torch.randn_like(x), w=f.mask)
        x_next = x + (drift * dt) + noise
        xsc_next = interface.estimate_x1(xt=x, t=t_curr, pred=v)
        return x_next, xsc_next


class HeunSampler(Sampler):
    def step(
        self,
        interface: Interface,
        net: nn.Module,
        x: Tensor,
        xsc: Tensor | None,
        t_curr: Tensor,
        t_next: Tensor,
        f: Crystal,
        last: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if last and (type(interface).get_final is not Interface.get_final):
            return interface.get_final(net=net, x=x, xsc=xsc, t=t_curr, f=f), None

        gt = self.g(x, t_curr, last=last)
        dt = bcast_right(t_next - t_curr, x)
        v_curr = interface.pred(net=net, xt=x, xsc=xsc, t=t_curr, f=f)
        drift_curr = v_curr + interface.score(xt=x, t=t_curr, pred=v_curr, scale=gt)
        noise = torch.sqrt(2 * gt * dt) * zero_com_suffix(torch.randn_like(x), w=f.mask)
        x_euler = x + drift_curr * dt + noise
        xsc_euler = interface.estimate_x1(xt=x, t=t_curr, pred=v_curr)

        if last:
            return x_euler, xsc_euler
        gt_next = self.g(x_euler, t_next)
        v_next = interface.pred(net=net, xt=x_euler, xsc=xsc_euler, t=t_next, f=f)
        drift_next = v_next + interface.score(xt=x_euler, t=t_next, pred=v_next, scale=gt_next)
        x_next = x + 0.5 * (drift_curr + drift_next) * dt + noise
        xsc_next = interface.estimate_x1(xt=x_euler, t=t_next, pred=v_next)
        return x_next, xsc_next
