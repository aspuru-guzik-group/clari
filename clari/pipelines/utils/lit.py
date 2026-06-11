import numpy as np
try:
    import py3Dmol
except ImportError:
    py3Dmol = None
import torch
import torch.linalg as LA
from torch.optim.lr_scheduler import LambdaLR

from clari.pipelines.utils.metrics import assess_crystals_train
from clari.pipelines.utils.muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam


def check_sample(C) -> bool:
    if not torch.isfinite(C.x).all():
        print(f"ERROR: non-finite sample for {C.csd_id}")
        return False
    if LA.det(C.lattice.cpu()).abs() < 1:  # got some weird cuda SLURM error
        print(f"ERROR: degenerate lattice for {C.csd_id}")
        return False
    return True


def sample_metrics(preds, trues):
    metrics = []
    for Y, T in zip(preds, trues, strict=False):
        if not check_sample(Y):
            continue
        metrics.append(assess_crystals_train(pred=Y, true=T))
    if len(metrics) == 0:
        return dict()
    return {k: np.mean([x[k] for x in metrics]).item() for k in metrics[0]}


def sample_views(preds, trues, wrap="all"):
    if py3Dmol is None:
        raise ValueError("py3Dmol is not installed")
    grid = (len(preds), 4)
    width, height = 300, 200

    view = py3Dmol.view(width=(width * grid[1]), height=(height * grid[0]), viewergrid=grid)
    for i in range(grid[0]):
        if not check_sample(preds[i]):
            continue
        trues[i] = trues[i].aligned(preds[i], on="lattice")
        view = preds[i].show(wrap="none", view=view, viewer=(i, 0))
        view = trues[i].show(wrap="none", view=view, viewer=(i, 1))
        view = preds[i].show(wrap=wrap, view=view, viewer=(i, 2))
        view = trues[i].show(wrap=wrap, view=view, viewer=(i, 3))
    view.render()

    t = view.js()
    js = t.startjs + t.endjs
    return js


def build_muon_optimizer(params_adam, params_muon, lr, wd, lr_warmup=0):
    if torch.distributed.is_initialized():
        Muon = MuonWithAuxAdam
    else:
        Muon = SingleDeviceMuonWithAuxAdam
    optimizer = Muon(
        [
            dict(params=params_adam, lr=lr, weight_decay=wd, use_muon=False),
            dict(params=params_muon, lr=lr, weight_decay=wd, use_muon=True),
        ]
    )

    if lr_warmup >= 1:
        scheduler = LambdaLR(optimizer, lr_lambda=(lambda n: min(1, n / lr_warmup)))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    else:
        return optimizer
