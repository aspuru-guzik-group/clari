from __future__ import annotations

import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from clari.models import DiT
from clari.pipelines.base.interfaces import Interface
from clari.pipelines.base.samplers import Sampler
from clari.pipelines.utils import (
    EMA,
    build_muon_optimizer,
    instantiate,
    prefix_keys,
    sample_metrics,
    sample_views,
)


class LitDiT(L.LightningModule):
    def __init__(
        self,
        net: DiT,
        interface: Interface,
        sampler: Sampler,
        lr: float = 5e-4,
        lr_warmup: int = 5000,
        wd: float = 0.0,
        ema_decay: float | None = None,
        sample_every_n_epochs: int = 1,
        sample_n_repeat: int = 3,
        sample_n_visualize: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "interface", "sampler"])

        self.net = instantiate(net)
        self.interface = instantiate(interface)
        self.sampler = instantiate(sampler)

    def configure_model(self):
        self.net.compile()

    def configure_callbacks(self):
        if self.hparams.ema_decay is None:
            return []
        return [EMA(decay=self.hparams.ema_decay)]

    def configure_optimizers(self):
        hp = self.hparams
        params_adam, params_muon = [], []
        for name, p in self.named_parameters():
            if ("trunk" in name) and p.ndim >= 2:
                params_muon.append(p)
            else:
                params_adam.append(p)
        return build_muon_optimizer(params_adam, params_muon, hp.lr, hp.wd, hp.lr_warmup)

    def training_step(self, batch, batch_idx):
        return self._step(batch, split="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        should_sample = (self.current_epoch + 1) % self.hparams.sample_every_n_epochs == 0
        if dataloader_idx == 0:
            return self._step(batch, split="val")
        elif self.trainer.sanity_checking or not should_sample:
            pass
        elif dataloader_idx == 1:
            self._assess_samples(batch, batch_idx, split="val")
        else:
            raise ValueError()

    def _step(self, batch, split):
        losses = self.interface.loss(self.net, batch)
        self.log_dict(
            prefix_keys(split + "/", losses),
            batch_size=batch[0].batch_size,
            sync_dist=(split != "train"),
            add_dataloader_idx=False,
        )
        return losses["loss"]

    @torch.inference_mode()
    @torch.autocast("cuda", enabled=False)
    def _assess_samples(self, batch, idx, split):
        hp = self.hparams
        global_rank_zero = self.global_rank == 0

        C_pred = []
        C_true = []

        for _ in range(hp.sample_n_repeat):
            samples = self.sampler.sample(
                interface=self.interface,
                net=self.net,
                C=batch,
                pbar=("Sampling" if global_rank_zero else None),
            )
            C_pred.extend(samples.unbatch())
            C_true.extend(batch.unbatch())

        using_wandb = isinstance(self.logger, WandbLogger)
        if (idx == 0) and global_rank_zero and using_wandb:
            nviz = hp.sample_n_visualize
            js = sample_views(C_pred[:nviz], C_true[:nviz])
            wandb.log({f"samples/{split}": wandb.Html(js), "epoch": self.current_epoch})
            C_pred = C_pred[nviz:]
            C_true = C_true[nviz:]

        self.log_dict(
            prefix_keys(split + "/", sample_metrics(C_pred, C_true)),
            batch_size=len(C_pred),
            sync_dist=True,
            add_dataloader_idx=False,
        )
