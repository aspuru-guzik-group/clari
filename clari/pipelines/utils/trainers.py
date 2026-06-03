import json
import pathlib
import tempfile
from typing import Literal

import lightning as L
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm

from clari.paths import LOG_DIR, random_checkpoint_dir


class GradNormMonitor(L.Callback):

    def on_after_backward(self, trainer, pl_module):
        grad_2norm = grad_norm(pl_module, norm_type=2.0)["grad_2.0_norm_total"]
        pl_module.log("gradients/norm_before_clip", grad_2norm)


class WandbArtifactCallback(L.Callback):

    def __init__(self):
        super().__init__()

        # Manually keep track of version for compatibility with offline wandb
        self.last_version = -1
        self.last_epoch = -1

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        run = wandb.run
        if run is None:
            return
        assert trainer.current_epoch >= self.last_epoch

        artifact_name = f"model-{run.id}"
        if trainer.current_epoch > self.last_epoch:
            artifact = wandb.Artifact(name=artifact_name, type="model")
            with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
                json.dump({"epoch": trainer.current_epoch}, f)
                f.flush()
                artifact.add_file(f.name, is_tmp=True)
            run.log_artifact(artifact)
            self.last_version += 1
            self.last_epoch = trainer.current_epoch
        checkpoint["artifact"] = f"{run.entity}/{run.project}/{artifact_name}:v{self.last_version}"


class SimpleTrainer(L.Trainer):

    def __init__(
        self,
        accelerator: Literal["cpu", "gpu"] = "gpu",
        strategy: Literal["auto", "ddp"] = "auto",
        num_nodes: int = 1,
        devices: int | list[int] = 1,
        precision: Literal["32", "bf16", "bf16-mixed"] = "32",
        max_epochs: int = 1000,
        train_steps_per_epoch: int | None = None,
        val_steps_per_epoch: int | None = None,
        accumulate_grad_batches: int = 1,
        check_val_every_n_epoch: int = 1,
        log_every_n_steps: int = 10,
        progress_bar: bool = True,
        wandb: bool = False,
        wandb_dir: str = str(LOG_DIR),
        wandb_name: str | None = None,
        wandb_project: str = "clari",
        wandb_group: str | None = None,
        wandb_entity: str | None = None,
        checkpoint: bool = False,
        checkpoint_dir: str | None = random_checkpoint_dir(),
        checkpoint_freq: int = 20,
        checkpoint_last: int = 1,
        early_stop: bool = False,
        early_stop_on: str | None = None,
        early_stop_patience: int = 5,
        gradient_monitor: bool = False,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: Literal["norm", "value"] = "norm",
        num_sanity_val_steps: int = 1,
    ):
        callbacks = []

        if train_steps_per_epoch is not None:
            train_steps_per_epoch *= accumulate_grad_batches

        if checkpoint:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename=f"{pathlib.Path(checkpoint_dir).name}-{{epoch}}",
                    save_top_k=checkpoint_last,
                    every_n_epochs=checkpoint_freq,
                    verbose=True,
                )
            )
        if early_stop:
            callbacks.append(
                EarlyStopping(
                    monitor=early_stop_on,
                    patience=early_stop_patience,
                    mode=("min" if ("loss" in early_stop_on) else "max"),
                    verbose=True,
                )
            )

        if wandb:
            logger = WandbLogger(
                name=wandb_name,
                project=wandb_project,
                entity=wandb_entity,
                group=wandb_group,
                log_model=False,
                save_dir=wandb_dir,
            )
            if checkpoint:
                callbacks.append(WandbArtifactCallback())
            if gradient_monitor:
                callbacks.append(GradNormMonitor())
            callbacks.append(LearningRateMonitor())
        else:
            logger = False

        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            num_nodes=num_nodes,
            devices=devices,
            precision=precision,
            callbacks=callbacks,
            enable_checkpointing=checkpoint,
            logger=logger,
            max_epochs=max_epochs,
            limit_train_batches=train_steps_per_epoch,
            limit_val_batches=val_steps_per_epoch,
            accumulate_grad_batches=accumulate_grad_batches,
            check_val_every_n_epoch=check_val_every_n_epoch,
            log_every_n_steps=log_every_n_steps,
            enable_progress_bar=progress_bar,
            enable_model_summary=True,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            num_sanity_val_steps=num_sanity_val_steps,
        )
