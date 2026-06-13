import jsonargparse
import lightning as L
import torch

from clari.datamodules import CrystalDataModuleForFM
from clari.pipelines.base.lit import LitDiT
from clari.pipelines.utils import SimpleTrainer


def main():
    parser = jsonargparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matmul_precision", type=str, default="high")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--fit", type=bool, default=True)

    parser.add_class_arguments(CrystalDataModuleForFM, "data")
    parser.add_class_arguments(LitDiT, "model")
    parser.add_class_arguments(SimpleTrainer, "trainer")

    parser.link_arguments("seed", "data.seed")
    parser.link_arguments("model.interface.collate_fn", "data.collate_fn", apply_on="instantiate")
    parser.link_arguments("trainer.world_size", "data.world_size", apply_on="instantiate")

    args = parser.parse_args()

    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)

    init = parser.instantiate_classes(args)

    artifacts = [init.data.artifact]

    if args.resume_from is not None:
        artifacts.append(torch.load(args.resume_from, map_location="cpu")["artifact"])

    init.model.hparams.update(args.model.as_dict())
    for logger in init.trainer.loggers:
        logger.log_hyperparams(args.as_dict())
        try:
            for k in artifacts:
                logger.use_artifact(k)  # fails in offline mode
        except Exception:
            pass
    print(f"Artifacts used: {artifacts}")

    dispatch = init.trainer.fit if args.fit else init.trainer.validate
    dispatch(model=init.model, datamodule=init.data, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
