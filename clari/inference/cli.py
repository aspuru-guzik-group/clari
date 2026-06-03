from __future__ import annotations

from pathlib import Path

import jsonargparse

from clari.inference.sampler import (
    DEFAULT_MAX_RESAMPLE_FACTOR,
    ClariSampler,
    HubModel,
    make_requests,
)


def sample(
    output_dir: Path,
    smiles: str | list,
    checkpoint_path: Path | None = None,
    from_hub: HubModel | None = None,
    n_samples: int | list[int] = 1,
    copies: int | list[int] = 4,
    ids: str | list[str] | None = None,
    batch_size: int | None = None,
    num_gpus: int = 1,
    device: str | None = "auto",
    n_steps: int | None = None,
    use_ema: bool = True,
    use_bf16: bool = True,
    compile: bool | None = None,
    torch_threads: int = 1,
    overwrite: bool = False,
    keep_shards: bool = False,
    filter_clashing: bool = True,
    max_resample_factor: int = DEFAULT_MAX_RESAMPLE_FACTOR,
    pbar: bool = True,
    step_pbar: bool = False,
) -> None:
    """Sample crystal structures from asymmetric-unit SMILES strings.

    If batch_size is omitted, CLARI uses an automatic atom-count heuristic scaled from
    batch sizes that were fit on an 81 GB H100 GPU. Pass batch_size explicitly when you
    want fixed behavior across GPUs.

    By default the sampler filters out crystals whose atoms inter-penetrate (per
    clari.pipelines.utils.metrics.check_clashes_eval) and resamples the deficit until
    n_samples non-clashing structures are reached. Pass filter_clashing=False to keep
    all samples regardless of clashes. The loop gives up after
    max_resample_factor * n_samples total attempts to avoid spinning forever on
    pathological molecules.
    """
    if from_hub is not None:
        sampler = ClariSampler.from_hub(
            from_hub,
            use_ema=use_ema,
            device=device,
            n_steps=n_steps,
            use_bf16=use_bf16,
            compile=compile,
            torch_threads=torch_threads,
            num_gpus=num_gpus,
        )
    elif checkpoint_path is not None:
        sampler = ClariSampler.from_checkpoint(
            checkpoint_path,
            use_ema=use_ema,
            device=device,
            n_steps=n_steps,
            use_bf16=use_bf16,
            compile=compile,
            torch_threads=torch_threads,
            num_gpus=num_gpus,
        )
    else:
        raise ValueError("Either --checkpoint_path or --from_hub must be provided.")
    requests = make_requests(smiles, ids=ids, n_samples=n_samples, copies=copies)
    sampler.sample(
        requests,
        output_dir=output_dir,
        batch_size=batch_size,
        num_gpus=num_gpus,
        overwrite=overwrite,
        keep_shards=keep_shards,
        filter_clashing=filter_clashing,
        max_resample_factor=max_resample_factor,
        pbar=pbar,
        step_pbar=step_pbar,
    )
    predictions_path = output_dir / "predictions.parquet"
    print(f"Saved predictions to {predictions_path}")


def main() -> None:
    parser = jsonargparse.ArgumentParser(
        description="Sample organic crystal structures from asymmetric-unit SMILES strings."
    )
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
    parser.add_function_arguments(sample)
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if k != "config"}
    sample(**args)


if __name__ == "__main__":
    main()
