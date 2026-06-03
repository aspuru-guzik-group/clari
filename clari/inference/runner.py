from __future__ import annotations

import multiprocessing as mp
import shutil
from pathlib import Path

import torch

from clari.chem import Crystal
from clari.inference.io import merge_shards, prepare_output_dir, run_config, write_shard
from clari.inference.sampler import (
    DEFAULT_MAX_RESAMPLE_FACTOR,
    ClariSampler,
    SampleChunk,
    SampleRequest,
    crystal_from_request,
    make_progress,
    make_sample_chunks,
)


def sample_to_directory(
    checkpoint_path: str | Path,
    requests: list[SampleRequest],
    output_dir: str | Path,
    *,
    sampler: ClariSampler | None = None,
    batch_size: int | None = None,
    num_gpus: int = 1,
    device: str | torch.device | None = "auto",
    use_ema: bool = True,
    use_bf16: bool = True,
    n_steps: int | None = None,
    compile: bool | None = None,
    torch_threads: int = 1,
    overwrite: bool = False,
    keep_shards: bool = False,
    filter_clashing: bool = True,
    max_resample_factor: int = DEFAULT_MAX_RESAMPLE_FACTOR,
    pbar: bool | str = True,
    step_pbar: bool | str = False,
) -> Path:
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    if not requests:
        raise ValueError("At least one SampleRequest is required")

    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, overwrite=overwrite)
    shards_dir = output_dir / ".shards"
    shards_dir.mkdir()

    chunks = make_sample_chunks(requests, batch_size=batch_size, device=device)
    (output_dir / "config.json").write_text(
        _config_json(
            checkpoint_path=checkpoint_path,
            requests=requests,
            batch_size=batch_size,
            num_gpus=num_gpus,
            device=device,
            use_ema=use_ema,
            use_bf16=use_bf16,
            n_steps=n_steps,
            compile=compile,
            torch_threads=torch_threads,
            keep_shards=keep_shards,
            filter_clashing=filter_clashing,
            max_resample_factor=max_resample_factor if filter_clashing else None,
            chunks=chunks,
        )
    )

    if num_gpus == 1:
        if sampler is None:
            sampler = ClariSampler.from_checkpoint(
                checkpoint_path,
                use_ema=use_ema,
                device=device,
                n_steps=n_steps,
                use_bf16=use_bf16,
                compile=compile,
                torch_threads=torch_threads,
            )
        _run_chunks(
            sampler,
            requests,
            chunks,
            shards_dir,
            filter_clashing=filter_clashing,
            max_resample_factor=max_resample_factor,
            pbar=pbar,
            step_pbar=step_pbar,
        )
    else:
        _run_multi_gpu(
            checkpoint_path=checkpoint_path,
            requests=requests,
            chunks=chunks,
            shards_dir=shards_dir,
            num_gpus=num_gpus,
            use_ema=use_ema,
            n_steps=n_steps,
            use_bf16=use_bf16,
            compile=compile,
            torch_threads=torch_threads,
            filter_clashing=filter_clashing,
            max_resample_factor=max_resample_factor,
            pbar=pbar,
            step_pbar=step_pbar,
        )

    predictions_path = output_dir / "predictions.parquet"
    merge_shards(shards_dir, predictions_path, chunks)
    if not keep_shards:
        shutil.rmtree(shards_dir)
    return predictions_path


def _config_json(**kwargs) -> str:
    import json

    return json.dumps(run_config(**kwargs), indent=2) + "\n"


def _run_chunks(
    sampler: ClariSampler,
    requests: list[SampleRequest],
    chunks: list[SampleChunk],
    shards_dir: Path,
    *,
    filter_clashing: bool = True,
    max_resample_factor: int = DEFAULT_MAX_RESAMPLE_FACTOR,
    pbar: bool | str,
    step_pbar: bool | str,
) -> None:
    progress = make_progress(
        pbar,
        total=sum(chunk.count for chunk in chunks),
        desc="Sampling crystals",
        unit="sample",
    )
    crystals: dict[int, Crystal] = {}
    for chunk in chunks:
        request = requests[chunk.request_idx]
        if chunk.request_idx not in crystals:
            crystals[chunk.request_idx] = crystal_from_request(request)
        samples = sampler.sample_crystal(
            crystals[chunk.request_idx],
            n_samples=chunk.count,
            batch_size=chunk.count,
            sample_idx_start=chunk.sample_idx_start,
            id=chunk.id,
            metadata=request.metadata,
            filter_clashing=filter_clashing,
            max_resample_factor=max_resample_factor,
            pbar=False,
            step_pbar=step_pbar,
        )
        write_shard(shards_dir, chunk, samples)
        if progress is not None:
            progress.update(chunk.count)
    if progress is not None:
        progress.close()


def _run_multi_gpu(
    *,
    checkpoint_path: str | Path,
    requests: list[SampleRequest],
    chunks: list[SampleChunk],
    shards_dir: Path,
    num_gpus: int,
    use_ema: bool,
    n_steps: int | None,
    use_bf16: bool,
    compile: bool | None,
    torch_threads: int,
    filter_clashing: bool,
    max_resample_factor: int,
    pbar: bool | str,
    step_pbar: bool | str,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"Requested {num_gpus} GPUs, but CUDA is not available")
    if torch.cuda.device_count() < num_gpus:
        raise RuntimeError(f"Requested {num_gpus} GPUs, found {torch.cuda.device_count()}")

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(num_gpus):
        rank_chunks = chunks[rank::num_gpus]
        proc = ctx.Process(
            target=_gpu_worker,
            kwargs={
                "rank": rank,
                "checkpoint_path": checkpoint_path,
                "requests": requests,
                "chunks": rank_chunks,
                "shards_dir": shards_dir,
                "use_ema": use_ema,
                "n_steps": n_steps,
                "use_bf16": use_bf16,
                "compile": compile,
                "torch_threads": torch_threads,
                "filter_clashing": filter_clashing,
                "max_resample_factor": max_resample_factor,
                "pbar": pbar,
                "step_pbar": step_pbar,
            },
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Sampling worker failed with exit code {proc.exitcode}")


def _gpu_worker(
    *,
    rank: int,
    checkpoint_path: str | Path,
    requests: list[SampleRequest],
    chunks: list[SampleChunk],
    shards_dir: Path,
    use_ema: bool,
    n_steps: int | None,
    use_bf16: bool,
    compile: bool | None,
    torch_threads: int,
    filter_clashing: bool,
    max_resample_factor: int,
    pbar: bool | str,
    step_pbar: bool | str,
) -> None:
    torch.cuda.set_device(rank)
    sampler = ClariSampler.from_checkpoint(
        checkpoint_path,
        use_ema=use_ema,
        device=f"cuda:{rank}",
        n_steps=n_steps,
        use_bf16=use_bf16,
        compile=compile,
        torch_threads=torch_threads,
    )
    _run_chunks(
        sampler,
        requests,
        chunks,
        shards_dir,
        filter_clashing=filter_clashing,
        max_resample_factor=max_resample_factor,
        pbar=(f"GPU {rank}" if pbar else False),
        step_pbar=step_pbar,
    )
