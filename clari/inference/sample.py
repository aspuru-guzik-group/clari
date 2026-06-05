from __future__ import annotations

import json
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import torch
from tqdm.auto import tqdm

from clari.chem import Crystal
from clari.inference.inputs import (
    SampleRequest,
    _make_request,
    request_components,
    resolve_hub_checkpoint,
)
from clari.pipelines.base.lit import LitDiT

H100_REFERENCE_MEMORY_GB = 81.0


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def request_to_crystal(request: SampleRequest) -> Crystal:
    return Crystal.from_smiles(
        request_components(request),
        csd_id=request.id,
    )


def build_run_config(
    requests: list[SampleRequest],
    model: str,
    device: str,
    num_gpus: int,
    batch_size: int | None,
    n_steps: int | None,
    use_ema: bool,
    use_bf16: bool,
    compile: bool | None,
    overwrite: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "device": device,
        "num_gpus": num_gpus,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "use_ema": use_ema,
        "use_bf16": use_bf16,
        "compile": compile,
        "overwrite": overwrite,
        "requests": [
            {
                "id": request.id,
                "smiles": request.smiles,
                "copies": request.copies,
                "samples": request.samples,
            }
            for request in requests
        ],
    }


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not output_dir.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
    if not any(output_dir.iterdir()):
        return
    if not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    for name in ("predictions.parquet", "config.json", "rankings.csv", "energies.csv"):
        path = output_dir / name
        if path.is_file():
            path.unlink()
    for name in (".shards", "cifs"):
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)


def auto_batch_size(n_atoms: int, batch_size: int | None, device: str | torch.device | None) -> int:
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return batch_size
    if n_atoms > 1000:
        base = 1
    elif n_atoms < 200:
        base = 1000
    elif n_atoms < 300:
        base = 500
    elif n_atoms < 500:
        base = 200
    else:
        base = 25
    if torch.cuda.is_available():
        resolved = resolve_device(device)
        if resolved.type == "cuda":
            index = resolved.index if resolved.index is not None else torch.cuda.current_device()
            total_gb = torch.cuda.get_device_properties(index).total_memory / 1e9
            scale = min(
                1.0, max(1.0 / H100_REFERENCE_MEMORY_GB, total_gb / H100_REFERENCE_MEMORY_GB)
            )
            return max(1, int(base * scale))
    return base


def load_lit(
    path: str | Path,
    device: torch.device,
    use_ema: bool,
    n_steps: int | None,
    compile: bool,
) -> LitDiT:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    lit = LitDiT(**checkpoint["hyper_parameters"])
    if use_ema:
        ema_weights = (
            (checkpoint.get("ema") or {}).get("ema_weights")
            if isinstance(checkpoint, dict)
            else None
        )
        lit.load_state_dict(
            ema_weights if isinstance(ema_weights, dict) else checkpoint["state_dict"],
            strict=False,
        )
    else:
        lit.load_state_dict(checkpoint["state_dict"])
    if n_steps is not None:
        lit.sampler.num_steps = int(n_steps)
    lit = lit.to(device).eval()
    if compile:
        if device.type != "cuda":
            raise ValueError("compile=True is only supported on CUDA devices")
        import torch._inductor.config as inductor_config

        inductor_config.online_softmax = False
        lit.configure_model()
    return lit


class ClariSampler:
    def __init__(
        self,
        checkpoint: str = "clari-m",
        *,
        device: str | torch.device | None = "auto",
        use_ema: bool = True,
        use_bf16: bool = True,
        n_steps: int | None = 50,
        compile: bool = False,
        torch_threads: int = 1,
        num_gpus: int = 1,
        seed: int | None = None,
    ):
        if n_steps is not None and n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        checkpoint_key = str(checkpoint).strip().lower()
        if checkpoint_key in (
            "clari-m",
            "clari-l",
            "clari-h",
            "clari-med",
            "clari-large",
            "clari-huge",
        ):
            checkpoint = resolve_hub_checkpoint(checkpoint_key)
        resolved_device = resolve_device(device)
        torch.set_num_threads(torch_threads)
        torch.set_float32_matmul_precision("high")
        self.lit = load_lit(checkpoint, resolved_device, use_ema, n_steps, compile)
        self.device = resolved_device
        self.model = str(checkpoint)
        self.use_ema = use_ema
        self.use_bf16 = use_bf16 and resolved_device.type == "cuda"
        self.n_steps = n_steps
        self.compile = compile
        self.torch_threads = torch_threads
        self.num_gpus = num_gpus
        self.seed = seed
        if seed is not None:
            _seed_everything(seed)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = "auto",
        use_ema: bool = True,
        use_bf16: bool = True,
        n_steps: int | None = 50,
        compile: bool = False,
        torch_threads: int = 1,
        num_gpus: int = 1,
        seed: int | None = None,
    ) -> ClariSampler:
        return cls(
            str(path),
            device=device,
            use_ema=use_ema,
            use_bf16=use_bf16,
            n_steps=n_steps,
            compile=compile,
            torch_threads=torch_threads,
            num_gpus=num_gpus,
            seed=seed,
        )

    @torch.inference_mode()
    def sample_batch(self, crystal: Crystal, count: int) -> list[Crystal]:
        batch = Crystal.collate([crystal])
        current = count
        while True:
            batch_gpu = None
            try:
                crystals = batch.unbatch()
                batch_gpu = (
                    type(batch)
                    .collate([sample for sample in crystals for _ in range(current)])
                    .to(self.device)
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16 and self.device.type == "cuda",
                ):
                    out = self.lit.sampler.sample(
                        self.lit.interface,
                        self.lit.net,
                        batch_gpu,
                        pbar=None,
                    )
                return out.cpu().unbatch()
            except RuntimeError as exc:
                if (
                    self.device.type != "cuda"
                    or "out of memory" not in str(exc).lower()
                    or current <= 1
                ):
                    raise
                if batch_gpu is not None:
                    del batch_gpu
                torch.cuda.empty_cache()
                current = max(1, current // 2)

    @torch.inference_mode()
    def sample_request(
        self,
        request: SampleRequest,
        *,
        samples: int | None = None,
        batch_size: int | None = None,
        progress: tqdm | None = None,
    ) -> list[Crystal]:
        crystal = request_to_crystal(request)
        target_samples = request.samples if samples is None else samples
        per_batch = min(
            target_samples, auto_batch_size(int(crystal.num_atoms), batch_size, self.device)
        )
        produced: list[Crystal] = []
        while len(produced) < target_samples:
            need = min(per_batch, target_samples - len(produced))
            got = self.sample_batch(crystal, need)
            produced.extend(got[:need])
            per_batch = min(per_batch, len(got))
            if progress is not None:
                progress.update(min(len(got), need))
        return produced[:target_samples]

    @torch.inference_mode()
    def sample(
        self,
        smiles: str | list[str] | SampleRequest | list[SampleRequest],
        *,
        id: str | None = None,
        copies: int | list[int] = 4,
        samples: int = 1,
        output_dir: str | Path | None = None,
        batch_size: int | None = None,
        num_gpus: int | None = None,
        overwrite: bool = False,
        pbar: bool = True,
        seed: int | None = None,
    ) -> list[Crystal] | Path:
        if isinstance(smiles, SampleRequest):
            requests = [smiles]
        elif isinstance(smiles, list) and smiles and isinstance(smiles[0], SampleRequest):
            requests = list(smiles)
        else:
            requests = [_make_request(smiles, id=id, copies=copies, samples=samples)]
        num_gpus = self.num_gpus if num_gpus is None else num_gpus
        if output_dir is None and num_gpus > 1:
            raise ValueError("num_gpus > 1 requires output_dir.")
        if seed is not None:
            _seed_everything(seed)
        if output_dir is None:
            progress = (
                tqdm(
                    total=sum(request.samples for request in requests),
                    desc="Sampling",
                    unit="sample",
                )
                if pbar
                else None
            )
            samples: list[Crystal] = []
            try:
                for request in requests:
                    samples.extend(
                        self.sample_request(request, batch_size=request.batch_size or batch_size, progress=progress)
                    )
            finally:
                if progress is not None:
                    progress.close()
            return samples
        return sample_to_directory(
            self,
            requests=requests,
            output_dir=Path(output_dir),
            batch_size=batch_size,
            num_gpus=num_gpus,
            overwrite=overwrite,
            pbar=pbar,
            seed=seed,
        )


def build_chunks(
    requests: list[SampleRequest],
    batch_size: int | None,
    device: str | torch.device,
) -> list[dict[str, int]]:
    chunks: list[dict[str, int]] = []
    sample_idx = 0
    for request_index, request in enumerate(requests):
        crystal = request_to_crystal(request)
        chunk_size = auto_batch_size(int(crystal.num_atoms), request.batch_size or batch_size, device)
        for local_start in range(0, request.samples, chunk_size):
            count = min(chunk_size, request.samples - local_start)
            chunks.append(
                {
                    "request_index": request_index,
                    "sample_idx_start": sample_idx + local_start,
                    "count": count,
                    "shard_index": len(chunks),
                }
            )
        sample_idx += request.samples
    return chunks


def rows_for_samples(
    request: SampleRequest, sample_idx_start: int, samples: list[Crystal]
) -> list[dict[str, Any]]:
    rows = []
    for offset, crystal in enumerate(samples):
        rows.append(
            {"id": request.id, "sample_idx": sample_idx_start + offset, "cif": crystal.to_cif()}
        )
    return rows


def write_shard(shards_dir: Path, shard_index: int, rows: list[dict[str, Any]]) -> None:
    pl.DataFrame(rows).write_parquet(shards_dir / f"shard_{shard_index:06d}.parquet")


def merge_shards(shards_dir: Path, predictions_path: Path) -> None:
    paths = sorted(shards_dir.glob("shard_*.parquet"))
    if not paths:
        raise RuntimeError(f"No shard parquet files were written in {shards_dir}")
    pl.concat([pl.read_parquet(path) for path in paths], how="vertical").sort(
        "sample_idx"
    ).write_parquet(predictions_path)


def run_chunks(
    sampler: ClariSampler,
    requests: list[SampleRequest],
    chunks: list[dict[str, int]],
    shards_dir: Path,
    pbar: bool,
) -> list[Crystal]:
    progress = (
        tqdm(total=sum(chunk["count"] for chunk in chunks), desc="Sampling", unit="sample")
        if pbar
        else None
    )
    all_samples: list[Crystal] = []
    try:
        for chunk in chunks:
            request = requests[chunk["request_index"]]
            samples = sampler.sample_request(
                request,
                samples=chunk["count"],
                batch_size=chunk["count"],
                progress=progress,
            )
            write_shard(
                shards_dir,
                chunk["shard_index"],
                rows_for_samples(request, chunk["sample_idx_start"], samples),
            )
            all_samples.extend(samples)
    finally:
        if progress is not None:
            progress.close()
    return all_samples


def gpu_worker(
    rank: int,
    model: str,
    requests: list[SampleRequest],
    chunks: list[dict[str, int]],
    shards_dir: str,
    use_ema: bool,
    use_bf16: bool,
    n_steps: int | None,
    compile: bool | None,
    torch_threads: int,
    error_queue: mp.queues.Queue,
    seed: int | None,
) -> None:
    import sys
    import traceback

    try:
        torch.cuda.set_device(rank)
        sampler = ClariSampler.from_checkpoint(
            model,
            device=f"cuda:{rank}",
            use_ema=use_ema,
            use_bf16=use_bf16,
            n_steps=n_steps,
            compile=compile,
            torch_threads=torch_threads,
            num_gpus=1,
            seed=None if seed is None else seed + rank,
        )
        run_chunks(sampler, requests, chunks, Path(shards_dir), pbar=False)
    except Exception:
        error_queue.put((rank, traceback.format_exc()))
        sys.exit(1)


def sample_to_directory(
    sampler: ClariSampler,
    *,
    requests: list[SampleRequest],
    output_dir: Path,
    batch_size: int | None,
    num_gpus: int,
    overwrite: bool,
    pbar: bool,
    seed: int | None = None,
) -> Path:
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    if not requests:
        raise ValueError("At least one request is required.")
    prepare_output_dir(output_dir, overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(
            build_run_config(
                requests,
                sampler.model,
                str(sampler.device),
                num_gpus,
                batch_size,
                sampler.n_steps,
                sampler.use_ema,
                sampler.use_bf16,
                sampler.compile,
                overwrite,
            ),
            indent=2,
        )
        + "\n"
    )
    chunks = build_chunks(requests, batch_size, sampler.device)
    shards_dir = output_dir / ".shards"
    shards_dir.mkdir(exist_ok=True)
    if num_gpus == 1:
        run_chunks(sampler, requests, chunks, shards_dir, pbar=pbar)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {num_gpus} GPUs, but CUDA is not available")
        if torch.cuda.device_count() < num_gpus:
            raise RuntimeError(f"Requested {num_gpus} GPUs, found {torch.cuda.device_count()}")
        ctx = mp.get_context("spawn")
        error_queue = ctx.Queue()
        procs = []
        for rank in range(num_gpus):
            proc = ctx.Process(
                target=gpu_worker,
                args=(
                    rank,
                    sampler.model,
                    requests,
                    chunks[rank::num_gpus],
                    str(shards_dir),
                    sampler.use_ema,
                    sampler.use_bf16,
                    sampler.n_steps,
                    sampler.compile,
                    sampler.torch_threads,
                    error_queue,
                    seed,
                ),
            )
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get_nowait())
        if errors:
            msgs = "\n".join(f"GPU {r}:\n{tb}" for r, tb in sorted(errors))
            raise RuntimeError(f"Sampling worker(s) failed:\n{msgs}")
        elif any(proc.exitcode != 0 for proc in procs):
            raise RuntimeError("A sampling worker failed (no traceback captured).")
    merge_shards(shards_dir, output_dir / "predictions.parquet")
    shutil.rmtree(shards_dir)
    return output_dir


@torch.inference_mode()
def sample(
    requests: SampleRequest | list[SampleRequest],
    *,
    model: str = "clari-m",
    output_dir: str | Path | None = None,
    batch_size: int | None = None,
    num_gpus: int = 1,
    device: str | torch.device | None = "auto",
    n_steps: int | None = 50,
    use_ema: bool = True,
    use_bf16: bool = True,
    compile: bool = False,
    torch_threads: int = 1,
    overwrite: bool = False,
    pbar: bool = True,
    seed: int | None = None,
) -> list[Crystal] | Path:
    sampler = ClariSampler(
        model,
        device=device,
        use_ema=use_ema,
        use_bf16=use_bf16,
        n_steps=n_steps,
        compile=compile,
        torch_threads=torch_threads,
        num_gpus=num_gpus,
        seed=seed,
    )
    return sampler.sample(
        requests,
        output_dir=output_dir,
        batch_size=batch_size,
        num_gpus=num_gpus,
        overwrite=overwrite,
        pbar=pbar,
        seed=seed,
    )


def save(crystals: list[Crystal], output_dir: str | Path, overwrite: bool = False) -> Path:
    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": c.csd_id or "sample", "sample_idx": i, "cif": c.to_cif()}
        for i, c in enumerate(crystals)
    ]
    parquet_path = output_dir / "predictions.parquet"
    pl.DataFrame(rows).write_parquet(parquet_path)
    (output_dir / "config.json").write_text("{}\n")
    return parquet_path


# ---------------------------------------------------------------------------
# Demo-only utility — not part of the production API.
# Returns full diffusion trajectories for visualisation in notebooks.
# Does not support batching, multi-GPU, or disk output.
# ---------------------------------------------------------------------------

@dataclass
class CrystalTrajectory:
    crystal: Crystal           # final predicted structure
    trajectory: torch.Tensor  # (steps+1, 3+num_atoms, 3)


@torch.inference_mode()
def sample_trajectory(
    sampler: ClariSampler,
    smiles: str | list[str],
    *,
    id: str | None = None,
    copies: int | list[int] = 4,
    samples: int = 1,
) -> list[CrystalTrajectory]:
    """Demo-only: sample crystal structures and return their full diffusion trajectories."""
    request = _make_request(smiles, id=id, copies=copies, samples=samples)
    template = request_to_crystal(request)
    batch = Crystal.collate([template]).to(sampler.device)

    results = []
    for _ in range(samples):
        with torch.autocast(
            device_type=sampler.device.type,
            dtype=torch.bfloat16,
            enabled=sampler.use_bf16,
        ):
            traj = sampler.lit.sampler.sample(
                sampler.lit.interface,
                sampler.lit.net,
                batch,
                return_trajectory=True,
            )
        traj = traj.squeeze(1).cpu()
        results.append(CrystalTrajectory(
            crystal=template.replace(x=traj[-1]),
            trajectory=traj,
        ))
    return results
