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
    make_request,
    request_components,
    resolve_checkpoint,
)
from clari.pipelines.base.lit import LitDiT

H100_REFERENCE_MEMORY_GB = 81.0
MAX_CLASH_RESAMPLE_ROUNDS = 5


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _clash_free(crystals: list[Crystal]) -> list[Crystal]:
    from clari.pipelines.utils.metrics import is_clash_free

    return [c for c in crystals if is_clash_free(c)]


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def request_to_crystal(request: SampleRequest) -> Crystal:
    components = request_components(request)
    # AddHs for plain SMILES; .mol files and molblocks already carry explicit hydrogens.
    # Guard the filesystem stat with a length check (mirrors Crystal.from_smiles): a long
    # SMILES string would otherwise raise OSError("File name too long") from Path.is_file().
    add_hs = [
        not (s.lower().endswith(".mol") or "M  END" in s or (len(s) < 1024 and Path(s).is_file()))
        for s, _ in components
    ]
    return Crystal.from_smiles(components, csd_id=request.id, add_hs=add_hs)


def validate_requests(requests: list[SampleRequest]) -> None:
    """Parse every request's SMILES up front so bad input fails fast.

    Runs before the (slow) checkpoint download and any GPU work, and — for batch
    runs — before any request has sampled, so a single bad SMILES can't leave
    orphaned, un-manifested output behind.
    """
    for request in requests:
        if request.samples < 1:
            raise ValueError(f"Request {request.id!r}: samples must be >= 1, got {request.samples}")
        try:
            request_to_crystal(request)
        except Exception as exc:
            raise ValueError(f"Request {request.id!r}: {exc}") from exc


def build_run_config(
    requests: list[SampleRequest],
    model: str,
    checkpoint: str,
    device: str,
    num_gpus: int,
    batch_size: int | None,
    n_steps: int | None,
    use_ema: bool,
    use_bf16: bool,
    compile: bool | None,
    filter_clashing: bool,
    overwrite: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "checkpoint": checkpoint,
        "device": device,
        "num_gpus": num_gpus,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "use_ema": use_ema,
        "use_bf16": use_bf16,
        "compile": compile,
        "filter_clashing": filter_clashing,
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
    resolved = resolve_device(device)
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(index).total_memory / 1e9
        scale = min(1.0, total_gb / H100_REFERENCE_MEMORY_GB)
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
        ema_weights = (checkpoint.get("ema") or {}).get("ema_weights")
        if ema_weights is not None:
            lit.load_state_dict(ema_weights, strict=False)
        else:
            lit.load_state_dict(checkpoint["state_dict"])
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
        checkpoint: str = "clari-h",
        *,
        device: str | torch.device | None = "auto",
        use_ema: bool = True,
        use_bf16: bool = True,
        n_steps: int | None = 50,
        compile: bool = False,
        torch_threads: int = 1,
        num_gpus: int = 1,
        seed: int | None = None,
        filter_clashing: bool = False,
    ):
        if n_steps is not None and n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        requested_checkpoint = str(checkpoint)
        checkpoint = resolve_checkpoint(checkpoint)
        resolved_device = resolve_device(device)
        torch.set_num_threads(torch_threads)
        torch.set_float32_matmul_precision("high")
        # The model is loaded lazily on first use (see the `lit` property). This keeps the
        # main process weight-free in the multi-GPU path, where it only supplies metadata
        # and the per-GPU workers each load their own copy.
        self._lit: LitDiT | None = None
        self._checkpoint = checkpoint
        self.device = resolved_device
        self.model = requested_checkpoint
        self.checkpoint = str(checkpoint)
        self.use_ema = use_ema
        self.use_bf16 = use_bf16 and resolved_device.type == "cuda"
        self.n_steps = n_steps
        self.compile = compile
        self.torch_threads = torch_threads
        self.num_gpus = num_gpus
        self.seed = seed
        self.filter_clashing = filter_clashing

    @property
    def lit(self) -> LitDiT:
        if self._lit is None:
            self._lit = load_lit(
                self._checkpoint, self.device, self.use_ema, self.n_steps, self.compile
            )
        return self._lit

    @torch.inference_mode()
    def sample_batch(self, crystal: Crystal, count: int, pbar: str | None = None) -> list[Crystal]:
        current = count
        while True:
            batch_gpu = None
            try:
                batch_gpu = Crystal.collate([crystal] * current).to(self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16 and self.device.type == "cuda",
                ):
                    out = self.lit.sampler.sample(
                        self.lit.interface,
                        self.lit.net,
                        batch_gpu,
                        pbar=pbar,
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
        pbar: bool = True,
    ) -> list[Crystal]:
        crystal = request_to_crystal(request)
        target_samples = request.samples if samples is None else samples
        per_batch = min(
            target_samples, auto_batch_size(int(crystal.num_atoms), batch_size, self.device)
        )
        produced: list[Crystal] = []
        while len(produced) < target_samples:
            need = min(per_batch, target_samples - len(produced))
            chunk: list[Crystal] = []
            resample_rounds = 0
            no_progress = False
            while len(chunk) < need:
                missing = need - len(chunk)
                got = self.sample_batch(crystal, missing, pbar="Denoising" if pbar else None)
                if not got:
                    no_progress = True
                    break
                accepted = _clash_free(got) if self.filter_clashing else got
                chunk.extend(accepted[:missing])
                if len(got) < missing:
                    per_batch = max(1, min(per_batch, len(got)))
                if progress is not None:
                    progress.update(min(len(accepted), missing))
                if not self.filter_clashing:
                    break
                if len(accepted) < len(got):
                    resample_rounds += 1
                    if resample_rounds > MAX_CLASH_RESAMPLE_ROUNDS:
                        break
            produced.extend(chunk)
            if no_progress or (self.filter_clashing and len(chunk) < need):
                break
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
            requests = [make_request(smiles, id=id, copies=copies, samples=samples)]
        validate_requests(requests)
        num_gpus = self.num_gpus if num_gpus is None else num_gpus
        if output_dir is None and num_gpus > 1:
            raise ValueError("num_gpus > 1 requires output_dir.")
        seed = self.seed if seed is None else seed
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
            results: list[Crystal] = []
            try:
                for request in requests:
                    results.extend(
                        self.sample_request(
                            request,
                            batch_size=request.batch_size or batch_size,
                            progress=progress,
                            pbar=pbar,
                        )
                    )
            finally:
                if progress is not None:
                    progress.close()
            return results
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
        chunk_size = auto_batch_size(
            int(crystal.num_atoms), request.batch_size or batch_size, device
        )
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
                pbar=pbar,
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
    filter_clashing: bool,
) -> None:
    import sys
    import traceback

    try:
        torch.cuda.set_device(rank)
        sampler = ClariSampler(
            model,
            device=f"cuda:{rank}",
            use_ema=use_ema,
            use_bf16=use_bf16,
            n_steps=n_steps,
            compile=compile,
            torch_threads=torch_threads,
            num_gpus=1,
            filter_clashing=filter_clashing,
        )
        if seed is not None:
            _seed_everything(seed + rank)
        run_chunks(sampler, requests, chunks, Path(shards_dir), pbar=False)
    except Exception:
        error_queue.put((rank, traceback.format_exc()))
        sys.exit(1)


def write_run_config(
    output_dir: Path,
    requests: list[SampleRequest],
    sampler: ClariSampler,
    num_gpus: int,
    batch_size: int | None,
    overwrite: bool,
) -> None:
    (output_dir / "config.json").write_text(
        json.dumps(
            build_run_config(
                requests,
                sampler.model,
                sampler.checkpoint,
                str(sampler.device),
                num_gpus,
                batch_size,
                sampler.n_steps,
                sampler.use_ema,
                sampler.use_bf16,
                sampler.compile,
                sampler.filter_clashing,
                overwrite,
            ),
            indent=2,
        )
        + "\n"
    )


def run_chunks_on_gpus(
    sampler: ClariSampler,
    requests: list[SampleRequest],
    chunks: list[dict[str, int]],
    shards_dir: Path,
    num_gpus: int,
    pbar: bool,
    seed: int | None,
) -> None:
    """Sample every chunk into `shards_dir`, on one GPU or fanned out across many."""
    if num_gpus == 1:
        run_chunks(sampler, requests, chunks, shards_dir, pbar=pbar)
        return
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
                sampler.filter_clashing,
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


def merge_request_shards(
    shards_dir: Path,
    shard_indices: list[int],
    predictions_path: Path,
    sample_idx_offset: int,
) -> None:
    """Merge a request's own shards into its parquet, re-basing sample_idx to 0."""
    paths = [shards_dir / f"shard_{shard_index:06d}.parquet" for shard_index in shard_indices]
    paths = [path for path in paths if path.is_file()]
    if not paths:
        # No shards (e.g. samples == 0): write an empty, well-typed predictions file.
        pl.DataFrame(
            schema={"id": pl.String, "sample_idx": pl.Int64, "cif": pl.String}
        ).write_parquet(predictions_path)
        return
    pl.concat([pl.read_parquet(path) for path in paths], how="vertical").with_columns(
        (pl.col("sample_idx") - sample_idx_offset).alias("sample_idx")
    ).sort("sample_idx").write_parquet(predictions_path)


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
    write_run_config(output_dir, requests, sampler, num_gpus, batch_size, overwrite)
    chunks = build_chunks(requests, batch_size, sampler.device)
    shards_dir = output_dir / ".shards"
    shards_dir.mkdir(exist_ok=True)
    run_chunks_on_gpus(sampler, requests, chunks, shards_dir, num_gpus, pbar, seed)
    merge_shards(shards_dir, output_dir / "predictions.parquet")
    shutil.rmtree(shards_dir)
    return output_dir


def sample_batch_to_directories(
    sampler: ClariSampler,
    *,
    requests: list[SampleRequest],
    output_dirs: list[Path],
    base_dir: Path,
    batch_size: int | None,
    num_gpus: int,
    overwrite: bool,
    pbar: bool,
    seed: int | None = None,
) -> list[Path]:
    """Sample many independent requests, each to its own directory, in one dispatch.

    GPU workers spawn once for the whole batch (the model is loaded once per GPU,
    not once per request) and every chunk lands in a shared shards directory; each
    request's shards are then merged into its own predictions.parquet.
    """
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    if not requests:
        raise ValueError("At least one request is required.")
    if len(output_dirs) != len(requests):
        raise ValueError("output_dirs must have one entry per request.")
    for request, output_dir in zip(requests, output_dirs):
        prepare_output_dir(output_dir, overwrite)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_run_config(output_dir, [request], sampler, num_gpus, batch_size, overwrite)

    chunks = build_chunks(requests, batch_size, sampler.device)
    shards_dir = base_dir / ".shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    run_chunks_on_gpus(sampler, requests, chunks, shards_dir, num_gpus, pbar, seed)

    # Each request's parquet holds 0-based sample_idx, so subtract its base offset
    # (the cumulative sample count of earlier requests, == its first global sample_idx).
    base_offsets: list[int] = []
    running = 0
    for request in requests:
        base_offsets.append(running)
        running += request.samples
    for request_index, output_dir in enumerate(output_dirs):
        shard_indices = [c["shard_index"] for c in chunks if c["request_index"] == request_index]
        merge_request_shards(
            shards_dir,
            shard_indices,
            output_dir / "predictions.parquet",
            base_offsets[request_index],
        )
    shutil.rmtree(shards_dir)
    return list(output_dirs)


@torch.inference_mode()
def sample(
    requests: SampleRequest | list[SampleRequest],
    *,
    model: str = "clari-h",
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
    filter_clashing: bool = False,
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
        filter_clashing=filter_clashing,
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
    return parquet_path


# ---------------------------------------------------------------------------
# Demo-only utility — not part of the production API.
# Returns full diffusion trajectories for visualisation in notebooks.
# Does not support batching, multi-GPU, or disk output.
# ---------------------------------------------------------------------------


@dataclass
class CrystalTrajectory:
    crystal: Crystal  # final predicted structure
    trajectory: torch.Tensor  # (steps+1, 3+num_atoms, 3)


@torch.inference_mode()
def sample_trajectory(
    sampler: ClariSampler,
    smiles: str | list[str],
    *,
    id: str | None = None,
    copies: int | list[int] = 4,
    samples: int = 1,
    filter_clashing: bool = False,
) -> list[CrystalTrajectory]:
    """Demo-only: sample crystal structures and return their full diffusion trajectories."""
    request = make_request(smiles, id=id, copies=copies, samples=samples)
    template = request_to_crystal(request)

    results: list[CrystalTrajectory] = []
    resample_rounds = 0
    while len(results) < samples:
        need = samples - len(results)
        batch = Crystal.collate([template] * need).to(sampler.device)
        with torch.autocast(
            device_type=sampler.device.type,
            dtype=torch.bfloat16,
            enabled=sampler.use_bf16,
        ):
            traj = sampler.lit.sampler.sample(
                sampler.lit.interface,
                sampler.lit.net,
                batch,
                pbar="Denoising steps",
                return_trajectory=True,
            )
        traj = traj.cpu()
        batch_results = [
            CrystalTrajectory(crystal=template.replace(x=traj[:, i][-1]), trajectory=traj[:, i])
            for i in range(need)
        ]
        if filter_clashing:
            from clari.pipelines.utils.metrics import is_clash_free
            batch_results = [r for r in batch_results if is_clash_free(r.crystal)]
            if len(batch_results) < need:
                resample_rounds += 1
                if resample_rounds > MAX_CLASH_RESAMPLE_ROUNDS:
                    results.extend(batch_results)
                    break
        results.extend(batch_results)
        if not filter_clashing:
            break
    return results[:samples]
