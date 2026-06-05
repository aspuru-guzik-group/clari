from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

warnings.filterwarnings("ignore")

import jsonargparse
import polars as pl
import torch
from tqdm import tqdm

from clari.csd import AVAILABLE_CSD_SUBSETS, csd_fam
from clari.datamodules import CrystalDataModule
from clari.inference.core import resolve_hub_checkpoint
from clari.paths import RESULTS_DIR
from clari.pipelines.base.lit import LitDiT

DATA_CONFIG = {"batch_size": 1}
SUBSETS = ("oxtal", "teaching")
TIMING_WARMUP_BATCHES = 5
CSD_TIMING_GROUPS = ("csp5", "csp6", "csp7", "rigid", "flexible", "teaching")
CSD_FAM_TO_TIMING_GROUP = {}
for _group in CSD_TIMING_GROUPS:
    for _cid in AVAILABLE_CSD_SUBSETS[_group]:
        CSD_FAM_TO_TIMING_GROUP.setdefault(csd_fam(_cid), _group)
EXP_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SENTINEL = object()


def chunk_for(n):
    return 1 if n > 1000 else 1000 if n < 200 else 500 if n < 300 else 200 if n < 500 else 25


def make_dm(subset, families=None):
    if subset not in SUBSETS:
        raise ValueError(f"Unsupported subset: {subset}. Use one of: {SUBSETS}.")
    opts = {"group_by_fam": True, "random_repr": False}
    opts["filter_fams"] = families or list(dict.fromkeys(AVAILABLE_CSD_SUBSETS[subset]))
    return CrystalDataModule(
        **DATA_CONFIG,
        num_workers=0,
        collate_fn=None,
        remove_Hs=False,
        split_opts={"test": opts},
    )


def crystal_id(batch):
    ids = batch.csd_id if isinstance(batch.csd_id, tuple) else (batch.csd_id,)
    if len(ids) != 1:
        raise ValueError(f"Expected one test ID per batch, got {ids}")
    return str(ids[0])


def timing_group(cid):
    return CSD_FAM_TO_TIMING_GROUP.get(csd_fam(cid), "other")


def planned_ids(subset, families=None):
    ids = [crystal_id(batch) for batch in make_dm(subset, families).test_dataloader()]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate test IDs in evaluation split")
    return ids


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return None


def set_threads(n):
    if n <= 0:
        raise ValueError(f"torch_threads must be positive, got {n}")
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass


def load_lit(ckpt_path, use_ema, device, n_steps):
    configure_model = LitDiT.configure_model
    LitDiT.configure_model = lambda self: None
    try:
        lit = LitDiT.load_from_checkpoint(ckpt_path, map_location="cpu")
    finally:
        LitDiT.configure_model = configure_model
    if use_ema:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        weights = (ckpt.get("ema") or {}).get("ema_weights") if isinstance(ckpt, dict) else None
        if isinstance(weights, dict):
            lit.load_state_dict(weights, strict=False)
    if n_steps is not None:
        lit.sampler.num_steps = int(n_steps)
    lit = lit.to(device).eval()
    import torch._inductor.config as cfg

    cfg.online_softmax = False
    lit.configure_model()
    return lit


def replicate(batch, n):
    crystals = batch.unbatch()
    return type(batch).collate([crystal for crystal in crystals for _ in range(n)])


def split_chunk_trajectories(traj_cpu, n_kept):
    if traj_cpu is None or n_kept == 0:
        return [None] * n_kept
    T1, total, *rest = traj_cpu.shape
    if total % n_kept:
        raise ValueError(f"Trajectory atom count {total} is not divisible by chunk size {n_kept}")
    per_atoms = total // n_kept
    reshaped = traj_cpu.view(T1, n_kept, per_atoms, *rest)
    return [reshaped[:, k].clone() for k in range(n_kept)]


def writer_loop(queue, samples_dir, trajectory_dir, errors):
    while True:
        item = queue.get()
        try:
            if item is SENTINEL:
                return
            cid, idxs, crystals, trajs = item
            path = samples_dir / f"{cid}.parquet"
            if path.exists():
                raise FileExistsError(f"Duplicate sample shard path: {path}")
            data = {
                "sample_idx": idxs,
                "id": [str(c.csd_id) for c in crystals],
                "cif": [c.to_cif() for c in crystals],
            }
            pl.DataFrame(data).write_parquet(path)
            if any(t is not None for t in trajs):
                if trajectory_dir is None:
                    raise ValueError(
                        "Trajectory rows were produced without a trajectory output dir"
                    )
                trajectory_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = trajectory_dir / f"{cid}.parquet"
                if trajectory_path.exists():
                    raise FileExistsError(f"Duplicate trajectory shard path: {trajectory_path}")
                pl.DataFrame(
                    {
                        "sample_idx": idxs,
                        "id": [str(c.csd_id) for c in crystals],
                        "trajectory": [t.tolist() if t is not None else None for t in trajs],
                    }
                ).write_parquet(trajectory_path)
        except BaseException as exc:
            errors.append(exc)
        finally:
            queue.task_done()


def sample_chunk(lit, batch, n, device, use_bf16, timing=False, return_traj=False):
    while True:
        batch_gpu = None
        try:
            batch_gpu = replicate(batch, n).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16 and device.type == "cuda",
            ):
                if timing:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    out = lit.sampler.sample(
                        lit.interface, lit.net, batch_gpu, return_trajectory=return_traj
                    )
                    end.record()
                else:
                    out = lit.sampler.sample(
                        lit.interface, lit.net, batch_gpu, return_trajectory=return_traj
                    )
            if return_traj:
                traj_cpu = out.cpu()
                pred_gpu = batch_gpu.replace(x=out[-1])
            else:
                traj_cpu = None
                pred_gpu = out
            if timing:
                end.synchronize()
                gpu_ms = start.elapsed_time(end)
            pred = pred_gpu.cpu()
            del batch_gpu
            crystals = pred.unbatch()
            row = (
                {"sample_batch_size": n, "produced_n": len(crystals), "sample_gpu_ms": gpu_ms}
                if timing
                else None
            )
            return crystals, row, traj_cpu
        except RuntimeError as exc:
            if device.type != "cuda" or "out of memory" not in str(exc).lower() or n <= 1:
                raise
            if batch_gpu is not None:
                del batch_gpu
            torch.cuda.empty_cache()
            n = max(1, n // 2)


@torch.no_grad()
def sample_families(
    rank,
    families,
    checkpoint_path,
    n_samples,
    chunk_size,
    use_ema,
    use_bf16,
    subset,
    offsets,
    samples_dir,
    torch_threads,
    n_steps,
    timing=False,
    timing_dir=None,
    traj=False,
    trajectory_dir=None,
    tqdm_lock=None,
):
    if not families:
        return
    if tqdm_lock is not None:
        tqdm.set_lock(tqdm_lock)
    set_threads(torch_threads)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    lit = load_lit(checkpoint_path, use_ema, device, n_steps)
    dm = make_dm(subset, families)
    queue, errors = Queue(maxsize=2), []
    writer = Thread(target=writer_loop, args=(queue, samples_dir, trajectory_dir, errors))
    writer.start()
    timing_rows = []
    iterator = iter(dm.test_dataloader())
    progress = tqdm(
        iterator,
        total=len(dm.datasets["test"]),
        desc=f"GPU {rank}",
        unit="id",
        position=rank,
        leave=False,
        dynamic_ncols=True,
    )
    try:
        for batch in progress:
            if errors:
                raise RuntimeError("Sample writer failed") from errors[0]
            cid = crystal_id(batch)
            if cid not in offsets:
                continue
            atoms = (
                batch.num_atoms.tolist() if torch.is_tensor(batch.num_atoms) else batch.num_atoms
            )
            n_atoms = int(max(atoms)) if isinstance(atoms, list) else int(atoms)
            active = min(n_samples, chunk_size or chunk_for(n_atoms))
            if timing:
                warmup_done = 0
                for _ in range(TIMING_WARMUP_BATCHES):
                    if warmup_done >= n_samples:
                        break
                    chunk, _, _ = sample_chunk(
                        lit,
                        batch,
                        min(active, n_samples - warmup_done),
                        device,
                        use_bf16,
                        timing=False,
                    )
                    active = min(active, len(chunk))
                    warmup_done += len(chunk)
                    del chunk
            idxs, crystals, trajs, done, next_idx = [], [], [], 0, offsets[cid]
            chunk_idx = 0
            while done < n_samples:
                chunk, row, traj_cpu = sample_chunk(
                    lit,
                    batch,
                    min(active, n_samples - done),
                    device,
                    use_bf16,
                    timing,
                    return_traj=traj,
                )
                active = min(active, len(chunk))
                if row is not None:
                    timing_rows.append(
                        {
                            "rank": rank,
                            "subset": subset,
                            "timing_group": timing_group(cid),
                            "csd_id": cid,
                            "n_atoms": n_atoms,
                            "chunk_idx": chunk_idx,
                            "done_before": done,
                            "kept_n": len(chunk),
                            **row,
                        }
                    )
                trajs.extend(split_chunk_trajectories(traj_cpu, len(chunk)))
                idxs.extend(range(next_idx, next_idx + len(chunk)))
                crystals.extend(chunk)
                done += len(chunk)
                next_idx += len(chunk)
                chunk_idx += 1
            queue.put((cid, idxs, crystals, trajs))
    finally:
        queue.put(SENTINEL)
        queue.join()
        writer.join()
        progress.close()
    if errors:
        raise RuntimeError("Sample writer failed") from errors[0]
    if timing and timing_dir is not None:
        timing_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(timing_rows).write_csv(timing_dir / f"rank={rank}.csv")


def run_workers(num_gpus, families, kwargs):
    if num_gpus == 1:
        sample_families(rank=0, families=families, **kwargs)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
        raise RuntimeError(f"Requested {num_gpus} GPUs, found {torch.cuda.device_count()}")
    ctx, lock = mp.get_context("spawn"), None
    lock = ctx.RLock()
    procs = [
        ctx.Process(
            target=sample_families,
            kwargs={
                **kwargs,
                "rank": rank,
                "families": families[rank::num_gpus],
                "tqdm_lock": lock,
            },
        )
        for rank in range(num_gpus)
        if families[rank::num_gpus]
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Sampling worker failed with exit code {proc.exitcode}")


@torch.no_grad()
def main(
    model_or_checkpoint: str,
    n_samples: int,
    experiment_name: str,
    chunk_size: int | None = None,
    use_ema: bool = True,
    use_bf16: bool = True,
    subset: str = "oxtal",
    num_gpus: int = 1,
    torch_threads: int = 1,
    n_steps: int | None = None,
    timing: bool = False,
    family: str | None = None,
    traj: bool = False,
):
    if n_samples <= 0 or num_gpus <= 0 or torch_threads <= 0:
        raise ValueError("n_samples, num_gpus, and torch_threads must be positive")
    if n_steps is not None and n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not EXP_RE.fullmatch(experiment_name):
        raise ValueError(
            "experiment_name must contain only letters, numbers, dots, underscores, and dashes"
        )
    subset = subset.strip().lower()
    out = RESULTS_DIR / experiment_name
    samples_dir, pred_path = out / "samples", out / "predictions.parquet"
    trajectory_dir, trajectory_path = out / "trajectories", out / "trajectories.parquet"
    timing_dir = out / ".timing"
    timing_path = out / "timing.csv"
    if out.exists():
        raise FileExistsError(f"Experiment output already exists: {out}")
    source = model_or_checkpoint.strip()
    hub_model = source.lower() if source.lower() in ("clari-m", "clari-l") else None
    if hub_model is not None:
        ckpt_path = Path(resolve_hub_checkpoint(hub_model))
    else:
        ckpt_path = Path(source)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    family_filter = None
    id_filter = None
    if family is not None:
        raw = family.strip().upper()
        fam = csd_fam(raw)
        all_families = {csd_fam(cid) for cid in AVAILABLE_CSD_SUBSETS[subset]}
        if fam not in all_families:
            raise ValueError(f"Unknown family {family!r} for subset {subset!r}")
        family_filter = [fam]
        if raw != fam:
            id_filter = raw

    ids = planned_ids(subset, family_filter)
    if id_filter is not None:
        if id_filter not in ids:
            raise ValueError(f"CSD ID {id_filter!r} not in subset {subset!r}")
        ids = [id_filter]
    families = sorted({csd_fam(cid) for cid in ids})
    offsets = {cid: i * n_samples for i, cid in enumerate(ids)}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    artifact = ckpt.get("artifact") if isinstance(ckpt, dict) else None
    samples_dir.mkdir(parents=True)
    config = dict(
        experiment_name=experiment_name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        checkpoint_path=str(ckpt_path),
        checkpoint_source=source,
        hub_model=hub_model,
        checkpoint_artifact=artifact,
        n_samples=n_samples,
        use_ema=use_ema,
        use_bf16=use_bf16,
        subset=subset,
        family=family,
        num_gpus=num_gpus,
        n_steps=n_steps,
        traj=traj,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        git_commit=git_commit(),
    )
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    run_workers(
        num_gpus,
        families,
        dict(
            checkpoint_path=ckpt_path,
            n_samples=n_samples,
            chunk_size=chunk_size,
            use_ema=use_ema,
            use_bf16=use_bf16,
            subset=subset,
            offsets=offsets,
            samples_dir=samples_dir,
            torch_threads=torch_threads,
            n_steps=n_steps,
            timing=timing,
            timing_dir=timing_dir,
            traj=traj,
            trajectory_dir=(trajectory_dir if traj else None),
        ),
    )

    paths = sorted(samples_dir.glob("*.parquet"))
    if len(paths) != len(ids):
        raise RuntimeError(f"Expected {len(ids)} sample shards, found {len(paths)}")
    preds = pl.concat([pl.read_parquet(path) for path in paths], how="vertical").sort("sample_idx")
    if len(preds) != len(ids) * n_samples:
        raise RuntimeError(f"Expected {len(ids) * n_samples} samples, got {len(preds)}")
    preds.write_parquet(pred_path)
    print(f"Saved per-id samples to {samples_dir}")
    print(f"Saved merged predictions to {pred_path}")
    if traj:
        trajectory_paths = sorted(trajectory_dir.glob("*.parquet"))
        if len(trajectory_paths) != len(ids):
            raise RuntimeError(
                f"Expected {len(ids)} trajectory shards, found {len(trajectory_paths)}"
            )
        trajectories = pl.concat(
            [pl.read_parquet(path) for path in trajectory_paths],
            how="vertical",
        ).sort("sample_idx")
        if len(trajectories) != len(ids) * n_samples:
            raise RuntimeError(
                f"Expected {len(ids) * n_samples} trajectories, got {len(trajectories)}"
            )
        trajectories.write_parquet(trajectory_path)
        for path in trajectory_paths:
            path.unlink()
        trajectory_dir.rmdir()
        print(f"Saved trajectories to {trajectory_path}")
    if timing:
        timing_paths = sorted(timing_dir.glob("rank=*.csv"))
        timings = pl.concat([pl.read_csv(path) for path in timing_paths], how="vertical")
        timings.write_csv(timing_path)
        for path in timing_paths:
            path.unlink()
        timing_dir.rmdir()
        print(f"Saved sampler timings to {timing_path}")


def cli():
    p = jsonargparse.ArgumentParser()
    p.add_argument("model_or_checkpoint", type=str)
    p.add_argument("n_samples", type=int)
    p.add_argument("experiment_name", type=str)
    p.add_argument("--chunk_size", type=int, default=None)
    p.add_argument("--use_ema", type=bool, default=True)
    p.add_argument("--use_bf16", type=bool, default=True)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--subset", type=str, default="oxtal", choices=SUBSETS)
    p.add_argument(
        "--time", "--timing", "--timings", action="store_true", dest="timing", default=False
    )
    p.add_argument("--family", type=str, default=None)
    p.add_argument("--traj", action="store_true", default=False)
    main(**vars(p.parse_args()))


if __name__ == "__main__":
    cli()
