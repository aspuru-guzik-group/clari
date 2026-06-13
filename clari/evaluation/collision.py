from __future__ import annotations

import argparse
import os
from pathlib import Path

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ[_var] = "1"

import polars as pl
from p_tqdm import p_map

from clari.chem import Crystal
from clari.paths import resolve_results_path
from clari.pipelines.utils.metrics import check_clashes

REQUIRED_COLUMNS = ("sample_idx", "id", "cif")
_THREADS_SET = False


def _set_worker_threads() -> None:
    global _THREADS_SET
    if not _THREADS_SET:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _THREADS_SET = True


def _process_row(row: dict) -> dict:
    _set_worker_threads()
    try:
        crystal = Crystal.from_cif(row["cif"]).without_Hs()
        collision = bool(check_clashes(crystal)["clash_rate_oxt"])
    except Exception:
        collision = None
    return {"sample_idx": row["sample_idx"], "id": row["id"], "collision": collision}


def collision(
    experiment_dir: str | Path,
    num_processes: int = 192,
    overwrite: bool = False,
):
    experiment_dir = resolve_results_path(experiment_dir)
    if experiment_dir.is_file() and experiment_dir.name == "predictions.parquet":
        experiment_dir = experiment_dir.parent
    predictions_path = experiment_dir / "predictions.parquet"
    output_path = experiment_dir / "collision.csv"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    if num_processes <= 0:
        raise ValueError(f"num_processes must be positive, got {num_processes}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing predictions parquet: {predictions_path}")

    df = pl.read_parquet(predictions_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {predictions_path}: {missing}")
    if df.select("sample_idx").n_unique() != len(df):
        raise ValueError(f"sample_idx must be unique in {predictions_path}")

    rows = df.select(*REQUIRED_COLUMNS).to_dicts()
    if num_processes == 1:
        results = [_process_row(row) for row in rows]
    else:
        results = p_map(_process_row, rows, num_cpus=num_processes)

    pl.DataFrame(results).write_csv(output_path)
    print(f"Saved {len(results)} collisions to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--num_processes", "--num-processes", type=int, default=192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    collision(**vars(args))


if __name__ == "__main__":
    main()
