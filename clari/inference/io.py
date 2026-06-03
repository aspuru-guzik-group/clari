from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from rdkit import Chem

from clari.paths import resolve_results_path

CLARI_CONFIG_NAME = "config.json"
KNOWN_OUTPUT_FILES = (
    "predictions.parquet",
    CLARI_CONFIG_NAME,
    "rankings.csv",
    "energies.csv",
    "energy_timing.csv",
    "timing.csv",
)
KNOWN_OUTPUT_DIRS = (".shards", "cifs")


def resolve_predictions_path(input_path: str | Path) -> Path:
    input_path = resolve_results_path(input_path)
    if input_path.is_dir():
        input_path = input_path / "predictions.parquet"
    if not input_path.is_file():
        raise FileNotFoundError(f"Predictions parquet does not exist: {input_path}")
    return input_path


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def write_predictions(
    samples: list,
    *,
    output_dir: str | Path,
    config: dict[str, Any],
    overwrite: bool = False,
) -> None:
    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, overwrite=overwrite)
    pl.DataFrame([sample.to_row() for sample in samples]).write_parquet(
        output_dir / "predictions.parquet"
    )
    (output_dir / CLARI_CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")


def write_shard(shards_dir: Path, chunk, samples: list) -> None:
    if len(samples) != chunk.count:
        raise RuntimeError(f"Expected {chunk.count} samples for {chunk.id}, got {len(samples)}")
    path = shards_dir / f"shard_{chunk.shard_idx:06d}.parquet"
    if path.exists():
        raise FileExistsError(f"Shard already exists: {path}")
    pl.DataFrame([sample.to_row() for sample in samples]).write_parquet(path)


def merge_shards(shards_dir: Path, predictions_path: Path, chunks: list) -> None:
    paths = sorted(shards_dir.glob("shard_*.parquet"))
    if len(paths) != len(chunks):
        raise RuntimeError(f"Expected {len(chunks)} shards, found {len(paths)}")
    predictions = pl.concat([pl.read_parquet(path) for path in paths], how="vertical").sort(
        "sample_idx"
    )
    expected_rows = sum(chunk.count for chunk in chunks)
    if len(predictions) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} predictions, got {len(predictions)}")
    if predictions.get_column("sample_idx").n_unique() != expected_rows:
        raise RuntimeError("Duplicate sample_idx values in prediction shards")
    predictions.write_parquet(predictions_path)


def run_config(
    *,
    checkpoint_path: str | Path | None,
    requests: list | None,
    batch_size: int | None = None,
    num_gpus: int | None = None,
    device: str | None = None,
    use_ema: bool | None = None,
    use_bf16: bool | None = None,
    n_steps: int | None = None,
    compile: bool | None = None,
    torch_threads: int | None = None,
    keep_shards: bool | None = None,
    filter_clashing: bool | None = None,
    max_resample_factor: int | None = None,
    chunks: list | None = None,
    num_samples: int | None = None,
) -> dict[str, Any]:
    if num_samples is None and requests is not None:
        num_samples = sum(request.n_samples for request in requests)
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "batch_size": batch_size,
        "num_gpus": num_gpus,
        "device": device,
        "use_ema": use_ema,
        "use_bf16": use_bf16,
        "n_steps": n_steps,
        "compile": compile,
        "torch_threads": torch_threads,
        "keep_shards": keep_shards,
        "filter_clashing": filter_clashing,
        "max_resample_factor": max_resample_factor,
        "num_samples": num_samples,
        "num_chunks": len(chunks) if chunks is not None else None,
        "requests": [request_config(request) for request in requests] if requests else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": git_commit(),
    }
    return {key: value for key, value in config.items() if value is not None}


def request_config(request) -> dict[str, Any]:
    return {
        "id": request.id,
        "smiles": request.smiles,
        "copies": request.copies,
        "n_samples": request.n_samples,
        "metadata": request.metadata,
        "crystal_id": str(request.crystal.csd_id) if request.crystal is not None else None,
        "rdmol_smiles": rdmol_config(request.rdmol) if request.rdmol is not None else None,
    }


def rdmol_config(rdmol) -> str | list:
    if isinstance(rdmol, Chem.Mol):
        return Chem.MolToSmiles(rdmol)
    if _is_rdmol_pair(rdmol):
        return [Chem.MolToSmiles(rdmol[0]), rdmol[1]]
    if isinstance(rdmol, (list, tuple)):
        return [[Chem.MolToSmiles(mol), copies] for mol, copies in rdmol]
    raise TypeError(f"Unsupported RDKit molecule input type: {type(rdmol)!r}")


def _is_rdmol_pair(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], Chem.Mol)
        and isinstance(value[1], int)
    )


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not output_dir.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
    if not any(output_dir.iterdir()):
        return
    if not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not looks_like_clari_output(output_dir):
        raise FileExistsError(
            f"Refusing to overwrite non-CLARI directory: {output_dir}. "
            f"Expected an existing {CLARI_CONFIG_NAME} or predictions.parquet."
        )
    remove_known_outputs(output_dir)


def looks_like_clari_output(output_dir: Path) -> bool:
    config_path = output_dir / CLARI_CONFIG_NAME
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            return (output_dir / "predictions.parquet").is_file()
        return isinstance(config, dict) and (
            "requests" in config or "checkpoint_path" in config or "num_samples" in config
        )
    return (output_dir / "predictions.parquet").is_file()


def remove_known_outputs(output_dir: Path) -> None:
    for name in KNOWN_OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            if not path.is_file():
                raise FileExistsError(f"Expected generated file but found non-file: {path}")
            path.unlink()
    for name in KNOWN_OUTPUT_DIRS:
        path = output_dir / name
        if path.exists():
            if not path.is_dir():
                raise FileExistsError(
                    f"Expected generated directory but found non-directory: {path}"
                )
            shutil.rmtree(path)
