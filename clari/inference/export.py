from __future__ import annotations

import re
import shutil
from pathlib import Path

import polars as pl
from jsonargparse import ArgumentParser

from clari.chem import Crystal
from clari.inference.inputs import resolve_predictions_path

SAMPLE_IDX_COLUMN = "sample_idx"
ID_COLUMN = "id"
CIF_COLUMN = "cif"
RANK_COLUMN = "rank"


def export_cifs(
    input_path: Path | list[Crystal],
    output_dir: Path | None = None,
    rankings_path: Path | None = None,
    top_k: int | None = None,
    ids: str | list[str] | None = None,
    sample_idx: int | list[int] | None = None,
    overwrite: bool = False,
    id: str | None = None,
) -> None:
    """Export CIF files from a predictions.parquet results file, directory, or list of Crystal objects."""
    if isinstance(input_path, list):
        if output_dir is None:
            raise ValueError("output_dir is required when exporting from a list of Crystal objects.")
        output_dir = Path(output_dir)
        compound_dir = output_dir / _safe_path_part(id or "samples")
        if compound_dir.exists() and not overwrite:
            raise FileExistsError(f"Output directory already exists: {compound_dir}. Pass --overwrite.")
        compound_dir.mkdir(parents=True, exist_ok=True)
        for idx, crystal in enumerate(input_path):
            (compound_dir / f"sample_{idx:06d}.cif").write_text(crystal.to_cif())
        print(f"Exported {len(input_path)} CIFs to {compound_dir}")
        return

    predictions_path = resolve_predictions_path(input_path)
    output_dir = output_dir or predictions_path.with_name("cifs")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite.")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pl.scan_parquet(predictions_path).select(SAMPLE_IDX_COLUMN, ID_COLUMN, CIF_COLUMN)
    ids_list = _as_list(ids)
    sample_idx_list = _as_list(sample_idx)
    if ids_list is not None:
        df = df.filter(pl.col(ID_COLUMN).is_in(ids_list))
    if sample_idx_list is not None:
        df = df.filter(pl.col(SAMPLE_IDX_COLUMN).is_in(sample_idx_list))
    df = df.collect()

    rankings_path = _resolve_rankings_path(predictions_path, rankings_path)
    if rankings_path is not None:
        rankings = pl.read_csv(rankings_path).select(SAMPLE_IDX_COLUMN, ID_COLUMN, RANK_COLUMN)
        df = df.join(rankings, on=[SAMPLE_IDX_COLUMN, ID_COLUMN], how="left")
        if top_k is not None:
            if top_k <= 0:
                raise ValueError(f"top_k must be positive, got {top_k}")
            df = df.filter(pl.col(RANK_COLUMN) < top_k)
        df = df.sort([ID_COLUMN, RANK_COLUMN, SAMPLE_IDX_COLUMN])
    elif top_k is not None:
        raise ValueError("top_k requires rankings.csv. Run `uv run rank` first.")
    else:
        df = df.sort([ID_COLUMN, SAMPLE_IDX_COLUMN])

    for row in df.iter_rows(named=True):
        compound_dir = output_dir / _safe_path_part(row[ID_COLUMN])
        compound_dir.mkdir(parents=True, exist_ok=True)
        sample_idx_value = int(row[SAMPLE_IDX_COLUMN])
        rank_value = row.get(RANK_COLUMN)
        if rank_value is None:
            name = f"sample_{sample_idx_value:06d}.cif"
        else:
            name = f"rank_{int(rank_value):04d}_sample_{sample_idx_value:06d}.cif"
        path = compound_dir / name
        if path.exists() and not overwrite:
            raise FileExistsError(f"CIF already exists: {path}. Pass --overwrite.")
        path.write_text(row[CIF_COLUMN])

    print(f"Exported {len(df)} CIFs to {output_dir}")


def _resolve_rankings_path(predictions_path: Path, rankings_path: Path | None) -> Path | None:
    if rankings_path is not None:
        rankings_path = Path(rankings_path)
        if not rankings_path.is_file():
            raise FileNotFoundError(f"Rankings CSV does not exist: {rankings_path}")
        return rankings_path
    candidate = predictions_path.with_name("rankings.csv")
    return candidate if candidate.is_file() else None


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "unknown"


def main() -> None:
    parser = ArgumentParser(
        description="Export CIF files from CLARI predictions.parquet, optionally using rankings.csv."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--rankings_path", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--ids", action="append", default=None)
    parser.add_argument("--sample_idx", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.sample_idx is not None:
        args.sample_idx = [int(value) for value in args.sample_idx]
    export_cifs(**vars(args))


if __name__ == "__main__":
    main()
