from __future__ import annotations

from pathlib import Path

import polars as pl
from jsonargparse import ArgumentParser

from clari.inference.inputs import resolve_predictions_path

SAMPLE_IDX_COLUMN = "sample_idx"
ENERGIES_CSV_COLUMN = "energies"


def rank(
    input_path: Path,
    batch_size: int = 32,
    num_gpus: int = 1,
    torch_threads: int = 1,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Compute FairChem energies and write compact sample rankings.

    This intentionally does not duplicate CIFs from predictions.parquet. The output
    rankings.csv only contains sample_idx, id, energies, and rank.
    """
    predictions_path = resolve_predictions_path(input_path)
    energies_path = predictions_path.with_name("energies.csv")
    rankings_path = predictions_path.with_name("rankings.csv")
    if rankings_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {rankings_path}. Pass --overwrite.")

    from clari.evaluation.compute_energies import main as compute_energies

    compute_energies(
        input_path=input_path,
        batch_size=batch_size,
        num_gpus=num_gpus,
        torch_threads=torch_threads,
        overwrite=overwrite,
    )

    predictions = pl.scan_parquet(predictions_path).select(SAMPLE_IDX_COLUMN, "id").collect()
    energies = pl.read_csv(energies_path)
    rankings = (
        predictions.join(energies, on=SAMPLE_IDX_COLUMN, how="left")
        .with_columns(
            (pl.col(ENERGIES_CSV_COLUMN).rank("ordinal").over("id") - 1)
            .cast(pl.Int64)
            .alias("rank"),
        )
        .sort(["id", "rank"])
    )
    rankings.write_csv(rankings_path)
    print(f"Saved compact rankings to {rankings_path}")
    return rankings


def main() -> None:
    parser = ArgumentParser(description="Rank generated predictions using FairChem UMA energies.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rank(**vars(args))


if __name__ == "__main__":
    main()
