from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

SAMPLE_IDX_COLUMN = "sample_idx"
ENERGIES_CSV_COLUMN = "energies"


def _import_compute_energies():
    try:
        from clari.evaluation import compute_energies
    except ImportError as exc:
        raise ImportError(
            "rank requires fairchem-core — install with: pip install 'clari[uma]' "
            "or: uv sync --extra uma"
        ) from exc
    return compute_energies


def _rank_energies(predictions, energies):
    import polars as pl

    return (
        predictions.join(energies, on=SAMPLE_IDX_COLUMN, how="left")
        .with_columns(
            (pl.col(ENERGIES_CSV_COLUMN).rank("ordinal").over("id") - 1)
            .cast(pl.Int64)
            .alias("rank"),
        )
        .sort(["id", "rank"])
    )


def rank(
    input_path: Path | list,
    batch_size: int = 32,
    num_gpus: int = 1,
    torch_threads: int = 1,
    overwrite: bool = False,
):
    """Compute FairChem UMA energies and rank samples within each id group.

    Given a results path, writes energies.csv and rankings.csv next to
    predictions.parquet. Given a list of Crystal objects, ranks fully in
    memory and writes nothing. Returns the rankings DataFrame either way.
    """
    import polars as pl

    compute_energies = _import_compute_energies()

    if isinstance(input_path, list):
        predictions = pl.DataFrame(
            {
                SAMPLE_IDX_COLUMN: list(range(len(input_path))),
                "id": [crystal.csd_id or "sample" for crystal in input_path],
                "cif": [crystal.to_cif() for crystal in input_path],
            }
        )
        energy_values, _ = compute_energies.compute_energies_df(
            predictions, batch_size=batch_size, num_gpus=num_gpus, torch_threads=torch_threads
        )
        energies = pl.DataFrame(
            {
                SAMPLE_IDX_COLUMN: predictions.get_column(SAMPLE_IDX_COLUMN),
                ENERGIES_CSV_COLUMN: energy_values,
            }
        )
        return _rank_energies(predictions.select(SAMPLE_IDX_COLUMN, "id"), energies)

    from clari.inference.inputs import resolve_predictions_path

    predictions_path = resolve_predictions_path(input_path)
    energies_path = predictions_path.with_name("energies.csv")
    rankings_path = predictions_path.with_name("rankings.csv")
    if rankings_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {rankings_path}. Pass --overwrite.")

    compute_energies.main(
        input_path=input_path,
        batch_size=batch_size,
        num_gpus=num_gpus,
        torch_threads=torch_threads,
        overwrite=overwrite,
    )

    predictions = pl.scan_parquet(predictions_path).select(SAMPLE_IDX_COLUMN, "id").collect()
    energies = pl.read_csv(energies_path)
    rankings = _rank_energies(predictions, energies)
    rankings.write_csv(rankings_path)
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
    from clari.inference.inputs import resolve_predictions_path

    print(resolve_predictions_path(args.input_path).with_name("rankings.csv"))


if __name__ == "__main__":
    main()
