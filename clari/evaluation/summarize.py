from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from clari.csd import AVAILABLE_CSD_SUBSETS, csd_fam
from clari.evaluation.results_utils import (
    SOL_TABLE_GROUPS,
    SOL_TABLE_LABELS,
    annotate_compack_results,
    compute_metrics,
    select_topk_ranked_per_id,
)
from clari.paths import resolve_results_path

REQUIRED_PREDICTION_BASE_COLUMNS = ("sample_idx", "id")
REQUIRED_ENERGY_COLUMNS = ("sample_idx", "energies")
REQUIRED_COMPACK_COLUMNS = ("sample_idx", "nmatched", "rmsd")
REQUIRED_COLLISION_COLUMNS = ("sample_idx", "collision")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required results file: {path}")


def _require_unique_sample_idx(df: pl.DataFrame, path: Path) -> None:
    if "sample_idx" not in df.columns:
        raise ValueError(f"Missing required column 'sample_idx' in {path}")
    if df.select("sample_idx").n_unique() != len(df):
        raise ValueError(f"sample_idx must be unique in {path}")


def _require_columns(df: pl.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def _require_same_sample_idx(reference: pl.DataFrame, df: pl.DataFrame, path: Path) -> None:
    reference_keys = reference.select("sample_idx")
    df_keys = df.select("sample_idx")
    missing = reference_keys.join(df_keys, on="sample_idx", how="anti").height
    extra = df_keys.join(reference_keys, on="sample_idx", how="anti").height
    if missing or extra:
        raise ValueError(
            f"sample_idx mismatch in {path}: {missing} missing from predictions, "
            f"{extra} extra not in predictions"
        )


def _load_results(experiment_dir: Path) -> pl.DataFrame:
    predictions_path = experiment_dir / "predictions.parquet"
    energies_path = experiment_dir / "energies.csv"
    compack_path = experiment_dir / "compack.csv"
    collision_path = experiment_dir / "collision.csv"
    for path in (predictions_path, energies_path, compack_path, collision_path):
        _require_file(path)

    predictions = pl.read_parquet(predictions_path)
    energies = pl.read_csv(energies_path)
    compack = pl.read_csv(compack_path)
    collision = pl.read_csv(collision_path)

    _require_columns(predictions, REQUIRED_PREDICTION_BASE_COLUMNS, predictions_path)
    _require_columns(energies, REQUIRED_ENERGY_COLUMNS, energies_path)
    _require_columns(compack, REQUIRED_COMPACK_COLUMNS, compack_path)
    _require_columns(collision, REQUIRED_COLLISION_COLUMNS, collision_path)
    for df, path in (
        (predictions, predictions_path),
        (energies, energies_path),
        (compack, compack_path),
        (collision, collision_path),
    ):
        _require_unique_sample_idx(df, path)
    _require_same_sample_idx(predictions, energies, energies_path)
    _require_same_sample_idx(predictions, compack, compack_path)
    _require_same_sample_idx(predictions, collision, collision_path)

    return (
        predictions.select("sample_idx", "id")
        .join(
            energies.select("sample_idx", pl.col("energies").alias("energy")),
            on="sample_idx",
            how="left",
        )
        .join(compack.drop("id", "energy", strict=False), on="sample_idx", how="left")
        .join(collision.select("sample_idx", "collision"), on="sample_idx", how="left")
        .with_columns(pl.col("collision").cast(pl.Boolean, strict=False))
    )


def _solc(df: pl.DataFrame) -> float | None:
    metrics = compute_metrics(annotate_compack_results(df))
    return None if metrics is None or metrics["fSolC"] is None else float(metrics["fSolC"])


def _group_df(df: pl.DataFrame, group_name: str) -> pl.DataFrame:
    if group_name == "all":
        return df
    if group_name in df.columns and df.schema[group_name] == pl.Boolean:
        return df.filter(pl.col(group_name))
    if group_name in AVAILABLE_CSD_SUBSETS:
        families = {csd_fam(cid) for cid in AVAILABLE_CSD_SUBSETS[group_name]}
        return df.filter(pl.col("id").map_elements(csd_fam, return_dtype=pl.Utf8).is_in(families))
    return df.clear()


def _group_solc(df: pl.DataFrame, group: str, k: int | None = None) -> float | None:
    group_df = _group_df(df, group)
    if group_df.is_empty():
        return None
    return _solc(group_df if k is None else select_topk_ranked_per_id(group_df, k=k))


def _normalize_ks(ks: tuple[int, ...] | None, n_s: int) -> tuple[int, ...]:
    if not ks:
        return (int(n_s),)
    seen: set[int] = set()
    out: list[int] = []
    for k in ks:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k > n_s:
            raise ValueError(f"k={k} exceeds n_s={n_s} (samples per structure)")
        if k not in seen:
            seen.add(k)
            out.append(k)
    return tuple(out)


def _timing_summary_table(experiment_dir: Path) -> pl.DataFrame | None:
    path = experiment_dir / "timing.csv"
    if not path.is_file():
        return None
    timing = pl.read_csv(path)
    required = ("timing_group", "sample_gpu_ms")
    if timing.is_empty() or not all(c in timing.columns for c in required):
        return None
    out = timing.group_by("timing_group").agg(
        (pl.col("sample_gpu_ms").sum() / 1000.0).alias("seconds")
    )
    preferred = list(SOL_TABLE_GROUPS) + ["other"]
    present = out.get_column("timing_group").to_list()
    extras = sorted(x for x in present if x not in preferred)
    order_df = pl.DataFrame(
        {"timing_group": preferred + extras, "_sort": range(len(preferred) + len(extras))}
    )
    out = (
        out.join(order_df, on="timing_group", how="left")
        .sort(by=["_sort", "timing_group"], nulls_last=True)
        .drop("_sort")
    )
    labels = {**SOL_TABLE_LABELS, "other": "Other"}
    tall = out.with_columns(
        pl.col("timing_group").replace(labels, default=pl.col("timing_group")).alias("col")
    )
    cols = tall.get_column("col").to_list()
    secs = tall.get_column("seconds").to_list()
    total_seconds = out.select(pl.col("seconds").sum()).item()
    data = {c: [s] for c, s in zip(cols, secs)}
    data["All"] = [total_seconds]
    return pl.DataFrame(data)


def _print_table(df: pl.DataFrame) -> None:
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(df.with_columns(pl.col(pl.Float64).round(4)))


def summarize(
    experiment_dir: str | Path,
    *,
    ks: tuple[int, ...] | None = None,
    timing_only: bool = False,
    teaching: bool = False,
    save: bool = False,
) -> pl.DataFrame:
    experiment_dir = resolve_results_path(experiment_dir)
    if experiment_dir.is_file() and experiment_dir.name == "predictions.parquet":
        experiment_dir = experiment_dir.parent
    if experiment_dir.is_file() and experiment_dir.name == "timing.csv":
        experiment_dir = experiment_dir.parent
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    if timing_only:
        timing_tbl = _timing_summary_table(experiment_dir)
        if timing_tbl is None:
            path = experiment_dir / "timing.csv"
            raise FileNotFoundError(
                f"timing_only requires a readable {path} with columns "
                "timing_group and sample_gpu_ms and at least one row"
            )
        _print_table(timing_tbl)
        if save:
            timing_tbl.write_csv(experiment_dir / "timing_results.csv")
        return timing_tbl

    df = _load_results(experiment_dir)
    if df.is_empty():
        raise ValueError(f"No result rows found in {experiment_dir}")
    counts = df.group_by("id").len().get_column("len")
    raw_n = counts.min()
    if raw_n is None or raw_n <= 0:
        raise ValueError(f"No result rows found in {experiment_dir}")
    n_s = int(raw_n)
    k_values = _normalize_ks(ks, n_s)
    groups = (
        ("all",) if teaching else tuple(name for name in SOL_TABLE_GROUPS if name != "teaching")
    )
    rows = []
    for k in k_values:
        row = {"Method": experiment_dir.name, "n_s": n_s, "k": k}
        for group in groups:
            row["Sol" if group == "all" else SOL_TABLE_LABELS.get(group, group)] = _group_solc(
                df, group, None if k == n_s else k
            )
        rows.append(row)
    summary = pl.DataFrame(rows).sort("k")

    print(f"Experiment: {experiment_dir}")
    _print_table(summary)
    if save:
        summary.write_csv(experiment_dir / "results.csv")

    timing_tbl = _timing_summary_table(experiment_dir)
    if timing_tbl is not None:
        print()
        _print_table(timing_tbl)
        if save:
            timing_tbl.write_csv(experiment_dir / "timing_results.csv")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        nargs="+",
        action="append",
        dest="ks",
        default=None,
        metavar="K",
        help="Energy top-k per structure.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Only print timing: one row, columns are groups, values are total GPU seconds (sum of chunk timings).",
    )
    parser.add_argument(
        "--teaching",
        action="store_true",
        help="Print one Sol metric over the whole teaching dataset instead of Oxtal subset columns.",
    )
    parser.add_argument("--save", action="store_true", help="Save printed tables to CSV files.")
    args = parser.parse_args()
    ks = tuple(k for group in args.ks for k in group) if args.ks else None
    summarize(
        args.experiment_dir,
        ks=ks,
        timing_only=args.timing,
        teaching=args.teaching,
        save=args.save,
    )


if __name__ == "__main__":
    main()
