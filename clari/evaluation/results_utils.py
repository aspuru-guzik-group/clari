from pathlib import Path

import polars as pl

from clari.csd import AVAILABLE_CSD_SUBSETS, csd_fam

SOL_TABLE_GROUPS = ("rigid", "flexible", "csp5", "csp6", "csp7", "teaching")
SOL_TABLE_LABELS = {
    "rigid": "Rigid",
    "flexible": "Flexible",
    "csp5": "CSP5",
    "csp6": "CSP6",
    "csp7": "CSP7",
    "teaching": "Teach.",
}


def format_metric_value(value: float | int | None) -> str:
    return "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def load_results(data: str | Path | pl.DataFrame) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        return data
    path = Path(data)
    if not path.is_dir():
        raise ValueError(f"results_utils path inputs must be experiment directories, got: {path}")
    predictions_path = path / "predictions.parquet"
    energies_path = path / "energies.csv"
    compack_path = path / "compack.csv"
    collision_path = path / "collision.csv"
    for source in (predictions_path, energies_path, compack_path, collision_path):
        if not source.is_file():
            raise FileNotFoundError(f"Missing required results file: {source}")
    predictions = pl.read_parquet(predictions_path).select("sample_idx", "id")
    energies = pl.read_csv(energies_path).select("sample_idx", pl.col("energies").alias("energy"))
    compack = pl.read_csv(compack_path)
    collision = pl.read_csv(collision_path).select("sample_idx", "collision")
    return (
        predictions.join(energies, on="sample_idx", how="left")
        .join(compack.drop("id", "energy", "collision", strict=False), on="sample_idx", how="left")
        .join(collision, on="sample_idx", how="left")
    )


def annotate_compack_results(
    data: str | Path | pl.DataFrame, ignore_collisions: bool = False
) -> pl.DataFrame:
    df = load_results(data)
    nmatched = (
        "nmatched"
        if "nmatched" in df.columns
        else "compack_nmatched"
        if "compack_nmatched" in df.columns
        else None
    )
    rmsd = (
        "rmsd" if "rmsd" in df.columns else "compack_rmsd" if "compack_rmsd" in df.columns else None
    )
    if nmatched is None or rmsd is None:
        return df
    solc_pass = (pl.col(nmatched) >= 8) & (pl.col(rmsd) < 2.0)
    if not ignore_collisions and "collision" in df.columns:
        solc_pass &= ~pl.col("collision")
    return df.with_columns(
        (pl.col(nmatched) >= 8).alias("solc_nmatched_ok"),
        (pl.col(rmsd) < 2.0).alias("solc_rmsd_ok"),
        solc_pass.alias("solc_pass"),
    )


def compute_metrics(
    data: str | Path | pl.DataFrame, ignore_collisions: bool = False
) -> dict[str, float | int | None] | None:
    df = annotate_compack_results(data, ignore_collisions=ignore_collisions)
    n_samples, n_targets = len(df), df.select("id").n_unique() if "id" in df.columns else 0
    if n_samples == 0 or n_targets == 0:
        return None
    known_col = df.filter(pl.col("collision").is_not_null()) if "collision" in df.columns else None
    pac = df.filter(pl.col("solc_nmatched_ok")) if "solc_nmatched_ok" in df.columns else None
    solc = (
        df.filter(pl.col("solc_pass").is_not_null()).filter(pl.col("solc_pass"))
        if "solc_pass" in df.columns
        else None
    )
    return {
        "n_samples": n_samples,
        "n_targets": n_targets,
        "ColS": None
        if known_col is None or known_col.is_empty()
        else known_col.filter(pl.col("collision")).height / known_col.height,
        "PacS": None if pac is None else pac.height / n_samples,
        "PacC": None if pac is None else pac.select("id").n_unique() / n_targets,
        "RecS": None,
        "RecC": None,
        "fSolC": None
        if solc is None or solc.is_empty()
        else solc.select("id").n_unique() / n_targets,
    }


def select_topk_ranked_per_id(data: str | Path | pl.DataFrame, *, k: int) -> pl.DataFrame:
    df = load_results(data).with_row_index("row_idx")
    sort_cols = ["id", "row_idx"]
    if "energy" in df.columns:
        sort_cols = ["id", "energy", "row_idx"]
    return (
        df.sort(sort_cols, nulls_last=True)
        .group_by("id", maintain_order=True)
        .head(k)
        .drop("row_idx")
    )


def compute_sol_table(
    data: str | Path | pl.DataFrame | list[str | Path | pl.DataFrame],
    *,
    method: str | None = None,
    k: int,
    n_s: int | None = None,
    group_names: tuple[str, ...] = SOL_TABLE_GROUPS,
    ignore_collisions: bool = False,
) -> pl.DataFrame:
    rows = []
    for item in data if isinstance(data, list) else [data]:
        df = annotate_compack_results(item, ignore_collisions=ignore_collisions)
        max_samples = int(df.group_by("id").len().get_column("len").max()) if n_s is None else n_s
        df = (
            df
            if n_s is None
            else df.group_by("id").map_groups(lambda group: group.sort("sample_idx").head(n_s))
        )
        row = {
            "Method": method or (Path(item).name if isinstance(item, str | Path) else "method"),
            "n_s": max_samples,
            "k": k,
        }
        for group_name in group_names:
            if group_name in df.columns and df.schema[group_name] == pl.Boolean:
                group_df = df.filter(pl.col(group_name))
            elif group_name in AVAILABLE_CSD_SUBSETS and "id" in df.columns:
                families = {csd_fam(cid) for cid in AVAILABLE_CSD_SUBSETS[group_name]}
                group_df = df.filter(
                    pl.col("id").map_elements(csd_fam, return_dtype=pl.Utf8).is_in(families)
                )
            else:
                group_df = next(
                    (
                        df.filter(pl.col(col) == group_name)
                        for col in ("subset", "group", "split")
                        if col in df.columns
                    ),
                    df.clear(),
                )
            eval_df = select_topk_ranked_per_id(group_df, k=k)
            metric = compute_metrics(eval_df, ignore_collisions=ignore_collisions)
            row[SOL_TABLE_LABELS.get(group_name, group_name)] = (
                None if group_df.is_empty() or metric is None else metric["fSolC"]
            )
        rows.append(row)
    return pl.DataFrame(rows)
