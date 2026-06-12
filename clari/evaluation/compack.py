# /// script
# requires-python = "==3.11.*"
# dependencies = ["csd-python-api>=3.7.0", "polars", "tqdm"]
#
# [[tool.uv.index]]
# name = "ccdc"
# url = "https://pip.ccdc.cam.ac.uk/"
#
# [tool.uv.sources]
# csd-python-api = { index = "ccdc" }
# ///
import json
import os
import pickle
import select
import signal
import time
from argparse import ArgumentParser
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import polars as pl
from ccdc.crystal import Crystal as CCDCCrystal
from ccdc.crystal import PackingSimilarity
from tqdm import tqdm

DATA_DIR = Path(os.environ.get("CLARI_DATA_DIR", Path.cwd() / "data"))
RESULTS_DIR = Path(os.environ.get("CLARI_RESULTS_DIR", Path.cwd() / "results"))
GT_CIFS_PATH = DATA_DIR / "csd" / "test_cifs.parquet"

SETTINGS = dict(
    allow_artificial_inversion=True,
    ignore_smallest_components=False,
    allow_molecular_differences=False,
    match_entire_packing_shell=False,
    angle_tolerance=75.0,
    molecular_similarity_threshold=0.2,
    distance_tolerance=0.50,
    packing_shell_size=15,
    ignore_bond_counts=True,
    show_highest_similarity_result=False,
    ignore_bond_types=True,
    skip_when_identifiers_equal=False,
    ignore_hydrogen_counts=True,
    timeout_ms=10_000,
    ignore_hydrogen_positions=True,
)


def _cid(cid: str) -> str:
    return str(cid).strip().upper()


def _compare_key(cid: str, exact_ids: bool) -> str:
    cid = _cid(cid)
    return cid if exact_ids else cid[:6]


def _null(pair: dict) -> dict:
    return {"row_idx": pair["row_idx"], "nmatched": 0, "rmsd": None}


def _compare_pair(pair: dict) -> dict:
    engine = PackingSimilarity()
    for key, value in SETTINGS.items():
        setattr(engine.settings, key, value)
    gt = CCDCCrystal.from_string(pair["gt_cif"])
    try:
        compack_result = engine.compare(CCDCCrystal.from_string(pair["pred_cif"]), gt)
    except Exception:
        return _null(pair)
    if compack_result is None:
        return _null(pair)
    best = max(
        compack_result if isinstance(compack_result, (list, tuple)) else [compack_result],
        key=lambda r: (r.nmatched_molecules, -r.rmsd),
    )
    return {"row_idx": pair["row_idx"], "nmatched": best.nmatched_molecules, "rmsd": best.rmsd}


def _start(pair: dict, timeout_seconds: float) -> tuple[int, int, dict, float]:
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        with os.fdopen(w_fd, "wb") as f:
            try:
                pickle.dump(_compare_pair(pair), f)
            except Exception:
                pickle.dump(_null(pair), f)
        os._exit(0)
    os.close(w_fd)
    return pid, r_fd, pair, time.monotonic() + timeout_seconds


def _load_gt(config_path: Path, exact_ids: bool) -> dict[str, list[str]]:
    config = json.loads(config_path.read_text())
    subset = str(config.get("subset", "oxtal")).strip().lower()
    family = config.get("family")

    if not GT_CIFS_PATH.is_file():
        raise FileNotFoundError(f"Missing GT CIF cache at {GT_CIFS_PATH}")
    df = pl.read_parquet(GT_CIFS_PATH).filter(pl.col("subsets").list.contains(subset))
    if family:
        df = df.filter(pl.col("family") == _compare_key(family, exact_ids=False))

    gt: dict[str, list[str]] = {}
    for row in df.iter_rows(named=True):
        gt.setdefault(_compare_key(row["csd_id"], exact_ids), []).append(row["cif"])
    return gt


def _run_rows(
    rows: list[dict],
    gt_by_key: dict[str, list[str]],
    num_processes: int,
    timeout_seconds: float,
    exact_ids: bool,
    shard_dir: Path | None,
):
    results = [
        dict(
            sample_idx=r["sample_idx"], id=_cid(r["id"]), nmatched=0, rmsd=None, exact_ids=exact_ids
        )
        for r in rows
    ]
    best_keys: list[tuple[int, float] | None] = [None] * len(rows)
    rows_by_id: dict[str, list[int]] = {}
    remaining_by_id: dict[str, int] = {}
    pairs: list[dict] = []
    written_shards: set[str] = set()
    for row_idx, row in enumerate(rows):
        cid = _cid(row["id"])
        gt_cifs = gt_by_key.get(_compare_key(cid, exact_ids), [])
        rows_by_id.setdefault(cid, []).append(row_idx)
        remaining_by_id[cid] = remaining_by_id.get(cid, 0) + len(gt_cifs)
        pairs.extend(
            {
                "row_idx": row_idx,
                "sample_idx": row["sample_idx"],
                "id": cid,
                "pred_cif": row["cif"],
                "gt_cif": gt,
            }
            for gt in gt_cifs
        )

    missing = sorted(cid for cid, count in remaining_by_id.items() if count == 0)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = f", ... ({len(missing)} total)" if len(missing) > 10 else ""
        print(f"WARNING: no GT CIFs found for comparison ids: {preview}{suffix}", flush=True)

    def write_shard(cid: str) -> None:
        if shard_dir is None or cid in written_shards:
            return
        shard_dir.mkdir(exist_ok=True)
        shard_rows = [results[row_idx] for row_idx in rows_by_id[cid]]
        pl.DataFrame(shard_rows).sort("sample_idx").write_csv(shard_dir / f"{cid}.csv")
        written_shards.add(cid)

    def update(result: dict) -> None:
        row_idx = result["row_idx"]
        cid = results[row_idx]["id"]
        if result["rmsd"] is not None:
            key = (result["nmatched"], -result["rmsd"])
            if best_keys[row_idx] is None or key > best_keys[row_idx]:
                best_keys[row_idx] = key
                results[row_idx]["nmatched"] = result["nmatched"]
                results[row_idx]["rmsd"] = result["rmsd"]
        remaining_by_id[cid] -= 1
        if remaining_by_id[cid] == 0:
            write_shard(cid)

    pair_iter = iter(pairs)
    active: dict[int, tuple[int, int, dict, float]] = {}

    def fill() -> None:
        while len(active) < num_processes:
            try:
                job = _start(next(pair_iter), timeout_seconds)
            except StopIteration:
                return
            active[job[1]] = job

    with tqdm(total=len(pairs), desc="COMPACK pairs") as progress:
        fill()
        while active:
            wait = max(0.0, min(deadline for *_, deadline in active.values()) - time.monotonic())
            ready, _, _ = select.select(list(active), [], [], wait)
            for fd in ready:
                pid, _, pair, _ = active.pop(fd)
                try:
                    with os.fdopen(fd, "rb") as f:
                        update(pickle.load(f))
                except EOFError:
                    update(_null(pair))
                os.waitpid(pid, 0)
                progress.update(1)

            now = time.monotonic()
            for fd, (pid, _, pair, deadline) in list(active.items()):
                if deadline <= now:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                    os.close(fd)
                    active.pop(fd)
                    update({"row_idx": pair["row_idx"], "nmatched": 0, "rmsd": None})
                    progress.update(1)
            fill()

    return results


def _load_completed_shards(
    shard_dir: Path, rows_by_id: dict[str, list[dict]], exact_ids: bool
) -> tuple[dict[int, dict], set[str]]:
    if not shard_dir.is_dir():
        return {}, set()

    results: dict[int, dict] = {}
    completed_ids: set[str] = set()
    for path in shard_dir.glob("*.csv"):
        cid = _cid(path.stem)
        if cid not in rows_by_id:
            continue
        try:
            shard = pl.read_csv(path)
        except Exception:
            continue
        if not {"sample_idx", "nmatched", "rmsd", "exact_ids"}.issubset(shard.columns):
            continue
        if set(shard.get_column("exact_ids").to_list()) != {exact_ids}:
            continue
        needed = {row["sample_idx"] for row in rows_by_id[cid]}
        shard = shard.filter(pl.col("sample_idx").is_in(needed))
        if set(shard.get_column("sample_idx").to_list()) != needed:
            continue
        completed_ids.add(cid)
        for result in shard.to_dicts():
            result["id"] = cid
            results[result["sample_idx"]] = result
    return results, completed_ids


def compack(
    experiment_dir: str | Path,
    num_processes: int = 192,
    overwrite: bool = False,
    timeout_seconds: float = 100.0,
    exact_ids: bool = False,
    no_shards: bool = False,
):
    experiment_dir = Path(experiment_dir)
    if (
        not experiment_dir.is_absolute()
        and not experiment_dir.exists()
        and len(experiment_dir.parts) == 1
    ):
        experiment_dir = RESULTS_DIR / experiment_dir
    if experiment_dir.name == "predictions.parquet":
        experiment_dir = experiment_dir.parent

    csv_path = experiment_dir / "compack.csv"
    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {csv_path}. Pass --overwrite to replace it.")

    df = pl.read_parquet(experiment_dir / "predictions.parquet").select("sample_idx", "id", "cif")
    rows = df.to_dicts()
    gt_by_key = _load_gt(experiment_dir / "config.json", exact_ids)
    shard_dir = None if no_shards else experiment_dir / "compack"

    rows_by_id: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_id.setdefault(_cid(row["id"]), []).append(row)

    shard_results, completed_ids = (
        ({}, set())
        if overwrite or shard_dir is None
        else _load_completed_shards(shard_dir, rows_by_id, exact_ids)
    )
    pending_rows = [row for row in rows if _cid(row["id"]) not in completed_ids]
    skipped = len(rows) - len(pending_rows)
    if skipped:
        print(
            f"Resuming: skipping {skipped} rows from {len(completed_ids)} completed id shards, "
            f"{len(pending_rows)} rows remaining"
        )

    print(
        f"Processing {len(pending_rows)} predictions, "
        f"{sum(len(gt_by_key.get(_compare_key(row['id'], exact_ids), [])) for row in pending_rows)} "
        f"CSD test GT pairs "
        f"with {num_processes} workers"
    )

    new_results = _run_rows(
        pending_rows, gt_by_key, num_processes, timeout_seconds, exact_ids, shard_dir
    )

    all_results = list(shard_results.values()) + new_results
    pl.DataFrame(all_results).select("sample_idx", "nmatched", "rmsd").sort("sample_idx").write_csv(
        csv_path
    )

    options = {
        "exact_ids": exact_ids,
        "compare_key": "id" if exact_ids else "family",
        "shard_dir": None if no_shards else "compack",
        "shard_key": "id",
        "num_processes": num_processes,
        "timeout_seconds": timeout_seconds,
    }
    config_path = experiment_dir / "config.json"
    config = json.loads(config_path.read_text())
    config.setdefault("compack", {})["compack.csv"] = {
        "compack": dict(SETTINGS),
        "options": options,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved {len(all_results)} results to {csv_path}")
    return pl.DataFrame(all_results)


def main():
    parser = ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--num_processes", type=int, default=192)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout_seconds", type=float, default=100.0)
    parser.add_argument("--exact_ids", action="store_true")
    parser.add_argument("--no_shards", action="store_true")
    args = parser.parse_args()
    compack(**vars(args))


if __name__ == "__main__":
    main()
