from __future__ import annotations

import argparse
import multiprocessing as mp
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import polars as pl
import torch
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from tqdm import tqdm

from clari.chem import Crystal
from clari.paths import resolve_results_path

MODEL_NAME = "uma-s-1p2"
TASK_NAME = "omc"
ID_COLUMN = "id"
CIF_COLUMN = "cif"
SAMPLE_IDX_COLUMN = "sample_idx"
ENERGIES_CSV_COLUMN = "energies"
ENERGIES_CSV_NAME = "energies.csv"
ENERGY_KEYS = ("energy", "energy_pred", "y", "potential_energy")
FAILED_ENERGY = float("inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FAIRChem UMA energies for CIFs stored in parquet input."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Experiment directory or predictions parquet to evaluate.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timing", action="store_true")
    return parser.parse_args()


def set_torch_threads(torch_threads: int) -> None:
    if torch_threads <= 0:
        raise ValueError(f"torch_threads must be positive, got {torch_threads}")
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(torch_threads)
    except RuntimeError:
        pass


def load_atoms(cif_strings: list[str], *, desc: str, position: int = 0) -> list:
    atoms_list = []
    for cif in tqdm(
        cif_strings,
        desc=desc,
        position=position,
        leave=False,
        dynamic_ncols=True,
    ):
        try:
            atoms_list.append(Crystal.ase_from_cif(cif))
        except Exception:
            atoms_list.append(None)
    return atoms_list


def extract_energies(predictions: dict) -> list[float]:
    for key in ENERGY_KEYS:
        if key in predictions:
            values = predictions[key]
            return [float(x) for x in values.reshape(-1).detach().cpu().tolist()]
    raise KeyError(
        f"Could not find an energy key in predictor output. Available keys: {sorted(predictions)}"
    )


def compute_batch_energies(atoms_batch: list, predictor, *, timing: bool = False):
    atomic_data_list = [AtomicData.from_ase(atoms, task_name=TASK_NAME) for atoms in atoms_batch]
    batch = atomicdata_list_to_batch(atomic_data_list).to(predictor.device)
    if timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            ):
                start.record()
                predictions = predictor.predict(batch)
                end.record()
        end.synchronize()
        gpu_ms = start.elapsed_time(end)
    else:
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_available(),
            ):
                predictions = predictor.predict(batch)
    energies = extract_energies(predictions)
    if timing:
        return energies, gpu_ms
    return energies


def _is_oom_error(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def _is_fairchem_graph_error(exc: RuntimeError) -> bool:
    return not _is_oom_error(exc)


def build_groups(ids: list[str]) -> tuple[list[str], dict[str, list[int]]]:
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for idx, crystal_id in enumerate(ids):
        if crystal_id not in groups:
            groups[crystal_id] = []
            order.append(crystal_id)
        groups[crystal_id].append(idx)
    return order, groups


def chunk_size_for_atoms(atoms) -> int:
    num_atoms = len(atoms)
    if num_atoms > 1000:
        return 1
    if num_atoms < 200:
        return 1000
    if num_atoms < 300:
        return 500
    if num_atoms < 500:
        return 200
    return 25


def warmup_predictor(
    predictor, atoms_list: list, ordered_ids, grouped_indices, batch_size: int
) -> None:
    if not ordered_ids:
        return
    first_indices = grouped_indices[ordered_ids[0]]
    chunk_size = chunk_size_for_atoms(atoms_list[first_indices[0]])
    n = min(batch_size, chunk_size, len(first_indices))
    atoms_batch = [atoms_list[idx] for idx in first_indices[:n]]
    for _ in range(2):
        compute_batch_energies(atoms_batch, predictor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def compute_preloaded_energies(
    ids: list[str],
    atoms_list: list,
    *,
    row_indices: list[int] | None = None,
    batch_size: int,
    device: str,
    desc_prefix: str,
    position: int = 0,
    timing: bool = False,
    rank: int = 0,
) -> tuple[list[float], list[dict]]:
    predictor = pretrained_mlip.get_predict_unit(
        MODEL_NAME,
        device=device,
        workers=1,
    )
    energies: list[float | None] = [None] * len(ids)
    timing_rows: list[dict] = []
    source_indices = row_indices or list(range(len(ids)))
    ordered_ids, grouped_indices = build_groups(ids)

    if timing:
        warmup_predictor(predictor, atoms_list, ordered_ids, grouped_indices, batch_size)

    progress = tqdm(
        total=len(ids),
        desc=f"{desc_prefix} samples",
        position=position,
        leave=False,
        dynamic_ncols=True,
        unit="sample",
    )
    for crystal_id in ordered_ids:
        indices = grouped_indices[crystal_id]
        valid_indices = []
        for idx in indices:
            if atoms_list[idx] is None:
                energies[idx] = FAILED_ENERGY
                tqdm.write(
                    f"{desc_prefix}: degenerate CIF for id={crystal_id}, "
                    f"row={source_indices[idx]}; setting energy=inf"
                )
                progress.update(1)
            else:
                valid_indices.append(idx)
        if not valid_indices:
            continue
        indices = valid_indices
        chunk_size = chunk_size_for_atoms(atoms_list[indices[0]])
        for chunk_start in range(0, len(indices), chunk_size):
            chunk_indices = indices[chunk_start : chunk_start + chunk_size]
            batch_size_curr = min(batch_size, len(chunk_indices))
            batch_start = 0
            while batch_start < len(chunk_indices):
                batch_indices = chunk_indices[batch_start : batch_start + batch_size_curr]
                atoms_batch = [atoms_list[idx] for idx in batch_indices]
                try:
                    if timing:
                        batch_energies, gpu_ms = compute_batch_energies(
                            atoms_batch,
                            predictor,
                            timing=True,
                        )
                    else:
                        batch_energies = compute_batch_energies(atoms_batch, predictor)
                        gpu_ms = None
                except RuntimeError as e:
                    is_oom = torch.cuda.is_available() and _is_oom_error(e)
                    is_graph_error = _is_fairchem_graph_error(e)
                    if is_oom or is_graph_error:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if batch_size_curr > 1:
                            batch_size_curr = max(1, batch_size_curr // 2)
                            continue
                        row_idx = batch_indices[0]
                        energies[row_idx] = FAILED_ENERGY
                        reason = "CUDA OOM" if is_oom else "FairChem graph error"
                        tqdm.write(
                            f"{desc_prefix}: {reason} at batch_size=1 for id={crystal_id}, "
                            f"row={source_indices[row_idx]}; setting energy=inf"
                        )
                        progress.update(1)
                        batch_start += 1
                        del atoms_batch
                        continue
                    raise
                for row_idx, energy in zip(batch_indices, batch_energies, strict=True):
                    energies[row_idx] = energy
                if timing:
                    timing_rows.append(
                        {
                            "rank": rank,
                            "id": crystal_id,
                            "chunk_start": chunk_start,
                            "batch_start": batch_start,
                            "batch_size": len(batch_indices),
                            "requested_batch_size": batch_size,
                            "active_batch_size": batch_size_curr,
                            "gpu_ms": gpu_ms,
                            "row_indices": ",".join(
                                str(source_indices[idx]) for idx in batch_indices
                            ),
                        }
                    )
                progress.update(len(batch_indices))
                batch_start += len(batch_indices)
                del atoms_batch, batch_energies
    progress.close()

    if any(energy is None for energy in energies):
        raise RuntimeError("Missing energies after evaluation")
    return [float(energy) for energy in energies], timing_rows


def run_rows(
    rows: list[tuple[int, str, str]],
    *,
    batch_size: int,
    device: str,
    desc: str,
    position: int = 0,
    timing: bool = False,
    rank: int = 0,
):
    row_indices = [row_idx for row_idx, _, _ in rows]
    ids = [crystal_id for _, crystal_id, _ in rows]
    cifs = [cif for _, _, cif in rows]
    atoms_list = load_atoms(cifs, desc=f"{desc} load", position=position)
    energies, timing_rows = compute_preloaded_energies(
        ids,
        atoms_list,
        row_indices=row_indices,
        batch_size=batch_size,
        device=device,
        desc_prefix=desc,
        position=position,
        timing=timing,
        rank=rank,
    )
    return row_indices, energies, timing_rows


def assign_ids_to_shards(ordered_ids: list[str], num_shards: int) -> list[set[str]]:
    shards = [set() for _ in range(num_shards)]
    for idx, crystal_id in enumerate(ordered_ids):
        shards[idx % num_shards].add(crystal_id)
    return shards


def gpu_worker(
    rank: int,
    rows: list[tuple[int, str, str]],
    batch_size: int,
    return_dict,
    torch_threads: int,
    timing: bool,
    tqdm_lock,
) -> None:
    tqdm.set_lock(tqdm_lock)
    set_torch_threads(torch_threads)
    torch.cuda.set_device(rank)
    row_indices, energies, timing_rows = run_rows(
        rows,
        batch_size=batch_size,
        device="cuda",
        desc=f"GPU {rank}",
        position=rank,
        timing=timing,
        rank=rank,
    )
    return_dict[rank] = (row_indices, energies, timing_rows)


def compute_multi_gpu(
    rows: list[tuple[int, str, str]],
    *,
    batch_size: int,
    num_gpus: int,
    torch_threads: int,
    timing: bool = False,
) -> tuple[list[float], list[dict]]:
    ordered_ids, _ = build_groups([crystal_id for _, crystal_id, _ in rows])
    id_shards = assign_ids_to_shards(ordered_ids, num_gpus)
    row_shards: list[list[tuple[int, str, str]]] = []
    for shard_ids in id_shards:
        row_shards.append([row for row in rows if row[1] in shard_ids])

    ctx = mp.get_context("spawn")
    tqdm_lock = ctx.RLock()
    manager = ctx.Manager()
    return_dict = manager.dict()
    processes = []

    for rank, shard_rows in enumerate(row_shards):
        process = ctx.Process(
            target=gpu_worker,
            args=(rank, shard_rows, batch_size, return_dict, torch_threads, timing, tqdm_lock),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"GPU worker failed with exit code {process.exitcode}")

    energies: list[float | None] = [None] * len(rows)
    timing_rows: list[dict] = []
    for rank in range(len(row_shards)):
        row_indices, shard_energies, shard_timing_rows = return_dict[rank]
        for row_idx, energy in zip(row_indices, shard_energies, strict=True):
            energies[row_idx] = energy
        timing_rows.extend(shard_timing_rows)

    if any(energy is None for energy in energies):
        raise RuntimeError("Missing energies after multi GPU evaluation")
    return [float(energy) for energy in energies], timing_rows


def load_input_df(input_path: Path) -> pl.DataFrame:
    if input_path.is_dir():
        input_path = input_path / "predictions.parquet"
    if not input_path.is_file():
        raise FileNotFoundError(f"Input parquet does not exist: {input_path}")
    if input_path.suffix != ".parquet":
        raise ValueError(f"Input must be a parquet file: {input_path}")

    df = pl.read_parquet(input_path)
    required_columns = {SAMPLE_IDX_COLUMN, ID_COLUMN, CIF_COLUMN}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {input_path}: {missing_columns}")
    return df.select(SAMPLE_IDX_COLUMN, ID_COLUMN, CIF_COLUMN)


def resolve_input_path(input_path: Path) -> Path:
    input_path = resolve_results_path(input_path)
    if input_path.is_dir():
        return input_path / "predictions.parquet"
    return input_path


def compute_energies_df(
    df: pl.DataFrame,
    *,
    batch_size: int = 32,
    num_gpus: int = 1,
    torch_threads: int = 1,
    timing: bool = False,
) -> tuple[list[float], list]:
    """Compute UMA energies for an in-memory predictions DataFrame with id/cif columns."""
    ids = df.get_column(ID_COLUMN).to_list()
    cifs = df.get_column(CIF_COLUMN).to_list()
    rows = list(zip(range(len(df)), ids, cifs, strict=True))

    available_gpus = torch.cuda.device_count()
    if num_gpus > 1 and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {num_gpus} GPUs for energy eval, but CUDA is not available")
    use_multi_gpu = torch.cuda.is_available() and num_gpus > 1
    if use_multi_gpu:
        if available_gpus < num_gpus:
            raise ValueError(f"Requested {num_gpus} GPUs, but only found {available_gpus}")
        energies, timing_rows = compute_multi_gpu(
            rows,
            batch_size=batch_size,
            num_gpus=num_gpus,
            torch_threads=torch_threads,
            timing=timing,
        )
    else:
        set_torch_threads(torch_threads)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, energies, timing_rows = run_rows(
            rows,
            batch_size=batch_size,
            device=device,
            desc="Main",
            timing=timing,
            rank=0,
        )

    if len(energies) != len(df):
        raise RuntimeError(f"Expected {len(df)} energies, got {len(energies)}")
    return energies, timing_rows


def main(
    input_path: Path,
    batch_size: int,
    num_gpus: int,
    torch_threads: int,
    overwrite: bool = False,
    timing: bool = False,
) -> None:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    if torch_threads <= 0:
        raise ValueError(f"torch_threads must be positive, got {torch_threads}")

    input_path = resolve_input_path(input_path)
    df = load_input_df(input_path)
    if len(df) == 0:
        raise ValueError(f"No rows loaded from input: {input_path}")
    energies_csv_path = input_path.with_name(ENERGIES_CSV_NAME)
    if energies_csv_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {energies_csv_path}. Pass --overwrite.")

    energies, timing_rows = compute_energies_df(
        df, batch_size=batch_size, num_gpus=num_gpus, torch_threads=torch_threads, timing=timing
    )

    pl.DataFrame(
        {
            SAMPLE_IDX_COLUMN: df.get_column(SAMPLE_IDX_COLUMN),
            ENERGIES_CSV_COLUMN: energies,
        }
    ).write_csv(energies_csv_path)
    print(f"Saved {energies_csv_path}")
    if timing:
        timing_path = input_path.with_name("energy_timing.csv")
        pl.DataFrame(timing_rows).write_csv(timing_path)
        print(f"Saved energy timings to {timing_path}")


def cli() -> None:
    args = parse_args()
    main(
        input_path=args.input,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        torch_threads=args.torch_threads,
        overwrite=args.overwrite,
        timing=args.timing,
    )


if __name__ == "__main__":
    cli()
