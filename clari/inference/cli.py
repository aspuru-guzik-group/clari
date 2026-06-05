from __future__ import annotations

import argparse
import datetime
import hashlib
import re
import sys
from pathlib import Path

import polars as pl
import torch
from rdkit import Chem

from clari.chem import Crystal, silenced_rdlogger
from clari.inference.sampler import (
    DEFAULT_MAX_RESAMPLE_FACTOR,
    ClariSampler,
    SampleRequest,
    resolve_device,
    resolve_hub_checkpoint,
)


def sanitize_filename(name: str) -> str:
    """Sanitize the SMILES string to create a valid, clean folder and filename.
    
    This replaces any characters that are not alphanumeric, underscore, or dash
    with an underscore, collapses consecutive underscores, strips leading/trailing
    separators, and limits the length of the string to avoid filesystem/shell limits,
    appending an MD5 hash if truncated.
    """
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    clean = re.sub(r"_+", "_", clean)
    clean = clean.strip("_-")
    
    max_len = 64
    if len(clean) > max_len:
        h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        prefix = clean[:50].rstrip("-_")
        clean = f"{prefix}_{h}"
        
    return clean


def generate_id(smiles_input: list[tuple[str, int]]) -> str:
    """Generate a clean, sanitized ID from the list of (SMILES, copies)."""
    parts = []
    for s, c in smiles_input:
        s_clean = sanitize_filename(s)
        if c > 1:
            parts.append(f"{s_clean}_x{c}")
        else:
            parts.append(s_clean)
    res = "_".join(parts)
    
    # Ensure the combined ID is not excessively long
    max_id_len = 100
    if len(res) > max_id_len:
        h = hashlib.md5(res.encode("utf-8")).hexdigest()[:8]
        prefix = res[:80].rstrip("-_")
        res = f"{prefix}_{h}"
        
    return res


def normalize_inputs(
    pos_args: list[str] | None,
    smiles_flag: list[str] | None,
    copies_flag: list[int] | None,
    default_copies: int = 4
) -> list[tuple[str, int]]:
    """Normalize CLI inputs into a list of (SMILES, copies) tuples.
    This supports both positional arguments (alternating SMILES and copies)
    and flag arguments (with optional copies).
    If any SMILES string contains dots ('.'), it is split, and each part
    retains the corresponding copy count.
    """
    raw_pairs = []

    # 1. Parse positional arguments first if present
    if pos_args:
        i = 0
        while i < len(pos_args):
            s = pos_args[i]
            c = default_copies
            if i + 1 < len(pos_args):
                next_arg = pos_args[i + 1]
                if next_arg.isdigit():
                    c = int(next_arg)
                    i += 2
                    raw_pairs.append((s, c))
                    continue
            raw_pairs.append((s, c))
            i += 1

    # 2. Parse flag arguments if present
    if smiles_flag:
        for idx, s in enumerate(smiles_flag):
            c = default_copies
            if copies_flag and idx < len(copies_flag):
                c = copies_flag[idx]
            raw_pairs.append((s, c))

    if not raw_pairs:
        raise ValueError("No SMILES input provided. Please provide SMILES via positional arguments or --smiles.")

    return raw_pairs


def to_wrapped_cif(crystal: Crystal) -> str:
    from collections import defaultdict

    import gemmi
    import numpy as np
    import torch_geometric as pyg

    from clari.chem.common import PTABLE
    from clari.chem.crystal import softwrap

    lattice = crystal.lattice.detach().cpu().numpy()
    a, b, c = map(np.linalg.norm, lattice)
    alpha = np.degrees(np.arccos(np.dot(lattice[1], lattice[2]) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(lattice[0], lattice[2]) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(lattice[0], lattice[1]) / (a * b)))

    st = gemmi.SmallStructure()
    st.cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)
    st.spacegroup_hm = "P 1"

    # Wrap the centers of mass of each body into [0, 1] without zero-centering
    f = crystal.frac_coords
    fcoms = pyg.utils.scatter(f, crystal.body_ids, reduce="mean")
    shift = softwrap(fcoms, bounds=(0.0, 1.0), margin=1e-5) - fcoms
    frac_coords = (f + shift[crystal.body_ids]).detach().cpu().numpy()

    atom_nums = crystal.atom_nums.detach().cpu().numpy()
    label_counts = defaultdict(int)
    for z, (x, y, z_c) in zip(atom_nums, frac_coords, strict=False):
        symbol = PTABLE.GetElementSymbol(int(z))
        label_counts[symbol] += 1
        site = gemmi.SmallStructure.Site()
        site.label = f"{symbol}{label_counts[symbol]}"
        site.fract = gemmi.Fractional(float(x), float(y), float(z_c))
        site.element = gemmi.Element(int(z))
        site.occ = 1.0
        st.add_site(site)

    block = st.make_cif_block()
    block.name = crystal.csd_id
    return block.as_string()


def wrap_predictions_parquet(
    predictions_path: Path,
    template_body_ids: torch.Tensor,
    request_id: str,
    output_path: Path,
) -> None:
    from clari.chem import Crystal

    if not predictions_path.exists():
        return

    df = pl.read_parquet(predictions_path)
    wrapped_cifs = []
    
    for row in df.iter_rows(named=True):
        sample_idx = row["sample_idx"]
        cif_content = row["cif"]
        
        # Load crystal from CIF string
        crystal_obj = Crystal.from_cif(cif_content)
        # Set the correct body_ids
        crystal_obj = crystal_obj.replace(body_ids=template_body_ids)
        
        # Wrap the centers of mass and generate wrapped CIF
        wrapped_cif = to_wrapped_cif(crystal_obj)
        wrapped_cifs.append(wrapped_cif)
        
        # Save individual CIF file
        cif_file = output_path / f"{request_id}_sample{sample_idx}.cif"
        cif_file.write_text(wrapped_cif)
        
    # Update parquet file
    df = df.with_columns(pl.Series("cif", wrapped_cifs))
    df.write_parquet(predictions_path)


def sample(
    smiles: str | list[str] | list[tuple[str, int]],
    n_samples: int = 1,
    copies: int | list[int] | None = None,
    checkpoint_path: str = "clari-h",
    output_dir: str | Path | None = None,
    ids: str | list[str] | None = None,
    batch_size: int | None = None,
    num_gpus: int = 1,
    device: str | None = "auto",
    n_steps: int | None = 50,
    use_ema: bool = True,
    use_bf16: bool = True,
    compile: bool | None = None,
    torch_threads: int = 1,
    overwrite: bool = False,
    keep_shards: bool = False,
    filter_clashing: bool = True,
    max_resample_factor: int = DEFAULT_MAX_RESAMPLE_FACTOR,
    pbar: bool = True,
    step_pbar: bool = True,
    add_hs: bool = True,
    return_trajectory: bool = False,
) -> list | Path:
    """Sample crystal structures from unit cell SMILES strings.

    If batch_size is omitted, CLARI uses an automatic atom-count heuristic scaled from
    batch sizes that were fit on an 81 GB H100 GPU. Pass batch_size explicitly when you
    want fixed behavior across GPUs.

    By default the sampler filters out crystals whose atoms inter-penetrate (per
    clari.pipelines.utils.metrics.check_clashes_eval) and resamples the deficit until
    n_samples non-clashing structures are reached. Pass filter_clashing=False to keep
    all samples regardless of clashes. The loop gives up after
    max_resample_factor * n_samples total attempts to avoid spinning forever on
    pathological molecules.
    """
    if checkpoint_path is None:
        checkpoint_path = "clari-h"

    checkpoint_lower = str(checkpoint_path).strip().lower()
    if checkpoint_lower in ("clari-m", "clari-l", "clari-huge", "clari-h", "clari-med", "clari-large"):
        resolved_checkpoint = resolve_hub_checkpoint(checkpoint_lower)
    else:
        resolved_checkpoint = checkpoint_path

    # Normalize the smiles and copies inputs to list[tuple[str, int]]
    # Handle programmatic usage where smiles might already be normalized
    if isinstance(smiles, list) and all(isinstance(item, tuple) and len(item) == 2 for item in smiles):
        smiles_input = smiles
    else:
        # If it's single SMILES or list of SMILES, normalize using normalize_inputs helper
        smiles_list = [smiles] if isinstance(smiles, str) else list(smiles)
        copies_list = [copies] if isinstance(copies, int) else (list(copies) if isinstance(copies, list) else None)
        smiles_input = normalize_inputs(None, smiles_list, copies_list)

    sanitized_id = generate_id(smiles_input)

    # Determine ID
    if ids is not None:
        if isinstance(ids, list):
            request_id = ids[0] if ids else sanitized_id
        else:
            request_id = ids
    else:
        request_id = sanitized_id

    # Resolve output directory
    if output_dir is None:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"{sanitized_id}_{now}")
    else:
        output_path = Path(output_dir)

    # Construct the Crystal object
    try:
        with silenced_rdlogger():
            mols = []
            for s, c in smiles_input:
                mol = Chem.MolFromSmiles(s, sanitize=False)
                if mol is None:
                    raise ValueError(f"Could not parse SMILES: {s!r}")
                frags = Chem.GetMolFrags(mol, asMols=True)
                if frags:
                    for frag in frags:
                        mols.append((frag, c))
                else:
                    mols.append((mol, c))

        if add_hs:
            for m, _ in mols:
                m.UpdatePropertyCache(strict=False)
            mols = [(Chem.AddHs(m), c) for m, c in mols]

        crystal = Crystal.from_rdmol(
            mols,
            csd_id=request_id,
        )
    except Exception as e:
        print(f"Error constructing Crystal from SMILES: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize the sampler
    resolved_device = resolve_device(device)
    print(f"Loading model from {resolved_checkpoint} on device: {resolved_device}")
    
    sampler = ClariSampler.from_checkpoint(
        resolved_checkpoint,
        use_ema=use_ema,
        device=resolved_device,
        n_steps=n_steps,
        use_bf16=(resolved_device.type == "cuda" if use_bf16 else False),
        compile=compile,
        torch_threads=torch_threads,
        num_gpus=num_gpus,
    )

    request = SampleRequest(
        id=request_id,
        crystal=crystal,
        smiles=smiles_input,
        copies=1,
        n_samples=n_samples,
    )

    print(f"Sampling {n_samples} structure(s) for: {smiles_input}")
    
    if num_gpus == 1:
        # Run in-memory
        samples_result = sampler.sample(
            [request],
            batch_size=batch_size,
            return_trajectory=return_trajectory,
            filter_clashing=filter_clashing,
            max_resample_factor=max_resample_factor,
            pbar=pbar,
            step_pbar=step_pbar,
        )
        
        # Save output using write_output
        from clari.inference.io import prepare_output_dir
        prepare_output_dir(output_path, overwrite=overwrite)
        sampler.write_output(samples_result, output_dir=output_path, requests=[request], overwrite=overwrite)
        predictions_path = output_path / "predictions.parquet"
        print(f"Saved predictions to {predictions_path}")
        
        # Wrap COM in predictions.parquet and export wrapped CIFs
        wrap_predictions_parquet(predictions_path, crystal.body_ids, request_id, output_path)
        print(f"Success! Saved sample crystal structure(s) to: {output_path.resolve()}")
        return samples_result
    else:
        # Run on-disk (sharded multi-GPU)
        predictions_path = sampler.sample(
            [request],
            output_dir=output_path,
            batch_size=batch_size,
            num_gpus=num_gpus,
            overwrite=overwrite,
            keep_shards=keep_shards,
            filter_clashing=filter_clashing,
            max_resample_factor=max_resample_factor,
            pbar=pbar,
            step_pbar=step_pbar,
        )
        print(f"Saved predictions to {predictions_path}")
        
        # Wrap COM in predictions.parquet and export wrapped CIFs
        if predictions_path.exists():
            try:
                wrap_predictions_parquet(predictions_path, crystal.body_ids, request_id, output_path)
                print(f"Success! Saved sample crystal structure(s) to: {output_path.resolve()}")
            except Exception as e:
                print(f"Warning: Could not export and wrap individual CIF files: {e}", file=sys.stderr)
        return predictions_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample organic crystal structures from unit cell SMILES strings."
    )
    # Positional args (mixed SMILES and optional copies)
    parser.add_argument("pos_args", nargs="*", help="SMILES and optional copies (e.g. SMILES copies SMILES copies...)")
    
    # Flag arguments
    parser.add_argument("--config", type=str, help="Path to a JSON config file")
    parser.add_argument("--smiles", "-s", type=str, action="append", help="SMILES string(s)")
    parser.add_argument("--copies", action="append", type=int, help="Copies corresponding to each SMILES")
    parser.add_argument("--n-samples", "--samples", "-n", dest="n_samples", type=int, default=1, help="Number of samples (default: 1)")
    parser.add_argument("--checkpoint-path", "--checkpoint", "-c", type=str, default="clari-h", help="Local checkpoint path or Hugging Face model key ('clari-m', 'clari-l', 'clari-h')")
    parser.add_argument("--output-dir", "--out", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--ids", type=str, action="append", help="Request ID(s)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--device", "-d", type=str, default="auto", help="Device (auto, mps, cuda, cpu)")
    parser.add_argument("--n-steps", "--steps", dest="n_steps", type=int, default=50, help="Denoising steps (default: 50)")
    
    # Boolean configurations
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false", help="Disable EMA weights")
    parser.add_argument("--no-use-bf16", dest="use_bf16", action="store_false", help="Disable bf16 autocast")
    parser.add_argument("--compile", action="store_true", default=None, help="Compile the model")
    parser.add_argument("--torch-threads", type=int, default=1, help="Number of PyTorch threads")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    parser.add_argument("--keep-shards", action="store_true", help="Keep intermediate shard files")
    parser.add_argument("--no-filter-clashing", dest="filter_clashing", action="store_false", help="Disable clash filtering")
    parser.add_argument("--max-resample-factor", type=int, default=DEFAULT_MAX_RESAMPLE_FACTOR, help="Max resampling factor")
    parser.add_argument("--no-pbar", dest="pbar", action="store_false", help="Disable overall progress bar")
    parser.add_argument("--no-step-pbar", dest="step_pbar", action="store_false", help="Disable denoising step progress bar")
    parser.add_argument("--no-addHs", "--no-addhs", dest="add_hs", action="store_false", help="Do not add hydrogens to SMILES input")

    # Handle --config pre-parsing
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        import json
        try:
            with open(temp_args.config, 'r') as f:
                defaults = json.load(f)
            # Map samples/steps to n_samples/n_steps
            if "samples" in defaults and "n_samples" not in defaults:
                defaults["n_samples"] = defaults.pop("samples")
            if "steps" in defaults and "n_steps" not in defaults:
                defaults["n_steps"] = defaults.pop("steps")
            parser.set_defaults(**defaults)
        except Exception as e:
            print(f"Error loading config file {temp_args.config}: {e}", file=sys.stderr)
            sys.exit(1)

    args = parser.parse_args()

    # Parse and normalize SMILES and copies
    try:
        smiles_input = normalize_inputs(args.pos_args, args.smiles, args.copies)
    except Exception as e:
        print(f"Error parsing inputs: {e}", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    try:
        sample(
            smiles=smiles_input,
            n_samples=args.n_samples,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            ids=args.ids,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus,
            device=args.device,
            n_steps=args.n_steps,
            use_ema=args.use_ema,
            use_bf16=args.use_bf16,
            compile=args.compile,
            torch_threads=args.torch_threads,
            overwrite=args.overwrite,
            keep_shards=args.keep_shards,
            filter_clashing=args.filter_clashing,
            max_resample_factor=args.max_resample_factor,
            pbar=args.pbar,
            step_pbar=args.step_pbar,
            add_hs=args.add_hs,
        )
    except Exception as e:
        print(f"Error during sampling: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
