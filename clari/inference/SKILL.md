---
name: clari
description: Run CLARI crystal structure inference, batch sampling, ranking, and CIF export. Use for requests to sample crystal candidates from SMILES, run `sample`, `rank`, or `export-cifs`, prepare current-format inference configs, or explain the simplified inference package in `clari/inference/`.
---

# CLARI inference

CLARI predicts organic crystal structures from molecular SMILES. The workflow has three steps:

1. `clari` — samples candidate crystal structures, writes `predictions.parquet`
2. `rank` — scores each candidate with FairChem UMA energy, writes `rankings.csv`
3. `export-cifs` — writes `.cif` files to disk from saved samples

## Package layout

- `clari/inference/core.py` — CLI entrypoint (`clari` command), argument parsing, calls `sample()`
- `clari/inference/inputs.py` — `SampleRequest` dataclass, `build_request()`, config/CLI parsing, Hub checkpoint resolution
- `clari/inference/sample.py` — `ClariSampler` class, model loading, OOM-safe batch sampling, shard writing, multi-GPU dispatch
- `clari/inference/rank.py` — `rank()` wrapper: calls `compute_energies`, joins with predictions, writes `rankings.csv`
- `clari/inference/export.py` — `export_cifs()`: reads parquet, optionally filters by rank/id/index, writes `.cif` files

Do not reference deleted modules: `cli.py`, `sampler.py`, `runner.py`, `io.py`.

## Key concepts

**`--copies` (Z value):** Number of molecules per unit cell. Default is 4, which covers the most common organic packing motifs. Use smaller values (1–2) for large molecules or to explore low-Z polymorphs. For multi-component requests (co-crystals), `copies` is per-component.

**`SampleRequest`:** The core data object. Fields:
- `smiles: str | list[tuple[str, int]]` — single-component SMILES string, or list of `(smiles, copies)` pairs for co-crystals
- `id: str | None = None` — run identifier, used as subdirectory name and row key; auto-generated from SMILES if omitted
- `copies: int = 4` — molecules per unit cell (single-component only; ignored for co-crystal form)
- `samples: int = 1` — how many candidate structures to generate

## Available models

| Name | Alias | Notes |
|------|-------|-------|
| `clari-m` | `clari-med` | Medium — fastest, good for exploration |
| `clari-l` | `clari-large` | Large |
| `clari-h` | `clari-huge` | Huge — highest quality, slowest |

Downloaded automatically from HuggingFace (`the-matter-lab/clari`) on first use. Pass a local `.ckpt` path to use a custom checkpoint.

## CLI — sampling

### Single molecule

```bash
uv run clari \
  --smiles "CCO" \
  --samples 8 \
  --output_dir results/ethanol
```

Writes:
- `results/ethanol/predictions.parquet` — one row per sample: `id`, `sample_idx`, `cif`
- `results/ethanol/config.json` — full run config for reproducibility

### Multi-component (co-crystal)

Repeated `--smiles` in one CLI call describes one composition, not multiple independent jobs:

```bash
uv run clari \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --copies 1 \
  --smiles "O" \
  --copies 3 \
  --samples 8 \
  --output_dir results/aspirin_trihydrate
```

### Batch via config file

For multiple independent requests, use `--config`:

```bash
uv run clari --config batch.json
```

Config schema:

```json
{
  "checkpoint_path": "clari-m",
  "output_dir": "results/batch_run",
  "requests": [
    {
      "id": "ethanol",
      "smiles": "CCO",
      "copies": 4,
      "samples": 4
    },
    {
      "id": "aspirin_trihydrate",
      "smiles": [
        ["CC(=O)Oc1ccccc1C(=O)O", 1],
        ["O", 3]
      ],
      "samples": 4
    }
  ]
}
```

Top-level config keys (all optional, override CLI defaults):
- `checkpoint_path` — model name or local path
- `output_dir` — where to write results
- `use_ema`, `use_bf16`, `pbar` — booleans
Per-request keys: `id`, `smiles`, `copies`, `samples`, `batch_size`.

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--smiles` | required | SMILES string (repeatable for co-crystal) |
| `--copies` | 4 | Molecules per unit cell (repeatable, matched by index to `--smiles`) |
| `--samples` | 1 | Number of candidate structures to generate |
| `--id` | auto | Labels every row in `predictions.parquet` and becomes the CIF subdirectory name. Auto-generated from SMILES if omitted. Valid characters: letters, digits, `.`, `_`, `-`; others are replaced with `_`. Max 80 chars. |
| `--config` | — | Path to batch JSON config (mutually exclusive with direct SMILES) |
| `--checkpoint_path` | `clari-m` | Model name (`clari-m/l/h`) or local `.ckpt` path |
| `--output_dir` | — | Directory to write results; required for multi-GPU |
| `--batch_size` | auto | Samples per forward pass; auto-scaled to GPU memory if unset |
| `--num_gpus` | 1 | Number of GPUs (requires `--output_dir`) |
| `--device` | auto | `cuda`, `mps`, `cpu`, or `cuda:N` |
| `--n_steps` | 50 | Flow matching steps |
| `--torch_threads` | 1 | CPU thread count |
| `--compile` | off | Enable `torch.compile` (off by default; gives a meaningful speedup on CUDA after cold-start) |
| `--overwrite` | off | Overwrite existing output directory |
| `--no_ema` | off | Use raw weights instead of EMA weights |
| `--no_bf16` | off | Disable bfloat16 (CUDA only) |
| `--no_pbar` | off | Suppress progress bar |

## CLI — ranking

Requires `fairchem-core`. Install with `pip install clari[uma]"` or `uv sync --extra uma`.

```bash
uv run rank results/ethanol
```

Writes:
- `results/ethanol/energies.csv` — `sample_idx`, `energies` (UMA energy per structure)
- `results/ethanol/rankings.csv` — `sample_idx`, `id`, `energies`, `rank` (0-based rank within each `id` group)

Flags: `--batch_size 32`, `--num_gpus 1`, `--torch_threads 1`, `--overwrite`.

## CLI — export

```bash
# All samples
uv run export-cifs results/ethanol

# Top 3 ranked (requires rankings.csv)
uv run export-cifs results/ethanol --top_k 3

# Specific sample indices
uv run export-cifs results/ethanol --sample_idx 0 --sample_idx 2

# Filter by request id
uv run export-cifs results/ethanol --ids ethanol

# Custom output directory
uv run export-cifs results/ethanol --output_dir my_cifs/
```

`export-cifs` works with or without `rankings.csv`. Without it, all samples are exported and named by index. With it, filenames include the rank and `--top_k` filtering becomes available.

Writes CIFs to `<output_dir>/<id>/`:
- Without rankings: `sample_000000.cif`
- With rankings: `rank_0000_sample_000000.cif`

Flags: `--output_dir`, `--rankings_path`, `--top_k`, `--ids` (repeatable), `--sample_idx` (repeatable), `--overwrite`.

## Python API

```python
from clari.inference import ClariSampler

sampler = ClariSampler("clari-m")

# Single molecule — in-memory
crystals = sampler.sample("CCO", id="ethanol", samples=8)

# Single molecule — disk-backed
sampler.sample("CCO", id="ethanol", samples=8, output_dir="results/ethanol")

# Co-crystal: dot-separated SMILES, uniform copies (2 ethanols + 2 waters)
sampler.sample("CCO.O", id="ethanol_hydrate", copies=2, samples=4)

# Co-crystal: list of SMILES, per-component copies
sampler.sample(
    ["CC(=O)Oc1ccccc1C(=O)O", "O"],
    id="aspirin_trihydrate",
    copies=[1, 3],
    samples=4,
    output_dir="results/aspirin_trihydrate",
)

```

`sample()` keyword arguments:
- `id` — run identifier; auto-generated from SMILES if omitted
- `copies: int | list[int] = 4` — molecules per unit cell; int for uniform, list for per-component
- `samples: int = 1`
- `output_dir` — if set, writes to disk and returns the output `Path`; predictions at `<output_dir>/predictions.parquet`

`ClariSampler` constructor: `checkpoint` (hub name `"clari-m/l/h"` or local `.ckpt` path), `device` (default `"auto"`), `use_ema` (default `True`), `use_bf16` (default `True`), `n_steps` (default `50`), `compile` (default `False`), `num_gpus` (default `1`). Use `ClariSampler.from_checkpoint(path)` to be explicit about loading a local file.

## Ranking and export from Python

```python
from clari.inference import rank, export_cifs

# Rank by UMA energy — returns DataFrame: sample_idx, id, energies, rank
df = rank("results/ethanol")

# Export from disk
export_cifs("results/ethanol")
export_cifs("results/ethanol", top_k=3)           # top 3 ranked (requires rankings.csv)
export_cifs("results/ethanol", sample_idx=[0, 2]) # specific indices
export_cifs("results/ethanol", output_dir="my_cifs/ethanol")

# Export directly from an in-memory list of Crystal objects
crystals = sampler.sample("CCO", id="ethanol", samples=8)
export_cifs(crystals, output_dir="my_cifs/", id="ethanol")
```

## Output files reference

| File | Written by | Contents |
|------|-----------|---------|
| `predictions.parquet` | `clari` | `id`, `sample_idx`, `cif` |
| `config.json` | `clari` | Full run config for reproducibility |
| `energies.csv` | `rank` | `sample_idx`, `energies` |
| `rankings.csv` | `rank` | `sample_idx`, `id`, `energies`, `rank` (0-based within id group) |
| `cifs/<id>/` | `export-cifs` | `.cif` files, named by rank and/or sample index |

## Operational notes

- Use `uv run` from the source checkout, or install the package and call the entry points directly.
- Without `output_dir`, `sample()` holds everything in memory and returns `list[Crystal]`.
- With `output_dir`, results are written incrementally as shards and merged at the end; safe to interrupt and re-run with `--overwrite`.
- Multi-GPU sampling (`num_gpus > 1`) requires `output_dir` and CUDA.
- OOM is handled automatically: batch size halves and retries until it fits.
- `--compile` is off by default. Pass `--compile` (CLI) or `compile=True` (Python) to enable; gives a meaningful speedup on CUDA after cold-start compilation.
- `rank` path requires `fairchem-core` (`clari[uma]`); sampling and export do not.
