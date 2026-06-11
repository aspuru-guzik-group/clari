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

| Name | Notes |
|------|-------|
| `clari-m` | Medium — fastest, good for exploration |
| `clari-l` | Large |
| `clari-h` | Huge — highest quality, slowest |

Downloaded automatically from HuggingFace (`the-matter-lab/clari`) on first use. Pass a local `.ckpt` path to use a custom checkpoint.

## CLI — sampling

### The input grammar (one rule)

A request is a flat list of `(component, copies)` pairs. Dots in a SMILES always
split into components; a copies value broadcasts over the dot components of its
token; omitted copies default to 4. Hydrogens are added automatically (AddHs).

```bash
# Quickstart: 10 candidates for ethanol, writes results/CCO_x4/
uv run clari "CCO" --samples 10

# Positional grammar: SMILES [copies] [SMILES [copies]]...
uv run clari "CCO" 1 "O" 3 --samples 8 --output_dir results/ethanol_trihydrate

# Dotted SMILES split; copies broadcast: (CCO,2),(O,2)
uv run clari "CCO.O" 2
```

The `--smiles`/`--copies` flag form is a pure synonym of the positional form
(use one or the other, not both):

```bash
uv run clari --smiles "CC(=O)Oc1ccccc1C(=O)O" --copies 1 --smiles "O" --copies 3 --samples 8

# Repeated --copies with a single dotted --smiles distributes per component
uv run clari --smiles "CCO.O" --copies 1 --copies 3
```

Repeated SMILES in one CLI call describes one composition, not multiple
independent jobs — use `--config` for batches.

Writes (default `--output_dir` is `results/<id>`):
- `<output_dir>/predictions.parquet` — one row per sample: `id`, `sample_idx`, `cif`
- `<output_dir>/config.json` — full run config for reproducibility

### Batch via config file

For multiple independent requests, use `--config`. Batch configs write one
result directory per request, not one mixed parquet:

```bash
uv run clari --config batch.json
```

Config schema:

```json
{
  "model": "clari-m",
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
- `model` — model name (`clari-m`, `clari-l`, `clari-h`)
- `output_dir` — where to write results
- `use_ema`, `use_bf16`, `pbar` — booleans
Per-request keys: `id`, `smiles`, `copies`, `samples`, `batch_size`.

Output layout:

```text
results/batch_run/
  manifest.json
  ethanol/
    predictions.parquet
    config.json
  aspirin_trihydrate/
    predictions.parquet
    config.json
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| positional | — | `SMILES [copies] [SMILES [copies]]...` |
| `--smiles` | — | SMILES string (repeatable; synonym of positional form) |
| `--copies` | 4 | Molecules per unit cell (repeatable, matched by index to `--smiles`) |
| `--samples` | 1 | Number of candidate structures to generate |
| `--id` | auto | Labels every row in `predictions.parquet` and becomes the CIF subdirectory name. Prefer setting it explicitly — the auto-generated SMILES-based name is cryptic and can collide. Valid characters: letters, digits, `.`, `_`, `-`; others are replaced with `_`. Max 80 chars. |
| `--config` | — | Path to batch JSON config (mutually exclusive with direct SMILES) |
| `--model` | `clari-m` | Model name (`clari-m`, `clari-l`, `clari-h`) |
| `--output_dir` | `results/<id>` | Directory to write results |
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

# Export one request from a batch run
uv run export-cifs results/batch_run/ethanol

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

`ClariSampler` constructor: `checkpoint` (`"clari-m"`, `"clari-l"`, or `"clari-h"`), `device` (default `"auto"`), `use_ema` (default `True`), `use_bf16` (default `True`), `n_steps` (default `50`), `compile` (default `False`), `num_gpus` (default `1`), `filter_clashing` (default `False`; when `True`, drops sampled structures with inter-molecular atom clashes instead of resampling, so fewer than `samples` may be returned — clashes are rare at a high `n_steps` like 50, so few or none are dropped).

## Ranking and export from Python

```python
from clari.inference import rank, export_cifs

# Rank by UMA energy — returns DataFrame: sample_idx, id, energies, rank
df = rank("results/ethanol")   # writes energies.csv + rankings.csv next to predictions.parquet

# Rank an in-memory list of Crystal objects — writes nothing
crystals = sampler.sample("CCO", id="ethanol", samples=8)
df = rank(crystals)

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
