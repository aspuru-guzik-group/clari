<div align="center">

# Fast Organic Crystal Structure Prediction <br> with Unit Cell Flow Matching

[![arXiv](https://img.shields.io/badge/arXiv-2606.03199-b31b1b.svg)](https://arxiv.org/abs/2606.03199)
[![data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-blue)](https://huggingface.co/the-matter-lab/clari)

<br>

<img src="data/sampling.gif" width=500>

</div>

<br>

This repository contains code to reproduce the paper: Fast Organic Crystal Structure Prediction with Unit Cell Flow Matching ([arXiv](https://arxiv.org/abs/2606.03199)).

---

## Installation

Using pip:

```bash
pip install clari
```

Or by cloning this repository and running:

```bash
uv sync
```

## Usage

### Sampling tutorial

The main inference workflow is:

1. `clari` — samples candidate crystal structures, writes `predictions.parquet`
2. `rank` — scores candidates with FairChem UMA, writes `rankings.csv`
3. `export-cifs` — writes `.cif` files to disk

Available models: `clari-m`, `clari-l`, `clari-h`. Models are downloaded automatically from HuggingFace on first use.

**`--copies`** is the number of molecules per unit cell (crystallographic Z value). The default is 4, which covers the most common organic packing motifs.

**Hydrogen atoms** are added automatically. Write SMILES without explicit Hs. Pass `--no_add_hs` once per component to disable H addition for that component — the nth flag applies to the nth molecule in order. The model was trained on all-hydrogen structures, so H atoms should almost always be present.

#### 1. Sample one molecule

```bash
uv run clari \
  --smiles "CCO" \
  --id ethanol \
  --n_samples 8 \
  --output_dir results/ethanol
```

Writes:
- `results/ethanol/predictions.parquet` — one row per sample: `id`, `sample_idx`, `cif`
- `results/ethanol/config.json` — full run config

In simple terms: generate 8 crystal packings for ethanol and save the run.

**`--id`** labels every row in `predictions.parquet` and becomes the subdirectory name when exporting CIFs (e.g. `cifs/ethanol/sample_000000.cif`). If omitted, an id is auto-generated from the SMILES string. Valid characters are letters, digits, `.`, `_`, and `-`; anything else is replaced with `_`. Max 80 characters.

#### 2. Sample a co-crystal

Repeating `--smiles` in one call describes one multi-component composition, not multiple jobs:

```bash
uv run clari \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --copies 1 \
  --smiles "O" \
  --copies 3 \
  --n_samples 8 \
  --output_dir results/aspirin_trihydrate
```

In simple terms: sample one crystal made of one aspirin and three water molecules per unit cell.

#### 3. Sample a batch from config

For multiple independent requests, use `--config` with a JSON file:

```json
{
  "checkpoint_path": "clari-m",
  "output_dir": "results/batch_run",
  "requests": [
    {
      "id": "ethanol",
      "smiles": "CCO",
      "copies": 4,
      "n_samples": 4
    },
    {
      "id": "aspirin_trihydrate",
      "smiles": [
        ["CC(=O)Oc1ccccc1C(=O)O", 1],
        ["O", 3]
      ],
      "n_samples": 4
    }
  ]
}
```

```bash
uv run clari --config batch.json
```

Top-level config keys (all optional): `checkpoint_path`, `output_dir`, `use_ema`, `use_bf16`, `pbar`, `add_hs` (global H-addition default for all requests).

Per-request keys: `id`, `smiles`, `copies`, `n_samples`, `add_hs`.

#### 4. Rank samples

Ranking requires `fairchem-core`:

```bash
pip install "clari[uma]"
# or from source:
uv sync --extra uma
```

```bash
uv run rank results/ethanol
```

Writes:
- `results/ethanol/energies.csv` — `sample_idx`, `energies` (UMA energy per structure)
- `results/ethanol/rankings.csv` — `sample_idx`, `id`, `energies`, `rank` (0-based within each `id` group)

#### 5. Export CIF files

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

CIF filenames:
- Without rankings: `<id>/sample_000000.cif`
- With rankings: `<id>/rank_0000_sample_000000.cif`

#### 6. Python API

```python
from clari.inference import ClariSampler

sampler = ClariSampler("clari-m")

# Single molecule — in-memory
crystals = sampler.sample("CCO", id="ethanol", n_samples=8)

# Single molecule — disk-backed
sampler.sample("CCO", id="ethanol", n_samples=8, output_dir="results/ethanol")

# Co-crystal: dot-separated SMILES, uniform copies (2 ethanols + 2 waters per cell)
sampler.sample("CCO.O", id="ethanol_hydrate", copies=2, n_samples=4)

# Co-crystal: list of SMILES, per-component copies (1 aspirin + 3 waters per cell)
sampler.sample(
    ["CC(=O)Oc1ccccc1C(=O)O", "O"],
    id="aspirin_trihydrate",
    copies=[1, 3],
    n_samples=4,
    output_dir="results/aspirin_trihydrate",
)
```

`sample()` keyword arguments: `id` (auto-generated from SMILES if omitted), `copies` (default 4, int or list of ints for per-component), `n_samples` (default 1), `add_hs` (`bool` or `list[bool]`, default `True`), `output_dir`.

#### 7. Rank and export from Python

```python
from clari.inference import rank, export_cifs

# Rank by UMA energy — writes energies.csv and rankings.csv (requires a path, not in-memory)
rank("results/ethanol")

# Export from disk
export_cifs("results/ethanol")
export_cifs("results/ethanol", top_k=3)           # top 3 ranked (requires rankings.csv)
export_cifs("results/ethanol", sample_idx=[0, 2]) # specific indices
export_cifs("results/ethanol", output_dir="my_cifs/ethanol")

# Export directly from an in-memory list of Crystal objects
crystals = sampler.sample("CCO", id="ethanol", n_samples=8)
export_cifs(crystals, output_dir="my_cifs/", id="ethanol")
```

Agent-facing inference reference: [clari/inference/SKILL.md](clari/inference/SKILL.md).

## Development Installation

To install the full development environment:

```bash
uv sync --extra dev
```

⚠️ To generate data and run COMPACK, we require the **CCDC SDK**, whose dependencies conflict with FairChem. Thus, some scripts run as standalone uv scripts that resolve their own isolated environments from the CCDC index. The first invocation of `uv run -s *.py ...` resolves and caches that environment. You will still need a valid CCDC license configured on the machine.

## Data

### Generation

We expect the final data folder to be structured as follows:

```
data/
    raw/
        csd_metadata.parquet
        csd_conquest.parquet
    csd/
        config.json
        metdata.parquet
        {train,val,test}.pt
```

To generate the data, first extract the metadata of entries in CSD:

```
uv run -s scripts/data/0_metadata.py
```

This creates the `csd_metadata.parquet` file from above. Next, download **ALL** of CSD in `.mol2` and `.cif` format using ConQuest (not `csd-python-api` since it sanitizes molecules and removes some bond information) into the `csd_conquest.parquet` file. Finally, generate the `data/csd` folder with:

```
uv run python -m scripts.data.1_process --num_workers=16
```

For reference, the CSD refcodes we use and our dataset split are uploaded to [HuggingFace](https://huggingface.co/the-matter-lab/clari).

## Evaluation

Training and evaluation paths default to `data/`, `results/`, and `logs/` under the current working directory. Override them with `CLARI_DATA_DIR`, `CLARI_RESULTS_DIR`, and `CLARI_LOG_DIR`; see [`clari/paths.py`](clari/paths.py).

### OXtal and Teaching Test Sets

To reproduce the paper numbers for a Hub model or local checkpoint, run the stages below in order. Each stage writes into the same `<experiment_dir>` and reads what the previous stage produced.

These evaluation commands require the prepared `data/csd` directory described above. When running CLARI from an installed package, run the commands from a working directory containing `data/csd`, or set `CLARI_DATA_DIR=/path/to/data`.

```bash
# 0. One-time: build the GT CIF cache the standalone compack script reads
uv run python clari/evaluation/build_test_cifs_cache.py

# 1. Sample the CSD test set, creates a folder results/experiment_dir.
#    Use clari-m, clari-l, or a local checkpoint path as the first argument.
uv run sample-test clari-m <num_samples> <experiment_dir> --subset <teaching/oxtal>

# 2. Clash check (writes collision.csv)
uv run collision <experiment_dir>

# 3. UMA energies (writes energies.csv)
uv run compute-energies <experiment_dir>

# 4. COMPACK packing similarity (writes compack.csv, isolated uv script env)
uv run -s clari/evaluation/compack.py <experiment_dir> --num_processes n

# 5. Summary table (SolC per subset, all k)
uv run summarize <experiment_dir>
```

### Ablations

The exact commands used for to train our ablated and final models can be found in `scripts/train`. After running inference as above, the metrics used for ablations are defined in:

```
from clari.pipelines.utils.metrics import assess_crystals_eval
```

## Citation

```bibtex
@misc{lo2026clari,
      title={Fast Organic Crystal Structure Prediction with Unit Cell Flow Matching},
      author={Alston Lo and Luka Mucko and Austin H. Cheng and Andy Cai and Alastair J. A. Price and Wojciech Matusik and Alán Aspuru-Guzik},
      year={2026},
      eprint={2606.03199},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.03199},
}
```
