<div align="center">

# Fast Organic Crystal Structure Prediction <br> with Unit Cell Flow Matching

[![arXiv](https://img.shields.io/badge/arXiv-2606.03199-b31b1b.svg)](https://arxiv.org/abs/2606.03199)
[![data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-blue)](https://huggingface.co/the-matter-lab/clari)

<br>

<img src="data/sampling.gif" width=500>

</div>

<br>

---

## Installation

```bash
pip install clari
```

Or from source:

```bash
uv sync
```

## Inference

The workflow has three steps:

1. `clari` — sample candidate crystal structures → `predictions.parquet`
2. `rank` — score with FairChem UMA energy → `rankings.csv`
3. `export-cifs` — write `.cif` files to disk

Models (`clari-m`, `clari-l`, `clari-h`) download automatically from HuggingFace on first use.

### Single molecule

```bash
uv run clari \
  --smiles "CCO" \
  --id ethanol \
  --samples 8 \
  --output_dir results/ethanol
```

**`--copies`** sets the number of molecules per unit cell (Z value, default 4). **`--id`** labels the output rows and becomes the CIF subdirectory name; auto-generated from SMILES if omitted. Hydrogens are added automatically — write SMILES without explicit Hs.

### Co-crystal

Repeated `--smiles` flags describe one multi-component composition:

```bash
uv run clari \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" --copies 1 \
  --smiles "O"                       --copies 3 \
  --samples 8 \
  --output_dir results/aspirin_trihydrate
```

### Batch via config

```bash
uv run clari --config batch.json
```

```json
{
  "checkpoint_path": "clari-m",
  "output_dir": "results/batch_run",
  "requests": [
    { "id": "ethanol", "smiles": "CCO", "copies": 4, "samples": 4 },
    {
      "id": "aspirin_trihydrate",
      "smiles": [["CC(=O)Oc1ccccc1C(=O)O", 1], ["O", 3]],
      "samples": 4
    }
  ]
}
```

Top-level keys (all optional): `checkpoint_path`, `output_dir`, `use_ema`, `use_bf16`, `pbar`, `add_hs`.
Per-request keys: `id`, `smiles`, `copies`, `samples`, `add_hs`.

### Rank

Requires `fairchem-core`:

```bash
pip install "clari[uma]"   # or: uv sync --extra uma
uv run rank results/ethanol
```

### Export CIFs

```bash
uv run export-cifs results/ethanol                         # all samples
uv run export-cifs results/ethanol --top_k 3               # top 3 ranked (requires rankings.csv)
uv run export-cifs results/ethanol --sample_idx 0 --sample_idx 2
uv run export-cifs results/batch_run --ids ethanol         # one molecule from a batch parquet
uv run export-cifs results/ethanol --output_dir my_cifs/
```

Filenames: `<id>/sample_000000.cif` without rankings, `<id>/rank_0000_sample_000000.cif` with.

### Python API

```python
from clari.inference import ClariSampler

sampler = ClariSampler("clari-m")

crystals = sampler.sample("CCO", id="ethanol", samples=8)                    # in-memory
sampler.sample("CCO", id="ethanol", samples=8, output_dir="results/ethanol") # disk-backed

# Co-crystal: dot-separated SMILES (uniform copies) or list (per-component copies)
sampler.sample("CCO.O", id="ethanol_hydrate", copies=2, samples=4)
sampler.sample(
    ["CC(=O)Oc1ccccc1C(=O)O", "O"],
    id="aspirin_trihydrate",
    copies=[1, 3],
    samples=4,
    output_dir="results/aspirin_trihydrate",
)
```

`sample()` kwargs: `id`, `copies` (int or list, default 4), `samples` (default 1), `add_hs` (bool or list, default `True`), `output_dir`.

### Rank and export from Python

```python
from clari.inference import rank, export_cifs

rank("results/ethanol")  # requires a path — does not work on in-memory Crystal lists

export_cifs("results/ethanol")
export_cifs("results/ethanol", top_k=3)
export_cifs("results/ethanol", sample_idx=[0, 2])
export_cifs("results/ethanol", output_dir="my_cifs/ethanol")

# From an in-memory list
export_cifs(crystals, output_dir="my_cifs/", id="ethanol")
```

See also: [inference reference](clari/inference/SKILL.md).

## Development

```bash
uv sync --group dev
```

⚠️ Data generation and COMPACK require the **CCDC SDK**, which conflicts with FairChem. Those scripts run as standalone `uv -s` scripts with their own isolated environments. A valid CCDC license must be configured on the machine.

## Data

Expected layout:

```
data/
    raw/
        csd_metadata.parquet
        csd_conquest.parquet
    csd/
        config.json
        metadata.parquet
        {train,val,test}.pt
```

```bash
uv run -s scripts/data/0_metadata.py                          # extract CSD metadata
# download CSD via ConQuest into csd_conquest.parquet
uv run python -m scripts.data.1_process --num_workers=16      # build data/csd/
```

CSD refcodes and dataset splits are on [HuggingFace](https://huggingface.co/the-matter-lab/clari).

## Evaluation

Paths default to `data/`, `results/`, `logs/` under the working directory. Override with `CLARI_DATA_DIR`, `CLARI_RESULTS_DIR`, `CLARI_LOG_DIR` (see [`clari/paths.py`](clari/paths.py)).

```bash
uv run python clari/evaluation/build_test_cifs_cache.py          # one-time GT CIF cache

uv run sample-test clari-m <num_samples> <experiment_dir> --subset <teaching/oxtal>

uv run collision <experiment_dir>                                 # writes collision.csv

uv run compute-energies <experiment_dir>                          # writes energies.csv

uv run -s clari/evaluation/compack.py <experiment_dir> --num_processes <n> # writes compack.csv

uv run summarize <experiment_dir>                                 # SolC per subset
```

Ablation metrics: `from clari.pipelines.utils.metrics import assess_crystals_eval`. Training scripts are in `scripts/train/`.

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
