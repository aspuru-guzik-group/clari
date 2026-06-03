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

To install CLARI as a package:

```bash
uv pip install clari
```

or:

```bash
pip install clari
```

Install everything needed for training, sampling, ranking, and most of the evaluation pipeline:

```bash
uv sync
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

## Usage

After installing CLARI in an activated environment, use the command-line tools directly:

```bash
sample --help
sample-test --help
rank --help
export-cifs --help
summarize --help
skill
```

When working from a source checkout with `uv sync`, prefix commands with `uv run`, for example:

```
uv run train --help
```

Checkpoints for Clari-M and Clari-L are uploaded to [HuggingFace](https://huggingface.co/the-matter-lab/clari).
To run inference, see the [README.md](https://github.com/aspuru-guzik-group/clari/blob/main/clari/inference/README.md) in `clari.inference`.
We also include a [SKILL.md](https://github.com/aspuru-guzik-group/clari/blob/main/clari/inference/SKILL.md) so that coding agents can run inference; installed environments can print it with `skill` (`uv run skill`).

## Evaluation

Training and evaluation paths default to `data/`, `results/`, and `logs/` under the current working directory. Override them with `CLARI_DATA_DIR`, `CLARI_RESULTS_DIR`, and `CLARI_LOG_DIR`; see [`clari/paths.py`](https://github.com/aspuru-guzik-group/clari/blob/main/clari/paths.py).

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
@misc{lo2026fastorganiccrystalstructure,
      title={Fast Organic Crystal Structure Prediction with Unit Cell Flow Matching},
      author={Alston Lo and Luka Mucko and Austin H. Cheng and Andy Cai and Alastair J. A. Price and Wojciech Matusik and Alán Aspuru-Guzik},
      year={2026},
      eprint={2606.03199},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.03199},
}
```
