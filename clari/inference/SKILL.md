---
name: clari
description: Predict organic crystal structures from a molecule using CLARI. Use whenever the user asks to predict, generate, sample, propose, or simulate crystal structures, crystal packings, polymorph candidates, or unit cells for a molecule, SMILES string, or chemical name (e.g. "predict the crystal structure of aspirin", "give me 20 packings of ethanol", "sample crystals for this SMILES", "predict possible polymorphs of paracetamol", "run CSP on this molecule"). Also use for ranking already-generated crystals by FairChem energy, exporting CIF files from an existing CLARI run, or any question about running `uv run sample`, `uv run rank`, or `uv run export-cifs`.
---

# CLARI Crystal Structure Prediction

CLARI is a diffusion model that samples organic crystal packings (unit cell + atomic positions) from molecule specifications. A single-component input is a SMILES string. Co-crystals, hydrates, and solvates are represented as lists of `(SMILES, copy_count)` pairs. Use this skill to map natural-language CSP requests onto the CLARI CLI defined in `clari/inference/`.

## The three-stage workflow

Each stage consumes the previous stage's output directory:

1. **`sample`** — given SMILES, produce N candidate crystals → `<dir>/predictions.parquet`
2. **`rank`** *(optional)* — score samples with a FairChem UMA energy model → `<dir>/rankings.csv`
3. **`export-cifs`** *(optional)* — write CIF files to disk → `<dir>/cifs/<id>/...`

Stop at stage 1 if the user just wants candidates. Continue to stage 2 if they say "best", "lowest-energy", "ranked", or "good". Continue to stage 3 if they want CIF files on disk.

## What you need from the user

In priority order:

1. **A molecule.** Accept an explicit-hydrogen SMILES string, a chemical name, an RDKit mol, or a co-crystal list of `(SMILES, copy_count)` pairs. If they give a name (ethanol, aspirin, paracetamol, urea), convert it to explicit-hydrogen SMILES yourself only for common molecules you are confident about. For unfamiliar or ambiguous names, confirm the SMILES with the user before running. **Never guess a SMILES.**
2. **A checkpoint or Hub model.** Either pass a local checkpoint path with `--checkpoint_path` or a Hub model name with `--from_hub` (`Clari-M`, `Clari-L`). Hub checkpoints are available at `the-matter-lab/clari-data` as `clari-med.ckpt` and `clari-large.ckpt`, but Hub access requires internet or a warm local cache. Do not start sampling against a missing local checkpoint or an unavailable Hub cache.
3. **`n_samples`** — how many structures to generate. Use whatever the user said. If they didn't say, default to 100.
4. **`copies`** — molecules per asymmetric unit for single-component inputs. The CLI and Python defaults are 4. For agent-run CSP requests, default to **4** for small organics unless the user specified otherwise. Use 1 only when the molecule is very large (>100 heavy atoms) or the user explicitly asks. For co-crystals, hydrates, and solvates, encode the intended stoichiometry with per-component counts in the `(SMILES, copy_count)` pairs. The counts may differ between components. Ask if uncertain — wrong copy counts produce unphysical results.
5. **Output directory.** Default to `out/<id>/` where `<id>` is the user-provided id or a sanitized molecule name.

CLARI expects explicit-hydrogen SMILES. Prefer `C([H])([H])([H])C([H])([H])[H]` over `CC`.

## Default sampling command

Always run via `uv` (per project convention):

```bash
uv run sample \
  --checkpoint_path clari.ckpt \
  --output_dir out/<id> \
  --smiles '<explicit-H SMILES>' \
  --ids <id> \
  --copies <copies> \
  --n_samples <N>
```

If using a cached or internet-accessible Hub checkpoint instead of a local file:

```bash
uv run sample \
  --from_hub Clari-M \
  --output_dir out/<id> \
  --smiles '<explicit-H SMILES>' \
  --ids <id> \
  --copies <copies> \
  --n_samples <N>
```

Useful tweaks:

- `--n_steps 50` — faster sampling, slightly lower quality
- `--num_gpus N` — split across multiple GPUs
- `--device cpu --compile false --use_bf16 false --n_steps 2 --n_samples 1` — quick CPU smoke test
- `--batch_size N` — pin batch size if automatic halve-and-retry on OOM still fails
- `--overwrite true` — replace an existing CLARI output folder
- clashing samples are dropped and the deficit is resampled by default; pass `--filter_clashing false` if the user wants every sample regardless of clashes. The resample loop is capped at `--max_resample_factor` × `n_samples` total attempts (default 10).

## Ranking and exporting

When the user asks for "best", "top", "lowest energy", or "ranked" structures:

```bash
uv run rank out/<id>
uv run export-cifs out/<id> --top_k <K>
```

When they want all CIFs without ranking:

```bash
uv run export-cifs out/<id>
```

For specific samples:

```bash
uv run export-cifs out/<id> --sample_idx 0 --sample_idx 42
```

## Python alternative

Use this in a notebook context, or when the user asks for a Python snippet:

```python
from clari.inference import ClariSampler, SampleRequest

# from a local checkpoint
sampler = ClariSampler.from_checkpoint("clari.ckpt")

# or from the Hub (downloads once, then cached; requires internet or a warm cache)
sampler = ClariSampler.from_hub("Clari-M")  # or "Clari-L"

samples = sampler.sample(
    SampleRequest(
        id="ethanol",
        smiles="C([H])([H])([H])C([H])([H])O[H]",
        copies=4,
        n_samples=20,
    )
)
# samples[i] has .crystal, .sample_idx, .id
# samples[i].crystal.to_cif() gives a CIF string
```

Passing `output_dir="..."` switches to shard-and-merge: results stream to disk as they're produced and the call returns the `predictions.parquet` path instead of in-memory samples. For multi-GPU Python sampling:

```python
sampler = ClariSampler.from_checkpoint("clari.ckpt", num_gpus=4)
predictions_path = sampler.sample(
    SampleRequest(
        id="ethanol",
        smiles="C([H])([H])([H])C([H])([H])O[H]",
        copies=4,
        n_samples=20,
    ),
    output_dir="out/ethanol",
)
```

## Config files

For more than two or three molecules, prefer a JSON or YAML config over long parallel `--smiles`/`--ids`/`--copies`/`--n_samples` flag chains. Every `sample` flag is a key in the file.

```bash
uv run sample --config jobs.json
```

CLI flags layered on top of `--config` override the file values. An example co-crystal config lives at `scripts/inference/sample_config.json`.

## Multi-component asymmetric units

Co-crystals, hydrates, and solvates use lists of `(SMILES, copy_count)` pairs. Prefer config files for CLI runs because repeated `--smiles` flags mean multiple independent requests, not one co-crystal.

Python:

```python
samples = sampler.sample(
    SampleRequest(
        id="ethanol-water",
        smiles=[
            ("C([H])([H])([H])C([H])([H])O[H]", 1),
            ("O([H])[H]", 1),
        ],
        n_samples=20,
    )
)
```

Config:

```json
{
  "checkpoint_path": "clari.ckpt",
  "output_dir": "out/ethanol-water",
  "ids": "ethanol-water",
  "smiles": [
    ["C([H])([H])([H])C([H])([H])O[H]", 1],
    ["O([H])[H]", 1]
  ],
  "n_samples": 20
}
```

## Common request → command map

| User says | What to run |
|---|---|
| "predict the crystal structure of X" | `sample` with `n_samples=100` |
| "give me N structures of X" | `sample` with `n_samples=N` |
| "sample crystals of X with K copies" | `sample` with `copies=K` |
| "predict the crystal structure of the X·Y co-crystal" | `sample` from a config with `smiles=[[X_smiles, X_copies], [Y_smiles, Y_copies]]` |
| "rank the structures in F" | `uv run rank F` |
| "export top K CIFs from F" | rank (if not yet), then `uv run export-cifs F --top_k K` |
| "give me 20 structures of ethanol and the top 5 CIFs" | sample 20 → rank → export top 5 |
| "do a quick CPU test" | sample with the CPU smoke-test flags above |

## Sanity checks before running

- The molecule has been resolved to a SMILES you trust.
- The local checkpoint exists, or the requested Hub model is available from internet/cache.
- Sampling is a long-running command. For large jobs, run it as a background shell session and report progress instead of blocking the conversation.

## Reference files

- User docs: `clari/inference/README.md`
- CLI: `clari/inference/cli.py`, `clari/inference/rank.py`, `clari/inference/export.py`
- Python API: `clari/inference/sampler.py` (`ClariSampler`, `SampleRequest`, `CrystalSample`)
- Example shell scripts: `scripts/inference/`
