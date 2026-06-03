# CLARI: Fast Organic Crystal Structure Prediction with Unit Cell Flow Matching

CLARI takes a molecule and predicts how it packs into a crystal. A single run produces many candidate structures.

## Links

- [Source code](https://github.com/aspuru-guzik-group/clari)
- [Paper: Fast Organic Crystal Structure Prediction with Unit Cell Flow Matching](https://arxiv.org/abs/2606.03199)
- [Checkpoints and data](https://huggingface.co/the-matter-lab/clari)

Checkpoints are available on Hugging Face as `clari-large.ckpt` and `clari-med.ckpt`.

Inputs are expected to use explicit-hydrogen SMILES. For example, prefer
`C([H])([H])([H])C([H])([H])[H]` over `CC`.

---

## Basic sampling

**CLI**

```bash
uv run sample \
  --checkpoint_path clari.ckpt \
  --output_dir out/ \
  --smiles 'C([H])([H])([H])C([H])([H])[H]' \
  --ids ethane \
  --n_samples 8
```

**Python**

```python
from clari.inference import ClariSampler, SampleRequest

sampler = ClariSampler.from_checkpoint("clari.ckpt")
samples = sampler.sample(
    SampleRequest(
        id="ethane",
        smiles="C([H])([H])([H])C([H])([H])[H]",
        n_samples=8,
    ),
    output_dir="out/",
)
```

Load from the Hub instead of a local file (downloads once, then cached):

**CLI**

```bash
uv run sample \
  --from_hub Clari-M \
  --output_dir out/ \
  --smiles 'C([H])([H])([H])C([H])([H])[H]' \
  --ids ethane \
  --n_samples 8
```

**Python**

```python
sampler = ClariSampler.from_hub("Clari-M")  # or "Clari-L"
```

---

## Multiple molecules

**CLI**

```bash
uv run sample \
  --checkpoint_path clari.ckpt \
  --output_dir out/ \
  --smiles 'C([H])([H])([H])C([H])([H])O[H]' --ids ethanol \
  --smiles 'C1([H])=C([H])C([H])=C([H])C([H])=C1[H]' --ids benzene \
  --copies 4 \
  --n_samples 50
```

**Python**

```python
from clari.inference import ClariSampler, SampleRequest

sampler = ClariSampler.from_checkpoint("clari.ckpt")
samples = sampler.sample([
    SampleRequest(
        id="ethanol",
        smiles="C([H])([H])([H])C([H])([H])O[H]",
        copies=4,
        n_samples=50,
    ),
    SampleRequest(
        id="benzene",
        smiles="C1([H])=C([H])C([H])=C([H])C([H])=C1[H]",
        copies=4,
        n_samples=50,
    ),
], output_dir="out/")
```

For co-crystals, pass `(SMILES, copy_count)` pairs. Pair-level copy counts are passed
directly to `Crystal.from_smiles`.

```python
samples = sampler.sample(
    SampleRequest(
        id="ethanol-water",
        smiles=[
            ("C([H])([H])([H])C([H])([H])O[H]", 1),
            ("O([H])[H]", 1),
        ],
        n_samples=50,
    ),
    output_dir="out/",
)
```

For many molecules, use a config file instead of repeating flags:

```bash
uv run sample --config jobs.json
```

```json
{
  "checkpoint_path": "clari.ckpt",
  "output_dir": "out/",
  "smiles": [
    "C([H])([H])([H])C([H])([H])O[H]",
    "C1([H])=C([H])C([H])=C([H])C([H])=C1[H]"
  ],
  "ids": ["ethanol", "benzene"],
  "copies": [4, 4],
  "n_samples": [50, 50]
}
```

Co-crystal configs use the same pair shape:

```json
{
  "checkpoint_path": "clari.ckpt",
  "output_dir": "out/",
  "ids": "ethanol-water",
  "smiles": [
    ["C([H])([H])([H])C([H])([H])O[H]", 1],
    ["O([H])[H]", 1]
  ],
  "n_samples": 50
}
```

---

## Sample → rank → export top-K

**CLI**

```bash
uv run sample \
  --checkpoint_path clari.ckpt \
  --output_dir out/ \
  --smiles 'C([H])([H])([H])C([H])([H])[H]' \
  --ids ethane \
  --copies 4 \
  --n_samples 64

uv run rank out/

uv run export-cifs out/ --top_k 10
```

**Python**

```python
sampler = ClariSampler.from_checkpoint("clari.ckpt")
sampler.sample(
    "C([H])([H])([H])C([H])([H])[H]",
    copies=4,
    n_samples=64,
    output_dir="out/",
)
# rank and export are CLI steps
```

---

## Export specific samples by index

```bash
uv run export-cifs out/ --sample_idx 0 --sample_idx 7
```

---

## Multi-GPU

**CLI**

```bash
uv run sample \
  --checkpoint_path clari.ckpt \
  --output_dir out/ \
  --smiles 'C([H])([H])([H])C([H])([H])[H]' \
  --ids ethane \
  --copies 4 \
  --n_samples 1000 \
  --num_gpus 4
```

**Python**

```python
sampler = ClariSampler.from_checkpoint("clari.ckpt", num_gpus=4)
sampler.sample(
    "C([H])([H])([H])C([H])([H])[H]",
    copies=4,
    n_samples=1000,
    output_dir="out/",
)
```

---

## Fixed batch size

**CLI**

```bash
uv run sample \
  --checkpoint_path clari.ckpt \
  --output_dir out/ \
  --smiles 'C([H])([H])([H])C([H])([H])[H]' \
  --ids ethane \
  --copies 4 \
  --n_samples 32 \
  --batch_size 8 \
  --compile false
```

**Python**

```python
sampler = ClariSampler.from_checkpoint("clari.ckpt", compile=False)
sampler.sample(
    "C([H])([H])([H])C([H])([H])[H]",
    copies=4,
    n_samples=32,
    batch_size=8,
    output_dir="out/",
)
```

---

## CPU smoke test

**CLI**

```bash
uv run sample \
  --checkpoint_path clari.ckpt \
  --output_dir out/ \
  --smiles 'C([H])([H])([H])C([H])([H])[H]' \
  --ids ethane \
  --n_samples 1 \
  --batch_size 1 \
  --device cpu \
  --n_steps 2 \
  --compile false \
  --use_bf16 false
```

**Python**

```python
sampler = ClariSampler.from_checkpoint(
    "clari.ckpt",
    device="cpu",
    n_steps=2,
    compile=False,
    use_bf16=False,
)
sampler.sample(
    "C([H])([H])([H])C([H])([H])[H]",
    n_samples=1,
    batch_size=1,
    output_dir="out/",
)
```

---

For all options: `uv run sample --help`

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
