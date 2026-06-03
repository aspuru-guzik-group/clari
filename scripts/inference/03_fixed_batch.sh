#!/usr/bin/env bash
set -euo pipefail

checkpoint_path="${1:?Usage: $0 CHECKPOINT_PATH}"
aspirin_smiles='[H][O][C](=[O])[c]1[c]([H])[c]([H])[c]([H])[c]([H])[c]1[O][C](=[O])[C]([H])([H])[H]'

uv run sample \
  --checkpoint_path "${checkpoint_path}" \
  --output_dir samples/aspirin_fixed_batch \
  --smiles "${aspirin_smiles}" \
  --ids aspirin \
  --copies 4 \
  --n_samples 32 \
  --batch_size 8 \
  --compile false \
  --overwrite true
