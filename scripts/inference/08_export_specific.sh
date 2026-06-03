#!/usr/bin/env bash
set -euo pipefail

checkpoint_path="${1:?Usage: $0 CHECKPOINT_PATH}"
aspirin_smiles='[H][O][C](=[O])[c]1[c]([H])[c]([H])[c]([H])[c]([H])[c]1[O][C](=[O])[C]([H])([H])[H]'

uv run sample \
  --checkpoint_path "${checkpoint_path}" \
  --output_dir samples/aspirin_specific \
  --smiles "${aspirin_smiles}" \
  --ids aspirin \
  --copies 4 \
  --n_samples 16 \
  --overwrite true
uv run export-cifs \
  samples/aspirin_specific \
  --sample_idx 0 \
  --sample_idx 7 \
  --overwrite true
