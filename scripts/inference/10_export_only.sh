#!/usr/bin/env bash
set -euo pipefail

input_path="${1:?Usage: $0 INPUT_PATH}"

uv run export-cifs "${input_path}" --overwrite true
