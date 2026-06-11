from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

HUB_MODELS: dict[str, tuple[str, str]] = {
    "clari-m": ("the-matter-lab/clari", "clari-med.ckpt"),
    "clari-l": ("the-matter-lab/clari", "clari-large.ckpt"),
    "clari-h": ("the-matter-lab/clari", "clari-huge.ckpt"),
}


def sanitize_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text[:80] or "sample"


@dataclass
class SampleRequest:
    smiles: str | list[tuple[str, int]]
    id: str | None = None
    copies: int = 4
    samples: int = 1
    batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            if isinstance(self.smiles, str):
                self.id = sanitize_id(f"{self.smiles}_x{self.copies}")
            else:
                self.id = sanitize_id("_".join(f"{s}_x{c}" for s, c in self.smiles))


def make_request(
    smiles: str | list[str] | list[tuple[str, int]],
    *,
    id: str | None = None,
    copies: int | list[int] = 4,
    samples: int = 1,
    batch_size: int | None = None,
) -> SampleRequest:
    """Normalize SMILES input into a SampleRequest.

    Accepts a single SMILES string (dot-separated for co-crystals), a list of
    SMILES strings, or a list of `(smiles, copies)` pairs.
    """
    if isinstance(smiles, str):
        parts = smiles.split(".")
    elif smiles and all(isinstance(item, (list, tuple)) for item in smiles):
        parts = [str(part) for part, _ in smiles]
        copies = [int(part_copies) for _, part_copies in smiles]
    else:
        parts = [str(part) for part in smiles]
    if not parts:
        raise ValueError("At least one SMILES component is required.")
    if isinstance(copies, int):
        copies = [copies] * len(parts)
    if len(copies) != len(parts):
        raise ValueError(f"Got {len(parts)} SMILES components but {len(copies)} copies values.")
    if len(parts) == 1:
        return SampleRequest(
            smiles=parts[0], id=id, copies=int(copies[0]), samples=samples, batch_size=batch_size
        )
    return SampleRequest(
        smiles=list(zip(parts, map(int, copies))),
        id=id,
        copies=1,
        samples=samples,
        batch_size=batch_size,
    )


def resolve_checkpoint(checkpoint: str | Path) -> str:
    """Resolve a hub model name/alias to a downloaded checkpoint, or pass through a local path."""
    key = str(checkpoint).strip().lower()
    if key in HUB_MODELS:
        repo_id, filename = HUB_MODELS[key]
        return hf_hub_download(repo_id=repo_id, filename=filename)
    if not Path(checkpoint).is_file():
        raise ValueError(
            f"Unknown model {checkpoint!r}: not a hub model ({sorted(HUB_MODELS)}) "
            "or an existing checkpoint file."
        )
    return str(checkpoint)


def resolve_predictions_path(input_path: str | Path) -> Path:
    from clari.paths import resolve_results_path

    input_path = resolve_results_path(input_path)
    if input_path.is_dir():
        input_path = input_path / "predictions.parquet"
    if not input_path.is_file():
        raise FileNotFoundError(f"Predictions parquet does not exist: {input_path}")
    return input_path


def request_components(request: SampleRequest) -> list[tuple[str, int]]:
    if isinstance(request.smiles, str):
        return [(request.smiles, request.copies)]
    return [(str(smiles), int(copies)) for smiles, copies in request.smiles]


def parse_cli_request(
    pos_args: list[str],
    smiles_flags: list[str] | None,
    copies_flags: list[str] | list[int] | None,
    request_id: str | None,
    samples: int,
) -> list[SampleRequest]:
    """Build one request from `SMILES [copies] [SMILES [copies]]...` or --smiles/--copies flags.

    Dots always split into components; a copies value broadcasts over the dot
    components of its token; omitted copies default to 4.
    """
    if pos_args and smiles_flags:
        raise ValueError("Use either positional SMILES or `--smiles`, not both.")
    tokens: list[tuple[str, int | None]] = []
    if pos_args:
        if copies_flags:
            raise ValueError(
                "`--copies` only combines with `--smiles`; put counts after each SMILES instead."
            )
        idx = 0
        while idx < len(pos_args):
            smiles = pos_args[idx]
            if smiles.isdigit():
                raise ValueError(
                    f"Unexpected copy count {smiles!r} without a preceding SMILES string."
                )
            copies = None
            if idx + 1 < len(pos_args) and pos_args[idx + 1].isdigit():
                copies = int(pos_args[idx + 1])
                idx += 2
            else:
                idx += 1
            tokens.append((smiles, copies))
    elif smiles_flags:
        copies_values = [int(value) for value in copies_flags or []]
        if len(smiles_flags) == 1 and len(copies_values) > 1:
            # Distribute repeated --copies over the dot components of a single --smiles.
            return [
                make_request(smiles_flags[0], id=request_id, copies=copies_values, samples=samples)
            ]
        if len(copies_values) > len(smiles_flags):
            raise ValueError("Received more `--copies` values than `--smiles` values.")
        tokens = [
            (smiles, copies_values[index] if index < len(copies_values) else None)
            for index, smiles in enumerate(smiles_flags)
        ]
    else:
        raise ValueError("Provide SMILES input or a `--config` file.")
    parts = [
        (component, 4 if copies is None else copies)
        for smiles, copies in tokens
        for component in smiles.split(".")
    ]
    return [make_request(parts, id=request_id, samples=samples)]


KNOWN_REQUEST_KEYS = {"id", "smiles", "copies", "samples", "batch_size"}


def _warn(message: str) -> None:
    import sys

    print(f"Warning: {message}", file=sys.stderr)


def parse_config_requests(config_path: str | Path) -> tuple[list[SampleRequest], dict[str, object]]:
    config = json.loads(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a JSON object.")
    items = config.get("requests")
    if not isinstance(items, list) or not items:
        raise ValueError("Config file must define a non-empty `requests` list.")
    requests = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Each request must be an object, got {type(item)!r}")
        unknown = sorted(set(item) - KNOWN_REQUEST_KEYS)
        if unknown:
            _warn(
                f"ignoring unknown request key(s) {unknown}; "
                f"known keys are {sorted(KNOWN_REQUEST_KEYS)}"
            )
        smiles = item.get("smiles")
        valid = isinstance(smiles, str) or (
            isinstance(smiles, list)
            and smiles
            and all(isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in smiles)
        )
        if not valid:
            raise ValueError(
                "Each request `smiles` value must be a string or a list of `[smiles, copies]` pairs."
            )
        requests.append(
            make_request(
                smiles,
                id=item.get("id"),
                copies=int(item.get("copies", 4)),
                samples=int(item.get("samples", 1)),
                batch_size=int(item["batch_size"]) if item.get("batch_size") is not None else None,
            )
        )
    return requests, {key: value for key, value in config.items() if key != "requests"}
