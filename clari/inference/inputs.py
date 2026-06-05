from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import hf_hub_download

from clari.paths import resolve_results_path

HubModel = Literal["Clari-M", "Clari-L", "Clari-H", "clari-m", "clari-l", "clari-h"]
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
    add_hs: bool | list[bool] = True
    batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            if isinstance(self.smiles, str):
                self.id = sanitize_id(f"{self.smiles}_x{self.copies}")
            else:
                self.id = sanitize_id("_".join(f"{s}_x{c}" for s, c in self.smiles))


def _smiles_to_components(
    smiles: str | list[str],
    copies: int | list[int],
) -> list[tuple[str, int]]:
    parts = smiles.split(".") if isinstance(smiles, str) else list(smiles)
    if isinstance(copies, int):
        return [(s, copies) for s in parts]
    if len(copies) != len(parts):
        raise ValueError(f"Got {len(parts)} SMILES components but {len(copies)} copies values.")
    return list(zip(parts, copies))


def _make_request(
    smiles: str | list[str],
    *,
    id: str | None,
    copies: int | list[int],
    samples: int,
    add_hs: bool | list[bool],
) -> SampleRequest:
    components = _smiles_to_components(smiles, copies)
    if len(components) == 1:
        s, c = components[0]
        return SampleRequest(smiles=s, id=id, copies=c, samples=samples, add_hs=add_hs)
    return SampleRequest(smiles=components, id=id, copies=1, samples=samples, add_hs=add_hs)


def resolve_hub_checkpoint(model: str) -> str:
    model = str(model).strip().lower()
    model = {"clari-med": "clari-m", "clari-large": "clari-l", "clari-huge": "clari-h"}.get(
        model, model
    )
    if model not in HUB_MODELS:
        raise ValueError(f"Unknown model {model!r}. Available: {sorted(HUB_MODELS)}")
    repo_id, filename = HUB_MODELS[model]
    return hf_hub_download(repo_id=repo_id, filename=filename)


def resolve_predictions_path(input_path: str | Path) -> Path:
    input_path = resolve_results_path(input_path)
    if input_path.is_dir():
        input_path = input_path / "predictions.parquet"
    if not input_path.is_file():
        raise FileNotFoundError(f"Predictions parquet does not exist: {input_path}")
    return input_path


def build_request(
    smiles: str | list[tuple[str, int]],
    *,
    id: str | None = None,
    copies: int = 4,
    samples: int = 1,
    add_hs: bool | list[bool] = True,
) -> SampleRequest:
    if isinstance(smiles, str):
        request_id = id or sanitize_id(f"{smiles}_x{copies}")
        return SampleRequest(
            id=request_id,
            smiles=smiles,
            copies=copies,
            samples=samples,
            add_hs=add_hs,
        )
    if not smiles:
        raise ValueError("Co-crystal requests need at least one `(smiles, copies)` pair.")
    parts = [(str(part), int(part_copies)) for part, part_copies in smiles]
    request_id = id or sanitize_id(
        "_".join(f"{part}_x{part_copies}" for part, part_copies in parts)
    )
    return SampleRequest(
        id=request_id,
        smiles=parts,
        copies=1,
        samples=samples,
        add_hs=add_hs,
    )


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
    no_add_hs_flags: list | None,
) -> list[SampleRequest]:
    parts: list[tuple[str, int]] = []
    idx = 0
    while idx < len(pos_args):
        smiles = pos_args[idx]
        if smiles.isdigit():
            raise ValueError(f"Unexpected copy count {smiles!r} without a preceding SMILES string.")
        copies = 4
        if idx + 1 < len(pos_args) and pos_args[idx + 1].isdigit():
            copies = int(pos_args[idx + 1])
            idx += 2
        else:
            idx += 1
        parts.append((smiles, copies))
    if copies_flags and not smiles_flags:
        raise ValueError("`--copies` requires `--smiles`.")
    if smiles_flags and copies_flags and len(copies_flags) > len(smiles_flags):
        raise ValueError("Received more `--copies` values than `--smiles` values.")
    for index, smiles in enumerate(smiles_flags or []):
        copies = int(copies_flags[index]) if copies_flags and index < len(copies_flags) else 4
        parts.append((smiles, copies))
    if not parts:
        raise ValueError("Provide direct SMILES input or a config file.")
    no_add_hs_count = len(no_add_hs_flags) if no_add_hs_flags else 0
    add_hs_per_component = [i >= no_add_hs_count for i in range(len(parts))]
    if len(parts) == 1:
        smiles, copies = parts[0]
        return [
            build_request(
                smiles,
                id=request_id,
                copies=copies,
                samples=samples,
                add_hs=add_hs_per_component[0],
            )
        ]
    return [build_request(parts, id=request_id, samples=samples, add_hs=add_hs_per_component)]


def parse_config_requests(
    config_path: str | Path, add_hs_default: bool
) -> tuple[list[SampleRequest], dict[str, Any]]:
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
        smiles = item.get("smiles")
        batch_size = int(item["batch_size"]) if item.get("batch_size") is not None else None
        if isinstance(smiles, str):
            req = build_request(
                smiles,
                id=item.get("id"),
                copies=int(item.get("copies", 4)),
                samples=int(item.get("samples", 1)),
                add_hs=bool(item.get("add_hs", add_hs_default)),
            )
            req.batch_size = batch_size
            requests.append(req)
        elif (
            isinstance(smiles, list)
            and smiles
            and all(isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in smiles)
        ):
            req = build_request(
                [(str(part), int(part_copies)) for part, part_copies in smiles],
                id=item.get("id"),
                samples=int(item.get("samples", 1)),
                add_hs=bool(item.get("add_hs", add_hs_default)),
            )
            req.batch_size = batch_size
            requests.append(req)
        else:
            raise ValueError(
                "Each request `smiles` value must be a string or a list of `[smiles, copies]` pairs."
            )
    return requests, {key: value for key, value in config.items() if key != "requests"}
