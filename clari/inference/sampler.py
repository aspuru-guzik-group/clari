from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal

import torch
from huggingface_hub import hf_hub_download
from rdkit import Chem
from tqdm.auto import tqdm

from clari.chem import Crystal
from clari.inference.io import run_config, write_predictions
from clari.pipelines.base.lit import LitDiT

H100_REFERENCE_MEMORY_GB = 81.0
DEFAULT_MAX_RESAMPLE_FACTOR = 10

HubModel = Literal["Clari-M", "Clari-L", "Clari-H", "clari-m", "clari-l", "clari-h"]
SmilesInput = str | tuple[str, int] | list[tuple[str, int]] | tuple[tuple[str, int], ...]
RDMolInput = (
    Chem.Mol | tuple[Chem.Mol, int] | list[tuple[Chem.Mol, int]] | tuple[tuple[Chem.Mol, int], ...]
)
_HUB_MODELS: dict[str, tuple[str, str]] = {
    "clari-m": ("the-matter-lab/clari", "clari-med.ckpt"),
    "clari-l": ("the-matter-lab/clari", "clari-large.ckpt"),
    "clari-h": ("the-matter-lab/clari", "clari-huge.ckpt"),
}


def resolve_hub_checkpoint(model: HubModel) -> str:
    model_key = str(model).strip().lower()
    if model_key in ("clari-huge", "clari-h"):
        model_key = "clari-h"
    elif model_key in ("clari-large", "clari-l"):
        model_key = "clari-l"
    elif model_key in ("clari-med", "clari-m"):
        model_key = "clari-m"

    if model_key not in _HUB_MODELS:
        raise ValueError(f"Unknown model {model!r}. Available: {sorted(_HUB_MODELS)}")
    repo_id, filename = _HUB_MODELS[model_key]

    from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
    cached_path = try_to_load_from_cache(repo_id=repo_id, filename=filename)
    if cached_path is None or cached_path is _CACHED_NO_EXIST:
        print(f"Resolving model '{model_key}' from HF Hub ({repo_id}/{filename})...")
        print("Downloading model checkpoint...")

    return hf_hub_download(repo_id=repo_id, filename=filename)


def _make_clash_check():
    from clari.pipelines.utils.metrics import check_clashes_eval
    return lambda crystal: bool(check_clashes_eval(crystal))


@dataclasses.dataclass
class SampleRequest:

    id: str
    crystal: Crystal | None = None
    smiles: SmilesInput | None = None
    rdmol: RDMolInput | None = None
    copies: int = 4
    n_samples: int = 1
    metadata: dict[str, Any] | None = None


def make_requests(
    items,
    *,
    ids: str | list[str] | None = None,
    n_samples: int | list[int] = 1,
    copies: int | list[int] = 4,
) -> list[SampleRequest]:
    """Build a list of SampleRequests from heterogeneous inputs.

    ``items`` is a SampleRequest, Crystal, RDKit Mol, SMILES string, or a list/tuple of any
    mixture of those. ``ids``/``n_samples``/``copies`` may be scalars (broadcast) or lists
    (must match the length of items). Pre-built SampleRequests pass through unchanged.
    """
    items_list = _items_list(items)
    n = len(items_list)
    n_samples_list = _broadcast(n_samples, n, "n_samples")
    copies_list = _broadcast(copies, n, "copies")
    ids_list = _resolve_ids(ids, items_list)
    return [
        _to_request(item, id=id_, n_samples=ns, copies=cps)
        for item, id_, ns, cps in zip(
            items_list, ids_list, n_samples_list, copies_list, strict=True
        )
    ]


def _to_request(item, *, id: str, n_samples: int, copies: int) -> SampleRequest:
    if isinstance(item, SampleRequest):
        return item
    if isinstance(item, Crystal):
        return SampleRequest(id=id, crystal=item, n_samples=n_samples)
    if isinstance(item, str) or _is_smiles_pair(item) or _is_smiles_pair_sequence(item):
        return SampleRequest(id=id, smiles=item, copies=copies, n_samples=n_samples)
    if isinstance(item, Chem.Mol) or _is_rdmol_pair(item) or _is_rdmol_pair_sequence(item):
        return SampleRequest(id=id, rdmol=item, copies=copies, n_samples=n_samples)
    raise TypeError(f"Unsupported inference input type: {type(item)!r}")


def _items_list(items) -> list:
    if isinstance(items, (list, tuple)) and not (
        _is_smiles_pair(items)
        or _is_smiles_pair_sequence(items)
        or _is_rdmol_pair(items)
        or _is_rdmol_pair_sequence(items)
    ):
        return list(items)
    return [items]


def crystal_from_request(request: SampleRequest) -> Crystal:
    if request.n_samples <= 0:
        raise ValueError(f"n_samples must be positive for {request.id!r}, got {request.n_samples}")
    if request.crystal is not None:
        return request.crystal
    if request.smiles is not None:
        return Crystal.from_smiles(
            _smiles_with_copies(request.smiles, request.copies),
            csd_id=request.id,
        )
    if request.rdmol is not None:
        return Crystal.from_rdmol(
            _rdmols_with_copies(request.rdmol, request.copies),
            csd_id=request.id,
        )
    raise ValueError(f"SampleRequest {request.id!r} must provide crystal, smiles, or rdmol")


def _smiles_with_copies(smiles: SmilesInput, copies: int) -> list[tuple[str, int]]:
    if isinstance(smiles, str):
        return [(smiles, copies)]
    if _is_smiles_pair(smiles):
        return [(smiles[0], smiles[1])]
    return [(smile, copy_count) for smile, copy_count in smiles]


def _rdmols_with_copies(rdmol: RDMolInput, copies: int) -> list[tuple[Chem.Mol, int]]:
    if isinstance(rdmol, Chem.Mol):
        return [(rdmol, copies)]
    if _is_rdmol_pair(rdmol):
        return [(rdmol[0], rdmol[1])]
    return [(mol, copy_count) for mol, copy_count in rdmol]


def _is_smiles_pair(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], str)
        and type(value[1]) is int
    )


def _is_rdmol_pair(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], Chem.Mol)
        and type(value[1]) is int
    )


def _is_smiles_pair_sequence(value) -> bool:
    return (
        isinstance(value, (list, tuple)) and bool(value) and all(_is_smiles_pair(v) for v in value)
    )


def _is_rdmol_pair_sequence(value) -> bool:
    return (
        isinstance(value, (list, tuple)) and bool(value) and all(_is_rdmol_pair(v) for v in value)
    )


def _resolve_ids(ids, items: list) -> list[str]:
    if ids is None:
        return [_default_id(item, idx) for idx, item in enumerate(items)]
    ids_list = [ids] if isinstance(ids, str) else list(ids)
    if len(ids_list) != len(items):
        raise ValueError(f"Expected {len(items)} ids, got {len(ids_list)}")
    return ids_list


def _default_id(item, idx: int) -> str:
    if isinstance(item, SampleRequest):
        return item.id
    if isinstance(item, Crystal):
        return str(item.csd_id)
    return f"input_{idx}"


def _broadcast(value: int | list[int], n: int, name: str) -> list[int]:
    if isinstance(value, int):
        values = [value] * n
    else:
        values = list(value)
    if len(values) != n:
        raise ValueError(f"Expected {n} {name} values, got {len(values)}")
    if any(v <= 0 for v in values):
        raise ValueError(f"All {name} values must be positive: {values}")
    return values


@dataclasses.dataclass
class CrystalSample:

    id: str
    sample_idx: int
    crystal: Crystal
    trajectory: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def to_row(self) -> dict[str, Any]:
        row = {
            "sample_idx": self.sample_idx,
            "id": self.id,
            "cif": self.crystal.to_cif(),
        }
        if self.metadata:
            collisions = row.keys() & self.metadata.keys()
            if collisions:
                raise ValueError(
                    f"SampleRequest metadata reuses reserved row keys: {sorted(collisions)}"
                )
            row.update(self.metadata)
        return row

    def to_ase(self):
        return self.crystal.to_ase()


@dataclasses.dataclass(frozen=True)
class SampleChunk:

    request_idx: int
    id: str
    sample_idx_start: int
    count: int
    shard_idx: int


def chunk_for(n_atoms: int) -> int:
    if n_atoms > 1000:
        return 1
    if n_atoms < 200:
        return 1000
    if n_atoms < 300:
        return 500
    if n_atoms < 500:
        return 200
    return 25


def resolve_batch_size(
    n_atoms: int,
    batch_size: int | None,
    *,
    device: str | torch.device | None = None,
) -> int:
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return batch_size
    return max(1, int(chunk_for(n_atoms) * cuda_memory_scale(device)))


def cuda_memory_scale(device: str | torch.device | None) -> float:
    if not torch.cuda.is_available():
        return 1.0
    resolved = resolve_device(device)
    if resolved.type != "cuda":
        return 1.0
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    total_gb = torch.cuda.get_device_properties(index).total_memory / 1e9
    return min(1.0, max(1.0 / H100_REFERENCE_MEMORY_GB, total_gb / H100_REFERENCE_MEMORY_GB))


def replicate(batch: Crystal, n: int) -> Crystal:
    crystals = batch.unbatch()
    return type(batch).collate([crystal for crystal in crystals for _ in range(n)])


def request_sample_offsets(requests: list[SampleRequest]) -> list[int]:
    offsets = []
    next_sample_idx = 0
    for request in requests:
        offsets.append(next_sample_idx)
        next_sample_idx += request.n_samples
    return offsets


def make_sample_chunks(
    requests: list[SampleRequest],
    *,
    batch_size: int | None = None,
    device: str | torch.device | None = None,
) -> list[SampleChunk]:
    chunks: list[SampleChunk] = []
    for request_idx, (request, sample_idx_start) in enumerate(
        zip(requests, request_sample_offsets(requests), strict=True)
    ):
        crystal = crystal_from_request(request)
        chunk_size = resolve_batch_size(int(crystal.num_atoms), batch_size, device=device)
        for local_start in range(0, request.n_samples, chunk_size):
            count = min(chunk_size, request.n_samples - local_start)
            chunks.append(
                SampleChunk(
                    request_idx=request_idx,
                    id=request.id,
                    sample_idx_start=sample_idx_start + local_start,
                    count=count,
                    shard_idx=len(chunks),
                )
            )
    return chunks


def split_chunk_trajectories(
    traj_cpu: torch.Tensor | None, n_kept: int
) -> list[torch.Tensor | None]:
    if traj_cpu is None or n_kept == 0:
        return [None] * n_kept
    _num_steps_plus_one, total_atoms, *rest = traj_cpu.shape
    if total_atoms % n_kept:
        raise ValueError(f"Trajectory atom count {total_atoms} is not divisible by {n_kept}")
    per_sample_atoms = total_atoms // n_kept
    reshaped = traj_cpu.reshape(traj_cpu.shape[0], n_kept, per_sample_atoms, *rest)
    if per_sample_atoms == 1:
        return [reshaped[:, sample_idx, 0].clone() for sample_idx in range(n_kept)]
    return [reshaped[:, sample_idx].clone() for sample_idx in range(n_kept)]


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            res = torch.device("cuda")
        elif torch.backends.mps.is_available():
            res = torch.device("mps")
        else:
            res = torch.device("cpu")
    else:
        res = torch.device(device)
    return res


def set_torch_threads(n: int) -> None:
    if n <= 0:
        raise ValueError(f"torch_threads must be positive, got {n}")
    torch.set_num_threads(n)


def load_lit(
    checkpoint_path: str | Path,
    *,
    use_ema: bool,
    device: torch.device,
    n_steps: int | None,
    compile: bool,
) -> LitDiT:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lit = LitDiT(**ckpt["hyper_parameters"])
    if use_ema:
        weights = (ckpt.get("ema") or {}).get("ema_weights") if isinstance(ckpt, dict) else None
        lit.load_state_dict(
            weights if isinstance(weights, dict) else ckpt["state_dict"], strict=False
        )
    else:
        lit.load_state_dict(ckpt["state_dict"])
    if n_steps is not None:
        lit.sampler.num_steps = int(n_steps)
    lit = lit.to(device).eval()
    if compile:
        if device.type != "cuda":
            raise ValueError("compile=True is only supported on CUDA devices")
        import torch._inductor.config as cfg

        cfg.online_softmax = False
        lit.configure_model()
    return lit


def make_progress(pbar: bool | str, *, total: int, desc: str, unit: str):
    if not pbar:
        return None
    if isinstance(pbar, str):
        desc = pbar
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=True)


def pbar_desc(pbar: bool | str) -> str | None:
    if not pbar:
        return None
    if isinstance(pbar, str):
        return pbar
    return "Denoising"


class ClariSampler:

    def __init__(
        self,
        lit: LitDiT,
        *,
        device: str | torch.device | None = None,
        use_bf16: bool = True,
        compile: bool = False,
        checkpoint_path: str | Path | None = None,
        use_ema: bool = True,
        n_steps: int | None = None,
        torch_threads: int = 1,
        num_gpus: int = 1,
    ):
        self.device = resolve_device(device)
        self.lit = lit.to(self.device).eval()
        self.use_bf16 = use_bf16
        self.compile = compile
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self.use_ema = use_ema
        self.n_steps = n_steps
        self.torch_threads = torch_threads
        self.num_gpus = num_gpus

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        use_ema: bool = True,
        device: str | torch.device | None = "auto",
        n_steps: int | None = None,
        use_bf16: bool = True,
        compile: bool | None = None,
        torch_threads: int = 1,
        num_gpus: int = 1,
    ) -> ClariSampler:
        if n_steps is not None and n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        set_torch_threads(torch_threads)
        resolved_device = resolve_device(device)
        compile = resolved_device.type == "cuda" if compile is None else compile
        torch.set_float32_matmul_precision("high")
        lit = load_lit(
            checkpoint_path,
            use_ema=use_ema,
            device=resolved_device,
            n_steps=n_steps,
            compile=compile,
        )
        return cls(
            lit,
            device=resolved_device,
            use_bf16=use_bf16,
            compile=compile,
            checkpoint_path=checkpoint_path,
            use_ema=use_ema,
            n_steps=n_steps,
            torch_threads=torch_threads,
            num_gpus=num_gpus,
        )

    @classmethod
    def from_hub(
        cls,
        model: HubModel,
        *,
        use_ema: bool = True,
        device: str | torch.device | None = "auto",
        n_steps: int | None = None,
        use_bf16: bool = True,
        compile: bool | None = None,
        torch_threads: int = 1,
        num_gpus: int = 1,
    ) -> ClariSampler:
        """Load a CLARI model from the Hugging Face Hub.

        Downloads to the default HF cache (~/.cache/huggingface/) on first call;
        subsequent calls reuse the cached file without re-downloading.
        """
        checkpoint_path = resolve_hub_checkpoint(model)
        return cls.from_checkpoint(
            checkpoint_path,
            use_ema=use_ema,
            device=device,
            n_steps=n_steps,
            use_bf16=use_bf16,
            compile=compile,
            torch_threads=torch_threads,
            num_gpus=num_gpus,
        )

    @torch.no_grad()
    def sample(
        self,
        inputs,
        *,
        n_samples: int = 1,
        copies: int = 4,
        batch_size: int | None = None,
        return_trajectory: bool = False,
        output_dir: str | Path | None = None,
        overwrite: bool = False,
        num_gpus: int | None = None,
        keep_shards: bool = False,
        filter_clashing: bool = True,
        max_resample_factor: int = DEFAULT_MAX_RESAMPLE_FACTOR,
        pbar: bool | str = True,
        step_pbar: bool | str = False,
    ) -> list[CrystalSample] | Path:
        requests = make_requests(inputs, n_samples=n_samples, copies=copies)
        effective_num_gpus = self.num_gpus if num_gpus is None else num_gpus
        if output_dir is not None:
            from clari.inference.runner import sample_to_directory

            return sample_to_directory(
                checkpoint_path=self.checkpoint_path,
                requests=requests,
                output_dir=output_dir,
                sampler=self if effective_num_gpus == 1 else None,
                batch_size=batch_size,
                num_gpus=effective_num_gpus,
                device=str(self.device),
                use_ema=self.use_ema,
                use_bf16=self.use_bf16,
                n_steps=self.n_steps,
                compile=self.compile,
                torch_threads=self.torch_threads,
                overwrite=overwrite,
                keep_shards=keep_shards,
                filter_clashing=filter_clashing,
                max_resample_factor=max_resample_factor,
                pbar=pbar,
                step_pbar=step_pbar,
            )
        if effective_num_gpus > 1:
            raise ValueError(
                "output_dir is required when num_gpus > 1; multi-GPU sampling writes shards to disk"
            )
        return self._sample_requests(
            requests,
            batch_size=batch_size,
            return_trajectory=return_trajectory,
            filter_clashing=filter_clashing,
            max_resample_factor=max_resample_factor,
            pbar=pbar,
            step_pbar=step_pbar,
        )

    def _sample_requests(
        self,
        requests: list[SampleRequest],
        *,
        batch_size: int | None,
        return_trajectory: bool,
        filter_clashing: bool,
        max_resample_factor: int,
        pbar: bool | str,
        step_pbar: bool | str,
    ) -> list[CrystalSample]:
        samples: list[CrystalSample] = []
        next_sample_idx = 0
        progress = make_progress(
            pbar,
            total=sum(request.n_samples for request in requests),
            desc="Sampling crystals",
            unit="sample",
        )
        for request in requests:
            samples.extend(
                self.sample_crystal(
                    crystal_from_request(request),
                    n_samples=request.n_samples,
                    batch_size=batch_size,
                    sample_idx_start=next_sample_idx,
                    id=request.id,
                    metadata=request.metadata,
                    return_trajectory=return_trajectory,
                    filter_clashing=filter_clashing,
                    max_resample_factor=max_resample_factor,
                    pbar=False,
                    step_pbar=step_pbar,
                    _progress=progress,
                )
            )
            next_sample_idx += request.n_samples
        if progress is not None:
            progress.close()
        return samples

    @torch.no_grad()
    def sample_crystal(
        self,
        crystal: Crystal,
        *,
        n_samples: int,
        batch_size: int | None = None,
        sample_idx_start: int = 0,
        id: str | None = None,
        metadata: dict | None = None,
        return_trajectory: bool = False,
        filter_clashing: bool = True,
        max_resample_factor: int = DEFAULT_MAX_RESAMPLE_FACTOR,
        pbar: bool | str = True,
        step_pbar: bool | str = False,
        _progress=None,
    ) -> list[CrystalSample]:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if filter_clashing and max_resample_factor <= 0:
            raise ValueError(
                f"max_resample_factor must be positive when filter_clashing=True, "
                f"got {max_resample_factor}"
            )
        if crystal.batched:
            raise ValueError("sample_crystal expects an unbatched Crystal")

        batch = Crystal.collate([crystal])
        active = min(
            n_samples,
            resolve_batch_size(int(crystal.num_atoms), batch_size, device=self.device),
        )
        progress = _progress or make_progress(
            pbar,
            total=n_samples,
            desc=f"Sampling {id or crystal.csd_id}",
            unit="sample",
        )

        is_clashing = _make_clash_check() if filter_clashing else None
        attempted = 0
        samples: list[CrystalSample] = []
        while len(samples) < n_samples:
            # Without a filter, ask for exactly what's missing. With a filter we expect
            # some samples to be dropped, so use the full active batch width.
            target = active if filter_clashing else min(active, n_samples - len(samples))
            chunk, trajs = self._sample_chunk(
                batch,
                target,
                return_trajectory=return_trajectory,
                step_pbar=step_pbar,
            )
            active = min(active, len(chunk))
            attempted += len(chunk)
            for pred_crystal, trajectory in zip(chunk, trajs, strict=True):
                if len(samples) >= n_samples:
                    break
                if is_clashing is not None and is_clashing(pred_crystal):
                    continue
                samples.append(
                    CrystalSample(
                        id=id or str(crystal.csd_id),
                        sample_idx=sample_idx_start + len(samples),
                        crystal=pred_crystal,
                        trajectory=trajectory,
                        metadata=metadata,
                    )
                )
                if progress is not None:
                    progress.update(1)
            if (
                filter_clashing
                and attempted >= max_resample_factor * n_samples
                and len(samples) < n_samples
            ):
                raise RuntimeError(
                    f"Produced only {len(samples)}/{n_samples} non-clashing samples for "
                    f"{id or crystal.csd_id} after {attempted} attempts. Increase "
                    f"max_resample_factor or disable filter_clashing."
                )
        if progress is not None and _progress is None:
            progress.close()
        return samples

    def _sample_chunk(
        self,
        batch: Crystal,
        n: int,
        *,
        return_trajectory: bool = False,
        step_pbar: bool | str = False,
    ) -> tuple[list[Crystal], list[torch.Tensor | None]]:
        while True:
            batch_gpu = None
            try:
                batch_gpu = replicate(batch.to(self.device), n)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16 and self.device.type == "cuda",
                ):
                    out = self.lit.sampler.sample(
                        self.lit.interface,
                        self.lit.net,
                        batch_gpu,
                        pbar=pbar_desc(step_pbar),
                        return_trajectory=return_trajectory,
                    )
                if return_trajectory:
                    traj_cpu = out.cpu()
                    pred_gpu = batch_gpu.replace(x=out[-1])
                else:
                    traj_cpu = None
                    pred_gpu = out
                pred = pred_gpu.cpu()
                del batch_gpu
                crystals = pred.unbatch()
                return crystals, split_chunk_trajectories(traj_cpu, len(crystals))
            except RuntimeError as exc:
                is_oom = self.device.type == "cuda" and "out of memory" in str(exc).lower()
                if not is_oom or n <= 1:
                    raise
                if batch_gpu is not None:
                    del batch_gpu
                torch.cuda.empty_cache()
                n = max(1, n // 2)

    def write_output(
        self,
        samples: list[CrystalSample],
        *,
        output_dir: str | Path,
        requests: list[SampleRequest] | None = None,
        overwrite: bool = False,
    ) -> None:
        write_predictions(
            samples,
            output_dir=output_dir,
            overwrite=overwrite,
            config=run_config(
                checkpoint_path=self.checkpoint_path,
                requests=requests,
                num_gpus=self.num_gpus,
                device=str(self.device),
                use_ema=self.use_ema,
                use_bf16=self.use_bf16,
                n_steps=self.n_steps,
                compile=self.compile,
                torch_threads=self.torch_threads,
                num_samples=len(samples),
            ),
        )
