from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable

import lightning as L
import numpy as np
import torch
import torch_geometric as pyg
from torch.utils.data import DataLoader, Dataset

from clari.chem import Crystal, featurize
from clari.csd import AVAILABLE_CSD_SUBSETS, csd_fam
from clari.paths import DATA_DIR

TEACHING_MAX_ATOMS = 512
TEACHING_FAMS = set(AVAILABLE_CSD_SUBSETS["teaching"])
EXCLUDED_CSD_IDS = {"BTCOAC"}  # BTCOAC03 is the usable representative for this family.


class CrystalDataset(Dataset):
    def __init__(
        self,
        crystals,
        slices,
        *,
        group_by_fam: bool = True,
        filter_fams: list[str] | None = None,
        random_repr: bool = True,
        remove_Hs: bool = False,
        augment: bool = True,
        debug: int | str = -1,
    ):
        self.crystals = crystals
        self.slices = slices
        self.random_repr = random_repr
        self.remove_Hs = remove_Hs
        self.augment = augment
        self.debug = debug

        universe = [
            (i, cid)
            for i, cid in enumerate(self.crystals.csd_id)
            if cid not in EXCLUDED_CSD_IDS
            and (
                csd_fam(cid) not in TEACHING_FAMS
                or int(self.crystals.num_atoms[i]) <= TEACHING_MAX_ATOMS
            )
        ]
        if filter_fams is not None:
            filter_fams = {csd_fam(cid) for cid in filter_fams}
            universe = [(i, cid) for i, cid in universe if (csd_fam(cid) in filter_fams)]
        if debug == "mem":
            i = max(universe, key=lambda ic: int(self.crystals.num_atoms[ic[0]]))[0]
            self.classes = [[i]]
        elif group_by_fam:
            fam_groups = defaultdict(list)
            for i, cid in universe:
                fam_groups[csd_fam(cid)].append(i)
            self.classes = [sorted(indices) for _, indices in sorted(fam_groups.items())]
        else:
            self.classes = [[i] for i, _ in universe]

    def __len__(self):
        n = len(self.classes)
        if self.debug == "mem":
            return 10**9
        if isinstance(self.debug, int) and self.debug > 0:
            return min(self.debug, n)
        return n

    def crystal_pygraph(self, i) -> pyg.data.Data:
        return pyg.data.separate.separate(
            cls=pyg.data.Data,
            batch=self.crystals,
            idx=i,
            slice_dict=self.slices,
            decrement=False,
        )

    def __getitem__(self, idx):
        if self.debug == "mem":
            idx = 0
        elif isinstance(self.debug, int) and (self.debug > 0) and (idx >= self.debug):
            raise IndexError(idx)
        if self.random_repr:
            idx = random.choice(self.classes[idx])
        else:
            idx = min(self.classes[idx], key=lambda i: self.crystals.r_factor[i].item())
        G = self.crystal_pygraph(idx)

        if self.remove_Hs:
            heavy_mask = G.atom_nums[G.asu_ids] != 1
            asu_ids = G.asu_ids[heavy_mask].clone()
            bod_ids = torch.unique(G.body_ids[heavy_mask], return_inverse=True)[1]
            frac = G.frac[heavy_mask]
            ignore_Hs = True
        else:
            asu_ids = G.asu_ids
            bod_ids = G.body_ids
            frac = G.frac
            ignore_Hs = False
        asu_feats, asu_bonds = featurize(G, ignore_Hs=ignore_Hs)

        bonds = asu_bonds[asu_ids, :][:, asu_ids]
        bonds = torch.where(bod_ids.unsqueeze(-1) == bod_ids, bonds, -17)

        # Transfer to our internal data structure
        coords = frac @ G.lattice
        C1 = Crystal(
            x=Crystal.pack_to_x(G.lattice, coords),
            atom_nums=G.atom_nums[asu_ids].clone(),
            atom_charges=G.atom_charges[asu_ids].clone(),
            atom_feats=asu_feats[asu_ids].clone(),
            body_ids=bod_ids.long(),
            asu_ids=asu_ids.long(),
            bonds=bonds,
            csd_id=G.csd_id,  # str
        )

        if self.augment:  # generally safer to enable in case dataset is biased
            C1 = C1.augment()
        else:
            C1 = C1.wrapped(mode="com")
        return C1


class RandomSubset(Dataset):
    def __init__(self, dataset, n, generator):
        super().__init__()

        self.dataset = dataset
        self.indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


DEFAULT_SPLIT_OPTS = {
    "train": dict(group_by_fam=True, random_repr=True),
    "val": dict(group_by_fam=True, random_repr=True),
    "predict": dict(group_by_fam=True, random_repr=False, augment=False),
    "test": dict(group_by_fam=False, augment=False),
}


class CrystalDataModule(L.LightningDataModule):
    def __init__(
        self,
        seed: int = 0,
        world_size: int = 1,
        predict_size: int | None = None,  # default; one batch
        batch_size: int = 32,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        split_opts: dict[str, dict[str, bool]] = DEFAULT_SPLIT_OPTS,
        remove_Hs: bool = False,
        debug: int | str = -1,
    ):
        super().__init__()

        self.seed = seed
        self.root = DATA_DIR / "csd"
        self.world_size = world_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        config_path = self.root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Missing CLARI CSD data config: {config_path}. "
                "Paper evaluation commands such as sample-test require a prepared "
                "data/csd directory. Run from the repository root after generating the data, "
                "or set CLARI_DATA_DIR to a directory containing csd/config.json and "
                "{train,val,test}.pt."
            )
        with open(config_path, "r") as f:
            self.artifact = json.load(f)["artifact"]

        datasets = dict()

        for k in DEFAULT_SPLIT_OPTS:
            if k not in split_opts:
                split_opts[k] = dict(DEFAULT_SPLIT_OPTS[k])
        for v in split_opts.values():
            v["remove_Hs"] = remove_Hs
            v["debug"] = debug

        g = torch.Generator()
        g.manual_seed(self.seed)

        # TODO: a bit inefficient to load everything, should move to setup()
        for k in ["train", "val", "test"]:
            pyg_path = self.root / f"{k}.pt"
            if not pyg_path.exists():
                raise ValueError(f"PyG file not found: {pyg_path.resolve()}")
            crystals, slices, _ = pyg.io.fs.torch_load(pyg_path)
            crystals = pyg.data.Data.from_dict(crystals)
            datasets[k] = CrystalDataset(crystals, slices, **split_opts[k])

            # Fixed subset for prediction
            if k == "val":
                D = CrystalDataset(crystals, slices, **split_opts["predict"])
                if predict_size is None:
                    predict_size = batch_size * world_size
                predict_size = min(predict_size, len(D))
                predict_size = (predict_size // world_size) * world_size  # prevents repeats
                datasets[f"predict_{k}"] = RandomSubset(D, predict_size, g)

        self.datasets = datasets

    def train_dataloader(self):
        return self._loader(dataset="train")

    def val_dataloader(self):
        return self._loader(dataset="val")

    def test_dataloader(self):
        return self._loader(dataset="test")

    def _loader(self, dataset, **kwargs):
        training = dataset == "train"
        init_kwargs = dict(
            dataset=self.datasets[dataset],
            batch_size=self.batch_size,
            num_workers=min(self.num_workers, 128 if training else 8),
            shuffle=(training or (dataset == "val")),
            drop_last=training,
            collate_fn=self.collate_fn,
            worker_init_fn=seed_worker,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=(4 if self.num_workers > 0 else None),
        )
        init_kwargs.update(kwargs)
        if init_kwargs["collate_fn"] is None:
            init_kwargs["collate_fn"] = Crystal.collate
        return DataLoader(**init_kwargs)


class CrystalDataModuleForFM(CrystalDataModule):
    def val_dataloader(self):
        return [
            self._loader(dataset="val"),
            self._loader(dataset="predict_val", num_workers=1, shuffle=True, collate_fn=None),
        ]


# Patch: https://github.com/Lightning-AI/pytorch-lightning/issues/20412
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
