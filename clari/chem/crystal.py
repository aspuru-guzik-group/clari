import dataclasses
import itertools
from collections import defaultdict
from functools import cached_property

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import einops
import gemmi
import numpy as np
import torch
import torch.linalg as LA
import torch_geometric as pyg
from gemmi import UnitCell, cif
from pymatgen.core import Species, Structure
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from clari import geometry as geom
from clari.chem import BOND_TO_INDEX, INDEX_TO_BOND, silenced_rdlogger
from clari.chem.common import PTABLE
from clari.chem.featurize import featurize

# https://github.com/project-gemmi/gemmi/issues/380
gemmi.set_leak_warnings(False)


@dataclasses.dataclass(frozen=True)
class Crystal:
    x: Tensor  # (* 3+N 3) (0.5*abc coords)

    # Sequence
    atom_nums: Tensor  # (* N)
    atom_charges: Tensor  # (* N)
    atom_feats: Tensor  # (* N D)
    body_ids: Tensor  # (* N)
    asu_ids: Tensor  # (* N)

    # Pair
    # Encodes both covalent bonds and topological distances over the ASU molecular graph:
    #   B[i,j] > 0: directly bonded, value is bond type index (1-5)
    #   B[i,j] = 0: same atom (i == j)
    #   B[i,j] < 0: non-bonded, value is -L where L is shortest-path hop count (2-16)
    #   B[i,j] = -16: graph distance > 16
    #   B[i,j] = -17: no path (disconnected)
    bonds: Tensor  # (* N N)

    # Global
    csd_id: str | tuple[str]
    mask: Tensor | None = None  # (... N) if batched

    def asdict(self):  # shallow copy, unlike dataclasses.asdict
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    @classmethod
    def collate(cls, batch: list[Self]) -> Self:
        device = batch[0].x.device
        Ls = torch.tensor([C.num_atoms for C in batch], device=device).int()  # (B)
        mask = mask_from_sizes(Ls)

        kwargs = dict()
        for field in dataclasses.fields(Crystal):
            k = field.name
            v = [getattr(C, k) for C in batch]
            if k == "bonds":
                Amax = max(A.shape[-1] for A in v)
                kwargs[k] = torch.zeros([len(batch), Amax, Amax], dtype=torch.int, device=device)
                for i, A in enumerate(v):
                    n = A.shape[0]
                    kwargs[k][i, :n, :n] = A
            elif k == "csd_id":
                kwargs[k] = tuple(v)
            elif k == "mask":
                continue
            else:
                vpad = -1 if k.endswith("_ids") else 0
                kwargs[k] = pad_sequence(v, batch_first=True, padding_value=vpad)
        return Crystal(**kwargs, mask=mask)

    def unbatch(self) -> list[Self]:
        assert self.batched

        items = []
        for i, n in enumerate(self.num_atoms.tolist()):
            kwargs = dict()
            for k, v in self.asdict().items():
                if k == "x":
                    kwargs[k] = v[i, : (3 + n)]
                elif k == "csd_id":
                    kwargs[k] = v[i]
                elif k == "mask":
                    kwargs[k] = None
                elif k == "bonds":
                    kwargs[k] = v[i, :n, :n]
                else:
                    kwargs[k] = v[i, :n]
            items.append(Crystal(**kwargs))
        return items

    def __repr__(self) -> str:
        kwargs = []
        for k, v in self.asdict().items():
            if isinstance(v, Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    v = list(v.shape)
            elif isinstance(v, (tuple, list)):
                v = [len(v)]
            kwargs.append(f"{k}={v}")
        return f"Crystal({', '.join(kwargs)})"

    def to(self, device: torch.device | str) -> Self:
        kwargs = dict()
        for k, v in self.asdict().items():
            if isinstance(v, Tensor):
                v = v.to(device)
            kwargs[k] = v
        return Crystal(**kwargs)

    def cpu(self) -> Self:
        return self.to(torch.device("cpu"))

    def replace(self, **kwargs) -> Self:
        return dataclasses.replace(self, **kwargs)

    def subset(self, indices) -> Self:
        assert self.batched
        kwargs = {k: v[indices] for k, v in dataclasses.asdict(self).items()}
        return Crystal(**kwargs)

    @property
    def device(self) -> torch.device:
        return self.x.device

    @property
    def batched(self) -> bool:
        return self.mask is not None

    @property
    def batch_size(self) -> int | None:
        return self.mask.shape[0] if self.batched else None

    @property
    def num_atoms(self) -> int | Tensor:
        return self.mask.int().sum(dim=-1) if self.batched else len(self.atom_nums)

    COORD_NORM = 8.0

    @classmethod
    def pack_to_x(cls, lattice: Tensor, coords: Tensor, mask: Tensor | None = None) -> Tensor:
        coords = geom.zero_com(coords, w=mask)
        return torch.cat([0.5 * lattice, coords], dim=-2) / cls.COORD_NORM

    def update_x(
        self,
        lattice: Tensor | None = None,
        coords: Tensor | None = None,
    ) -> Self:
        lattice = self.lattice if (lattice is None) else lattice
        coords = self.coords if (coords is None) else coords
        x = self.pack_to_x(lattice=lattice, coords=coords, mask=self.mask)
        return self.replace(x=x)

    @property
    def lattice(self) -> Tensor:
        return 2 * self.COORD_NORM * self.x[..., :3, :]

    @property
    def coords(self) -> Tensor:
        return self.COORD_NORM * self.x[..., 3:, :]

    @property
    def frac_coords(self) -> Tensor:
        return self.cob_fractional(self.coords)

    def cob_euclidean(self, fcoords: Tensor) -> Tensor:
        return einops.einsum(self.lattice, fcoords, "... j i, ... n j -> ... n i")

    def cob_fractional(self, coords: Tensor) -> Tensor:
        return einops.einsum(LA.inv(self.lattice), coords, "... j i, ... n j -> ... n i")

    @property
    def num_bodies(self) -> Tensor:
        return sizes_from_ids(self.body_ids, mask=self.mask)

    @cached_property
    def body_num_atoms(self) -> Tensor:
        assert not self.batched
        body_ids, body_sizes = torch.unique_consecutive(self.body_ids, return_counts=True)
        assert sorted(body_ids.tolist()) == list(range(self.num_bodies))
        return body_sizes

    @cached_property
    def body_quotients(self) -> list[list[int]]:
        assert not self.batched
        groups = defaultdict(list)
        body_sizes = self.body_num_atoms.tolist()
        for i, a in enumerate(self.asu_ids.split(body_sizes)):
            groups[a.min().item()].append(i)
        return list(groups.values())

    def wrapped(
        self,
        mode: str,
        bounds: tuple[float, float] = (-0.5, 0.5),
        offset: float | Tensor = 0.0,
    ) -> Self:
        f = self.frac_coords + offset
        if mode == "none":
            pass
        elif mode == "all":
            f = softwrap(f, bounds=bounds)
        elif mode == "com":
            assert not self.batched
            fcoms = pyg.utils.scatter(f, self.body_ids, reduce="mean")
            shift = softwrap(fcoms, bounds=bounds, margin=1e-5) - fcoms
            f = f + shift[self.body_ids]
        else:
            raise ValueError()
        return self.update_x(coords=self.cob_euclidean(f))

    def augment(self) -> Self:
        assert not self.batched

        # Wrap with random offset
        offset = torch.rand(3, device=self.device)
        C = self.wrapped(mode="com", offset=offset)

        # Random lattice row permutation and per-row sign flips
        perm = torch.randperm(3, device=self.device)
        signs = torch.randint(0, 2, (3, 1), device=self.device) * 2 - 1
        C = C.update_x(lattice=(signs * C.lattice[perm]))

        # Random SO(3) rotation
        Q, _ = torch.linalg.qr(torch.randn(3, 3, device=self.device))
        if torch.linalg.det(Q) < 0:
            Q[0] *= -1
        return C.replace(x=einops.einsum(C.x, Q, "n i, i j -> n j"))

    def aligned_lattice(self, other: Self) -> Self:
        assert not self.batched
        assert self.csd_id == other.csd_id

        # Build all 12 candidates: 6 permutations x 2 global signs
        candidates = list(itertools.product(itertools.permutations(range(3)), (1, -1)))
        Ls = torch.stack([s * self.lattice[list(p)] for p, s in candidates])  # (12 3 3)

        # Optimal alignment
        R, _ = geom.kabsch_align(Ls, other.lattice, sym="so3", return_rot=True)  # (12 3 3)
        mse = torch.square(Ls @ R.mT - other.lattice).mean(dim=[1, 2])  # (12)
        best = mse.argmin().item()

        C = self.update_x(lattice=Ls[best])
        return C.replace(x=(C.x @ R[best].mT))

    def aligned_perm(self, other: Self) -> Self:
        assert not self.batched
        assert self.csd_id == other.csd_id

        # Permute and sign-align lattice rows
        L = torch.stack([self.lattice, -self.lattice], dim=0)  # (2 S 3)
        d = torch.cdist(other.lattice, L).square()  # (2 O S)
        cost = d.amin(dim=0).cpu().numpy()  # (O S)
        sign = d.argmin(dim=0)  # (O S)
        I, J = linear_sum_assignment(cost)  # I is sorted
        L = torch.stack([L[sign[i, j], j] for i, j in zip(I, J, strict=False)], dim=0)

        # Permute bodies (coords unchanged by lattice permutation)
        body_sizes = self.body_num_atoms.tolist()
        mobile = self.coords.split(body_sizes, dim=0)
        target = other.coords.split(body_sizes, dim=0)

        cost = np.full([len(mobile)] * 2, np.inf)
        for Q in self.body_quotients:
            for i, j in itertools.product(Q, repeat=2):
                cost[i, j] = torch.square(target[i] - mobile[j]).sum()
        I, J = linear_sum_assignment(cost)  # I is sorted
        X = torch.cat([mobile[j] for j in J], dim=0)

        return self.update_x(lattice=L, coords=X)

    def aligned_pose(self, other: Self, on: str = "x") -> Self:
        if on == "x":
            mobile = self.x
            target = other.x
        elif on == "lattice":
            mobile = self.lattice
            target = other.lattice
        elif on == "cell":
            mobile = torch.cat([self.lattice, self.coords], dim=0)
            target = torch.cat([other.lattice, other.coords], dim=0)
        else:
            raise ValueError()
        if on in {"x", "cell"}:
            w = torch.full([3 + self.num_atoms], 1 / self.num_atoms).to(self.x)
            w[:3] = 1 / 3
            w = w / w.sum()
        else:
            w = None

        R, _ = geom.kabsch_align(mobile, target, w=w, sym="so3", return_rot=True)
        return self.replace(x=(self.x @ R.mT))

    def aligned(self, other: Self, on: str = "x", niters: int = 2) -> Self:
        C = self.aligned_lattice(other)
        for _ in range(niters):
            C = C.aligned_perm(other)
            C = C.aligned_pose(other, on=on)
        return C

    def without_Hs(self) -> Self:
        assert not self.batched
        keep = self.atom_nums != 1
        _, body_ids = torch.unique(self.body_ids[keep], return_inverse=True)
        _, asu_ids = torch.unique(self.asu_ids[keep], return_inverse=True)
        return self.replace(
            x=Crystal.pack_to_x(self.lattice, self.coords[keep]),
            atom_nums=self.atom_nums[keep],
            atom_charges=self.atom_charges[keep],
            atom_feats=None,  # for safety
            body_ids=body_ids,
            asu_ids=asu_ids,
            bonds=None,
        )

    def show(self, lattice: bool = True, wrap: str = "none", **kwargs):
        assert not self.batched
        from clari.chem.draw import draw_crystal
        C = self.wrapped(mode=wrap, bounds=(-0.5, 0.5))
        return draw_crystal(
            lattice=(C.lattice.numpy(force=True) if lattice else None),
            coords=C.coords.numpy(force=True),
            atoms=C.atom_nums.numpy(force=True),
            **kwargs,
        )

    def to_pymatgen(self) -> Structure:
        assert not self.batched
        return Structure(
            lattice=self.lattice.numpy(force=True),
            coords=self.coords.numpy(force=True),
            species=[Species(PTABLE.GetElementSymbol(z.item())) for z in self.atom_nums],
            coords_are_cartesian=True,
            properties={"id": self.csd_id},
        )

    def to_ase(self):
        assert not self.batched
        from ase import Atoms

        return Atoms(
            numbers=self.atom_nums.detach().cpu().numpy(),
            positions=self.coords.detach().cpu().numpy(),
            cell=self.lattice.detach().cpu().numpy(),
            pbc=True,
        )

    def to_cif(self, as_string=True) -> cif.Document | str:
        assert not self.batched
        lattice = self.lattice.detach().cpu().numpy()
        a, b, c = map(np.linalg.norm, lattice)
        alpha = np.degrees(np.arccos(np.dot(lattice[1], lattice[2]) / (b * c)))
        beta = np.degrees(np.arccos(np.dot(lattice[0], lattice[2]) / (a * c)))
        gamma = np.degrees(np.arccos(np.dot(lattice[0], lattice[1]) / (a * b)))

        st = gemmi.SmallStructure()
        st.cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)
        st.spacegroup_hm = "P 1"

        frac_coords = self.frac_coords.detach().cpu().numpy()
        atom_nums = self.atom_nums.detach().cpu().numpy()
        label_counts = defaultdict(int)
        for z, (x, y, z_c) in zip(atom_nums, frac_coords, strict=False):
            symbol = PTABLE.GetElementSymbol(int(z))
            label_counts[symbol] += 1
            site = gemmi.SmallStructure.Site()
            site.label = f"{symbol}{label_counts[symbol]}"
            site.fract = gemmi.Fractional(float(x), float(y), float(z_c))
            site.element = gemmi.Element(int(z))
            site.occ = 1.0
            st.add_site(site)

        block = st.make_cif_block()
        block.name = self.csd_id
        return block.as_string() if as_string else block

    @classmethod
    def from_cif(cls, cif_str: str) -> Self:
        b = cif.read_string(cif_str)[0]
        a = float(b.find_value("_cell_length_a"))
        b_len = float(b.find_value("_cell_length_b"))
        c = float(b.find_value("_cell_length_c"))
        alpha = float(b.find_value("_cell_angle_alpha"))
        beta = float(b.find_value("_cell_angle_beta"))
        gamma = float(b.find_value("_cell_angle_gamma"))
        unit_cell = UnitCell(a, b_len, c, alpha, beta, gamma)
        lattice = torch.tensor(unit_cell.orth.mat.array.T, dtype=torch.float32)
        atom_symbols = b.find_loop("_atom_site_type_symbol")
        frac_x = [float(x) for x in b.find_loop("_atom_site_fract_x")]
        frac_y = [float(y) for y in b.find_loop("_atom_site_fract_y")]
        frac_z = [float(z) for z in b.find_loop("_atom_site_fract_z")]
        frac_coords = torch.tensor([frac_x, frac_y, frac_z], dtype=torch.float32).T
        coords = geom.zero_com(torch.einsum("nj,ji->ni", frac_coords, lattice))
        atom_nums = torch.tensor(
            [PTABLE.GetAtomicNumber(sym.strip()) for sym in atom_symbols],
            dtype=torch.long,
        )
        num_atoms = len(atom_nums)
        atom_charges = torch.zeros(num_atoms, dtype=torch.long)
        atom_feats = torch.zeros((num_atoms, 0), dtype=torch.float32)
        body_ids = torch.zeros(num_atoms, dtype=torch.long)
        asu_ids = torch.zeros(num_atoms, dtype=torch.long)
        bonds = torch.zeros((num_atoms, num_atoms), dtype=torch.long)
        csd_id = b.name if b.name else "unknown"
        x = cls.pack_to_x(lattice, coords)
        return cls(
            x=x,
            atom_nums=atom_nums,
            atom_charges=atom_charges,
            atom_feats=atom_feats,
            body_ids=body_ids,
            asu_ids=asu_ids,
            bonds=bonds,
            csd_id=csd_id,
        )

    @classmethod
    def from_smiles(
        cls,
        smiles: list[tuple[str, int]],
        *,
        csd_id: str = "smiles",
        add_hs: bool | list[bool] = False,
    ) -> Self:
        from pathlib import Path

        with silenced_rdlogger():
            mols = []
            for s, c in smiles:
                if isinstance(s, str) and (
                    s.lower().endswith(".mol") or (len(s) < 1024 and Path(s).is_file())
                ):
                    mol = Chem.MolFromMolFile(s, sanitize=False)
                elif isinstance(s, str) and ("M  END" in s or "V2000" in s or "V3000" in s):
                    mol = Chem.MolFromMolBlock(s, sanitize=False)
                else:
                    mol = Chem.MolFromSmiles(s, sanitize=False)
                mols.append((mol, c))
        if any(m is None for m, _ in mols):
            raise ValueError(f"Could not parse SMILES or read .mol file: {smiles!r}")
        add_hs_list = [add_hs] * len(mols) if isinstance(add_hs, bool) else list(add_hs)
        for flag, (m, _) in zip(add_hs_list, mols):
            if flag:
                m.UpdatePropertyCache(strict=False)
        mols = [(Chem.AddHs(m) if flag else m, c) for flag, (m, c) in zip(add_hs_list, mols)]
        return cls.from_rdmol(mols, csd_id=csd_id)

    @classmethod
    def from_rdmol(cls, mols: list[tuple[Chem.Mol, int]], *, csd_id: str = "rdmol") -> Self:
        if not mols or any(c <= 0 for _, c in mols):
            raise ValueError("mols must be non-empty with positive copy counts")
        if any(m.GetNumAtoms() == 0 for m, _ in mols):
            raise ValueError("RDKit molecule has no atoms")
        if any(len(Chem.GetMolFrags(m)) > 1 for m, _ in mols):
            raise ValueError("Each molecule must be a single connected component")

        # Build the ASU by concatenating all distinct molecules.
        atom_nums, atom_charges, src, dst, bond_types, spans = [], [], [], [], [], []
        for mol, _ in mols:
            start = len(atom_nums)
            atom_nums += [a.GetAtomicNum() for a in mol.GetAtoms()]
            atom_charges += [a.GetFormalCharge() for a in mol.GetAtoms()]
            for b in mol.GetBonds():
                src.append(start + b.GetBeginAtomIdx())
                dst.append(start + b.GetEndAtomIdx())
                bond_types.append(BOND_TO_INDEX[int(b.GetBondType())])
            spans.append((start, len(atom_nums)))

        asu_atom_nums = torch.tensor(atom_nums, dtype=torch.long)
        asu_atom_charges = torch.tensor(atom_charges, dtype=torch.long)
        asu_feats, asu_bonds = featurize(
            pyg.data.Data(
                atom_nums=asu_atom_nums,
                atom_charges=asu_atom_charges,
                edge_index=torch.tensor([src, dst], dtype=torch.long).reshape(2, -1),
                edge_attr=torch.tensor(bond_types, dtype=torch.int).reshape(-1, 1),
                num_asu=len(atom_nums),
            )
        )

        # Replicate copy-major, matching the co-crystal ordering used by the data module.
        asu_ids, body_ids, body_id = [], [], 0
        max_copies = max(n_copies for _, n_copies in mols)
        for copy_idx in range(max_copies):
            for (start, stop), (_, n_copies) in zip(spans, mols, strict=True):
                if copy_idx >= n_copies:
                    continue
                idx = torch.arange(start, stop, dtype=torch.long)
                asu_ids.append(idx)
                body_ids.append(torch.full((stop - start,), body_id, dtype=torch.long))
                body_id += 1
        asu_ids = torch.cat(asu_ids)
        body_ids = torch.cat(body_ids)

        bonds = asu_bonds[asu_ids][:, asu_ids].long()
        bonds = torch.where(body_ids.unsqueeze(-1) == body_ids, bonds, -17)

        num_atoms = asu_ids.numel()
        lattice = torch.eye(3, dtype=torch.float32) * max(10.0, float(num_atoms))
        return cls(
            x=cls.pack_to_x(lattice, torch.zeros((num_atoms, 3), dtype=torch.float32)),
            atom_nums=asu_atom_nums[asu_ids],
            atom_charges=asu_atom_charges[asu_ids],
            atom_feats=asu_feats[asu_ids],
            body_ids=body_ids,
            asu_ids=asu_ids,
            bonds=bonds,
            csd_id=csd_id,
        )

    @classmethod
    def ase_from_cif(cls, cif_str: str):
        from ase import Atoms

        b = cif.read_string(cif_str)[0]
        a = float(b.find_value("_cell_length_a"))
        b_len = float(b.find_value("_cell_length_b"))
        c = float(b.find_value("_cell_length_c"))
        alpha = float(b.find_value("_cell_angle_alpha"))
        beta = float(b.find_value("_cell_angle_beta"))
        gamma = float(b.find_value("_cell_angle_gamma"))
        unit_cell = UnitCell(a, b_len, c, alpha, beta, gamma)
        lattice = np.asarray(unit_cell.orth.mat.array.T, dtype=np.float32)

        atom_symbols = b.find_loop("_atom_site_type_symbol")
        frac_x = np.asarray([float(x) for x in b.find_loop("_atom_site_fract_x")], dtype=np.float32)
        frac_y = np.asarray([float(y) for y in b.find_loop("_atom_site_fract_y")], dtype=np.float32)
        frac_z = np.asarray([float(z) for z in b.find_loop("_atom_site_fract_z")], dtype=np.float32)
        frac_coords = np.stack([frac_x, frac_y, frac_z], axis=1)
        atom_nums = np.asarray(
            [PTABLE.GetAtomicNumber(sym.strip()) for sym in atom_symbols],
            dtype=np.int64,
        )

        return Atoms(
            numbers=atom_nums,
            scaled_positions=frac_coords,
            cell=lattice,
            pbc=True,
        )

    def to_rdmol(self) -> Chem.Mol:
        assert not self.batched

        mol = Chem.RWMol()
        for i in range(self.num_atoms):
            a = Chem.Atom(self.atom_nums[i].item())
            a.SetFormalCharge(self.atom_charges[i].item())
            mol.AddAtom(a)
        for u, v in zip(*torch.nonzero(self.bonds > 0, as_tuple=True), strict=False):
            u, v = u.item(), v.item()
            if u < v:
                btype = INDEX_TO_BOND[self.bonds[u, v].item()]
                mol.AddBond(u, v, btype)

        coords = self.coords
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (coords[i, 0].item(), coords[i, 1].item(), coords[i, 2].item()))
        mol.AddConformer(conf)

        return mol.GetMol()


def sizes_from_ids(ids: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return ids.max().item() + 1
    assert mask.ndim == ids.ndim == 2
    ids = torch.where(mask, ids, -100)
    return ids.amax(dim=-1) + 1


def mask_from_sizes(sizes: Tensor) -> Tensor:
    max_size = sizes.max().item()
    mask = torch.arange(max_size, device=sizes.device) < sizes.unsqueeze(-1)
    return mask


def softwrap(
    x: Tensor,
    bounds: tuple[float, float] = (0, 1),
    margin: float | None = None,
) -> Tensor:
    a, b = bounds
    w = ((x - a) % (b - a)) + a
    if margin is not None:
        w = torch.where((a - margin < x) & (x < b + margin), x, w)  # soft wrap
    return w
