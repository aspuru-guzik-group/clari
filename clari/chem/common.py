import contextlib
import pathlib

import polars as pl
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import GetPeriodicTable

from clari.paths import ASSETS_DIR

PTABLE = GetPeriodicTable()


# 0 means no bonds!
INDEX_TO_BOND = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
    5: Chem.BondType.UNSPECIFIED,
}

BOND_TO_INDEX = [5] * 22
for i, btype in INDEX_TO_BOND.items():
    BOND_TO_INDEX[int(btype)] = i


@contextlib.contextmanager
def silenced_rdlogger():
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(RDLogger.INFO)


def xyzfile(atoms, coords):
    file = [f"{len(atoms)}\n"]
    for a, p in zip(atoms, coords, strict=False):
        symbol = PTABLE.GetElementSymbol(int(a))
        x, y, z = p.tolist()
        file.append(f"{symbol} {x:f} {y:f} {z:f}")
    return "\n".join(file)


# Downloaded from: https://www.ccdc.cam.ac.uk/media/Elemental_Radii_Alvarez.xlsx
def load_radii():
    radii = {"vdw": torch.full([118], 2.0), "cov": torch.full([118], 1.5)}
    df = pl.read_csv(pathlib.Path(ASSETS_DIR / "Elemental_Radii_Alvarez.csv"))
    for row in df.iter_rows(named=True):
        radii["vdw"][row["Atomic Number"]] = row["vdW Radius"]
        radii["cov"][row["Atomic Number"]] = row["Covalent Radius"]
    return radii


RADII_CACHE = load_radii()


def element_radii(z, rtype):
    radii = RADII_CACHE[rtype]
    if isinstance(z, int):
        r = radii[z].item()
    else:
        r = radii.to(z)[z]
    return r


def distance_lbound(z1, z2, bond_mask):
    nonmetals = torch.tensor([1, 2, 6, 7, 8, 9, 10, 15, 16, 17, 18, 34, 35, 36, 53, 54]).to(z1)
    is_metal1 = ~torch.isin(z1, nonmetals)
    is_metal2 = ~torch.isin(z2, nonmetals)
    metal_mask = is_metal1.unsqueeze(-1) | is_metal2.unsqueeze(-2)

    r1 = element_radii(z1, "cov")
    r2 = element_radii(z2, "cov")
    lb = r1.unsqueeze(-1) + r2.unsqueeze(-2)
    lb = torch.clip(torch.where(bond_mask | metal_mask, 0.6, 1.0) * lb, min=0.5)
    return lb
