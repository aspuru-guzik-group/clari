import itertools
from contextlib import redirect_stderr
from io import StringIO

import amd
import numpy as np
import torch
import torch.linalg as LA
from pymatgen.core import Lattice
from rdkit import Chem
from rdkit.Chem.MolStandardize.rdMolStandardize import DisconnectOrganometallicsInPlace
from yaml import safe_load

from clari import geometry as geom
from clari.chem import Crystal, element_radii, silenced_rdlogger
from clari.paths import ASSETS_DIR
from clari.pipelines.utils.utils import prefix_keys


def check_clashes(C: Crystal) -> dict:
    assert not C.batched

    not_same_body = C.body_ids.unsqueeze(-2) != C.body_ids.unsqueeze(-1)
    radii_vdw = torch.tensor([element_radii(z, "vdw") for z in C.atom_nums.tolist()]).to(C.x)
    radii_cov = torch.tensor([element_radii(z, "cov") for z in C.atom_nums.tolist()]).to(C.x)
    sum_radii_vdw = radii_vdw.unsqueeze(-1) + radii_vdw
    sum_radii_cov = radii_cov.unsqueeze(-1) + radii_cov

    L = Lattice(C.lattice.numpy(force=True))  # pymatgen more reliable
    f = C.frac_coords.numpy(force=True)
    pair_dists = torch.from_numpy(L.get_all_distances(f, f)).to(C.device).float()
    pair_dists.fill_diagonal_(torch.inf)
    inter_dists = torch.where(not_same_body, pair_dists, torch.inf)

    cf_0p5 = torch.all(inter_dists >= 0.5).item()
    cf_vdw = torch.all(inter_dists >= 0.5 * sum_radii_vdw).item()
    cr_oxt = 1 - torch.all(inter_dists >= sum_radii_vdw - 0.7).item()
    cr_pfw = (pair_dists < 0.75 * sum_radii_cov).any(dim=-1).float().mean().item()

    return {
        "clash_free_0.5": cf_0p5,
        "clash_free_vdw": cf_vdw,
        "clash_rate_oxt": cr_oxt,
        "clash_rate_pfw": cr_pfw,
    }


def check_clashes_eval(C: Crystal) -> float:
    assert not C.batched

    not_same_body = C.body_ids.unsqueeze(-2) != C.body_ids.unsqueeze(-1)
    radii_cov = torch.tensor([element_radii(z, "cov") for z in C.atom_nums.tolist()]).to(C.x)
    lb = radii_cov.unsqueeze(-1) + radii_cov

    L = Lattice(C.lattice.numpy(force=True))  # pymatgen more reliable
    f = C.frac_coords.numpy(force=True)
    pair_dists = torch.from_numpy(L.get_all_distances(f, f)).to(C.device).float()
    inter_dists = torch.where(not_same_body, pair_dists, torch.inf)

    return float((inter_dists < lb).any().item())


def is_clash_free(C: Crystal) -> bool:
    """Clash-free by BOTH the vdW and covalent criteria (used to filter sampled outputs)."""
    return bool(check_clashes(C)["clash_free_vdw"]) and not check_clashes_eval(C)


def volume_error(pred: Crystal, true: Crystal):
    v1 = LA.det(pred.lattice).abs().item()
    v2 = LA.det(true.lattice).abs().item()
    return abs(v1 - v2) / v2


def density_error(pred: Crystal, true: Crystal):
    v1 = LA.det(pred.lattice).abs().item()
    v2 = LA.det(true.lattice).abs().item()
    return abs(v1 - v2) / v1


def rmsd_lattice(pred: Crystal, true: Crystal):
    L1, L2 = pred.lattice, true.lattice
    perms = torch.tensor(list(itertools.permutations(range(3)))).to(L1.device).int()
    L1 = L1[perms, :]  # (6 3 3)
    L1 = geom.kabsch_align(L1, L2, sym="o3")
    return geom.rmsd(L1, L2).min().item()


def pdd(C: Crystal, k=100, canonical=True):
    pset = amd.periodicset_from_pymatgen_structure(C.to_pymatgen())
    return amd.PDD(pset, k=k, lexsort=canonical, collapse=canonical)


def dist_amd_and_pdd(pred: Crystal, true: Crystal, metric="chebyshev", k=100):
    pdd_pred = pdd(pred, k=k)
    pdd_true = pdd(true, k=k)
    pdd_dist = amd.EMD(pdd_pred, pdd_true, metric=metric)
    amd_pred = amd.PDD_to_AMD(pdd_pred)
    amd_true = amd.PDD_to_AMD(pdd_true)
    amd_dist = amd.AMD_cdist([amd_pred], [amd_true], metric=metric).item()
    return {"dist_amd": amd_dist, "dist_pdd": pdd_dist}


def pdist_hist(C: Crystal, r_max: float, n_bins) -> np.ndarray:
    L = Lattice(C.lattice.numpy(force=True))
    f = C.frac_coords.numpy(force=True)
    dists = L.get_all_distances(f, f)
    np.fill_diagonal(dists, np.inf)
    dists = dists[dists <= r_max]
    counts, _ = np.histogram(dists, bins=n_bins, range=(0.0, r_max))
    return counts / counts.sum()


def dist_rdf(pred: Crystal, true: Crystal) -> dict:
    r_max = 10.0
    r_short = 5.0
    n_bins = 100
    dr = r_max / n_bins

    p = pdist_hist(pred, r_max, n_bins)
    q = pdist_hist(true, r_max, n_bins)
    wass = dr * np.sum(np.abs(np.cumsum(p) - np.cumsum(q))).item()

    short_bins = round(r_short / dr)
    ps, qs = p[:short_bins], q[:short_bins]
    ps = ps / ps.sum()
    qs = qs / qs.sum()
    wass_short = dr * np.sum(np.abs(np.cumsum(ps) - np.cumsum(qs))).item()

    return {"dist_rdf": wass, "dist_rdf_short": wass_short}


POSE_BUSTERS_CONFIG = safe_load((ASSETS_DIR / "posebusters_no_strain.yml").open())
POSE_BUSTER_ATOMS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def posebusters_score(C: Crystal) -> dict:
    from posebusters import PoseBusters

    mol = C.to_rdmol()

    DisconnectOrganometallicsInPlace(mol)
    with silenced_rdlogger():
        valid = []
        for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
            if (
                (frag.GetNumAtoms() <= 1)
                or any(a.GetSymbol() not in POSE_BUSTER_ATOMS for a in frag.GetAtoms())
                or (Chem.SanitizeMol(frag, catchErrors=True) != 0)
                or any(a.GetTotalNumHs() > 0 for a in frag.GetAtoms())
            ):
                continue
            valid.append(frag)
        if not valid:
            return {"pb_score": 1.0, "pb_valid": 0.0}
        with StringIO() as buf:
            with redirect_stderr(buf):  # hack to silence annoying UFFTYPER logs
                results = PoseBusters(config=POSE_BUSTERS_CONFIG).bust(mol_pred=valid)
        passes = sum(1 for _, row in results.iterrows() if not row.isin([False]).any())

    return {"pb_score": passes / len(valid), "pb_valid": 1.0}


def assess_crystals_train(pred: Crystal, true: Crystal):
    pred = pred.wrapped(mode="com", bounds=(-0.5, 0.5))
    true = true.aligned(pred, on="x")

    pred_noH = pred.without_Hs()
    true_noH = true.without_Hs()

    return {
        "rmsd_lattice": rmsd_lattice(pred, true),
        "volume_error": volume_error(pred, true),
        "density_error": density_error(pred, true),
        **prefix_keys("heavy_", check_clashes(pred_noH)),
        **prefix_keys("heavy_", dist_amd_and_pdd(pred_noH, true_noH)),
        **prefix_keys("heavy_", dist_rdf(pred_noH, true_noH)),
    }


def assess_crystals_eval(pred: Crystal, true: Crystal, amd_metric="chebyshev"):
    pred = pred.wrapped(mode="com", bounds=(-0.5, 0.5))
    true = true.aligned(pred, on="x")

    return {
        "volume_error": volume_error(pred, true),
        "clash_rate": check_clashes_eval(pred),
        **dist_amd_and_pdd(pred, true, metric=amd_metric),
        **posebusters_score(pred),
    }
