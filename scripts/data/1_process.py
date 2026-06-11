import json
import os
import pathlib
import random
import traceback
import warnings
from datetime import date
from functools import partial

import multiprocess as mp
import networkx as nx
import numpy as np
import polars as pl
import torch
import torch_geometric as pyg
import tqdm
import wandb
from jsonargparse import auto_cli
from networkx.algorithms import isomorphism as iso
from pymatgen.io.cif import CifParser, str2float
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
from scipy.optimize import linear_sum_assignment

from clari.chem import BOND_TO_INDEX, INDEX_TO_BOND, distance_lbound, silenced_rdlogger
from clari.csd import AVAILABLE_CSD_SUBSETS, csd_fam
from clari.paths import DATA_DIR, LOG_DIR

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import stopit  # noqa: E402  # isort: skip

warnings.filterwarnings("ignore", module="pymatgen")

pl.Config.set_tbl_rows(-1)
pl.Config.set_fmt_str_lengths(100)


class CrystalError(ValueError):

    def __init__(self, msg, info=""):
        super().__init__(msg)
        self.info = info


def pmap(fn, inputs, workers, chunksize, pbar):
    if workers > 0:
        with mp.Pool(workers) as pool:
            out = pool.imap_unordered(fn, inputs, chunksize=chunksize)
            return list(tqdm.tqdm(out, **pbar))
    out = map(fn, inputs)
    return list(tqdm.tqdm(out, **pbar))


def parse_nxgraph(block, mol2):
    atom_names = block["_atom_site_label"]
    frac = [list(map(str2float, block[f"_atom_site_fract_{axis}"])) for axis in "xyz"]
    frac = np.asarray(frac).T
    frac = dict(zip(atom_names, frac, strict=False))
    cif_order = {name: i for i, name in enumerate(atom_names)}

    with silenced_rdlogger():
        mol = Chem.MolFromMol2Block(
            mol2,
            sanitize=False,
            removeHs=False,
            cleanupSubstructures=False,
        )
    if mol is None:
        raise CrystalError("failed to parse mol2")

    graph = nx.Graph()
    for atom in mol.GetAtoms():
        name = atom.GetProp("_TriposAtomName")
        if atom.GetProp("_TriposAtomType") == "Du":
            if (name + "?") in cif_order:
                continue
            raise CrystalError("dummy atom in mol2", info=name)
        if name not in cif_order:
            raise CrystalError("atom in mol2 not in cif", info=name)
        graph.add_node(
            atom.GetIdx(),
            name=name,
            num=atom.GetAtomicNum(),
            charge=int(atom.GetDoubleProp("_TriposPartialCharge")),
            frac=frac[name],
        )
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (u in graph) and (v in graph):
            graph.add_edge(u, v, bond=BOND_TO_INDEX[int(bond.GetBondType())])
    if not graph.edges:
        raise CrystalError("no bonds")

    bodies = []
    for component in nx.connected_components(graph):
        bodies.append(graph.subgraph(component).copy())
    return bodies


node_match = iso.categorical_node_match(["num", "charge"], [None] * 2)
edge_match = iso.categorical_edge_match("bond", None)


def node_data(graph, attr):
    return [graph.nodes[v][attr] for v in sorted(graph)]


def is_isomorphism(mapping, G, H):
    if mapping is None:
        return False
    return (
        (len(mapping) == len(G) == len(H))
        and (G.number_of_edges() == H.number_of_edges())
        and all(node_match(G.nodes[u], H.nodes[mapping[u]]) for u in mapping)
        and all(H.has_edge(mapping[u], mapping[v]) for (u, v) in G.edges)
        and all(edge_match(G.edges[u, v], H.edges[mapping[u], mapping[v]]) for (u, v) in G.edges)
    )


def guess_isomorphism(G, H):
    mapping = {}
    for u in sorted(G):
        for v in sorted(H):
            if (v not in mapping.values()) and node_match(G.nodes[u], H.nodes[v]):
                mapping[u] = v
                break
        else:
            return None
    return mapping


def find_isomorphism(G, H):
    if len(G) != len(H):
        return None

    guess = guess_isomorphism(G, H)
    if is_isomorphism(guess, G, H):
        return guess

    with stopit.ThreadingTimeout(180):
        matcher = iso.GraphMatcher(
            G,
            H,
            node_match=node_match,
            edge_match=edge_match,
        )
        return matcher.mapping if matcher.is_isomorphic() else None
    return None


def build_asu(graphs):
    asu = []
    asu_size = 0

    representatives = []
    body_remap = {}

    for graph in graphs:
        if id(graph) in body_remap:
            continue
        for rep_graph, rep_map in representatives:
            iso_map = find_isomorphism(graph, rep_graph)
            if iso_map is not None:
                body_remap[id(graph)] = {u: rep_map[iso_map[u]] for u in graph}
                break
        else:
            mapping = {v: i + asu_size for i, v in enumerate(sorted(graph))}
            body_remap[id(graph)] = mapping
            representatives.append((graph, mapping))
            asu.append(nx.relabel_nodes(graph, mapping))
            asu_size += len(graph)

    asu = nx.union_all(asu)
    asu_ids = [[body_remap[id(graph)][v] for v in sorted(graph)] for graph in graphs]
    return asu, asu_ids


def featurize_nxgraph(graph):
    assert sorted(graph) == list(range(len(graph)))
    atom_nums = torch.tensor(node_data(graph, "num")).int()
    atom_charges = torch.tensor(node_data(graph, "charge")).int()

    src, dst, bonds = zip(*graph.edges.data("bond"), strict=False)
    edge_index = torch.tensor([src, dst]).int()
    edge_attr = torch.tensor(bonds).unsqueeze(-1).int()

    smiles = []
    with silenced_rdlogger():
        for component in nx.connected_components(graph):
            component = graph.subgraph(component)
            rwmol = Chem.RWMol()
            backmap = dict()
            for v in sorted(component):
                z = graph.nodes[v]["num"]
                if z == 1:
                    continue
                atom = Chem.Atom(z)
                atom.SetFormalCharge(graph.nodes[v]["charge"])
                backmap[v] = rwmol.AddAtom(atom)
            for u, v, bond_idx in component.edges.data("bond"):
                if (u not in backmap) or (v not in backmap):
                    continue
                rwmol.AddBond(backmap[u], backmap[v], INDEX_TO_BOND[bond_idx])
            try:
                Chem.SanitizeMol(rwmol, catchErrors=True)
                smiles.append(Chem.MolToSmiles(rwmol))
            except Exception:
                pass
    smiles = sorted(smiles)

    return dict(
        atom_nums=atom_nums,
        atom_charges=atom_charges,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
    )


def check_clash(G1, G2, lattice, f1, f2, tol):
    intra = G2 is None
    if intra:
        assert f2 is None
        G2, f2 = G1, f1
        bond_mask = torch.from_numpy(nx.to_numpy_array(G1, nodelist=sorted(G1)) > 0)
    else:
        bond_mask = False

    cutoff = distance_lbound(
        z1=torch.tensor(node_data(G1, "num"), dtype=torch.long),
        z2=torch.tensor(node_data(G2, "num"), dtype=torch.long),
        bond_mask=bond_mask,
    ).numpy()

    distances = lattice.get_all_distances(f1, f2)
    if intra:
        np.fill_diagonal(distances, np.inf)
    if np.all(cutoff < distances):
        return True
    if (not intra) and (len(G1) == len(G2)):
        row, col = linear_sum_assignment(distances)
        if np.max(distances[row, col]) <= tol:
            return False

    names1 = node_data(G1, "name")
    names2 = node_data(G2, "name")
    clash_distances = np.where((distances <= cutoff), distances, np.inf)
    i, j = np.unravel_index(np.argmin(clash_distances), clash_distances.shape)
    d = round(distances[i, j].item(), 3)
    if d > tol:
        r = round(cutoff[i, j].item(), 3)
        msg = "clashing atoms"
        info = f"{names1[i]}-{names2[j]} ({d} Å, cutoff: {r} Å)"
    else:
        msg = "overlapping atoms"
        info = f"{names1[i]}-{names2[j]} ({d} Å)"
    msg += " within a body" if intra else " between bodies"
    raise CrystalError(msg, info=info)


def process_example(cif, mol, site_tol=0.01, max_atoms=512, **kwargs):
    try:
        cif = CifParser.from_str(cif)
    except Exception as exc:
        raise CrystalError("failed to parse cif") from exc
    block = next(iter(cif._cif.data.values()))

    check_fields = ["_atom_site_label", "_symmetry_Int_Tables_number"]
    for key in check_fields:
        if key not in block.data:
            raise CrystalError(f"cif missing field {key}")

    lattice = cif.get_lattice(block)
    sym_ops = cif.get_symops(block)

    graphs = []
    frac = []
    bod_ids = []

    bodies = parse_nxgraph(block, mol)

    for graph in bodies:
        asu_frac = np.stack(node_data(graph, "frac"), axis=0)

        for op in sym_ops:
            f = op.operate_multi(asu_frac)

            check_clash(graph, None, lattice, f, None, tol=site_tol)
            if any(
                not check_clash(graph, other_graph, lattice, f, other_frac, tol=site_tol)
                for other_graph, other_frac in zip(graphs, frac, strict=False)
            ):
                continue

            bod_ids.extend([len(graphs)] * len(graph))
            graphs.append(graph)
            frac.append(f)

    if (max_atoms is not None) and (sum(map(len, graphs)) > max_atoms):
        raise CrystalError(f"over {max_atoms} atoms in unit cell")

    asu, asu_ids = build_asu(graphs)
    asu_feats = featurize_nxgraph(asu)

    perms = [np.argsort(body_asu_ids) for body_asu_ids in asu_ids]
    asu_ids = [np.array(body_ids_i)[perm] for body_ids_i, perm in zip(asu_ids, perms, strict=False)]
    frac = [frac_i[perm] for frac_i, perm in zip(frac, perms, strict=False)]
    assert all(np.array_equal(x, np.arange(x[0], x[-1] + 1)) for x in asu_ids)

    frac = torch.from_numpy(np.concatenate(frac)).float()
    asu_ids = torch.from_numpy(np.concatenate(asu_ids)).int()
    bod_ids = torch.tensor(bod_ids).int()

    return pyg.data.Data(
        lattice=torch.tensor(lattice.matrix).float(),
        frac=frac,
        asu_ids=asu_ids,
        body_ids=bod_ids,
        **asu_feats,
        num_nodes=len(asu),
        num_atoms=len(frac),
        num_asu=len(asu),
        num_bodies=len(graphs),
        sg=block["_symmetry_Int_Tables_number"],
        **kwargs,
    )


def is_none_or_gt(x, threshold):
    return (x is None) or (x > threshold)


def check_metadata(row, max_r_factor, max_deposition_date):
    checks = [
        (not row["has_3d_structure"], "no 3d structure"),
        (row["is_powder_study"], "powder study"),
        (row["pressure"] is not None, "non-ambient pressure"),
        (row["is_polymeric"], "polymeric"),
    ]
    if max_r_factor is not None:
        checks.append((is_none_or_gt(row["r_factor"], max_r_factor), "r factor too high"))
    if max_deposition_date is not None:
        checks.append((is_none_or_gt(row["deposition_date"], max_deposition_date), "too recent"))
    for condition, reason in checks:
        if condition:
            raise CrystalError(reason)


def crystal_from_cif_and_mol(row, max_r_factor, max_deposition_date, max_atoms):
    cif, mol2 = row.pop("cif"), row.pop("mol2")
    r_factor = row["r_factor"]
    csd_id = row["id"]

    crystal = None
    try:
        if csd_fam(csd_id) in AVAILABLE_CSD_SUBSETS["test"]:
            max_atoms = None
            max_r_factor = None
            max_deposition_date = None
        check_metadata(row, max_r_factor, max_deposition_date)
        crystal = process_example(
            cif=cif,
            mol=mol2,
            max_atoms=max_atoms,
            r_factor=(100.0 if r_factor is None else r_factor),
            csd_id=csd_id,
        )
        row.update(keep=True, error=None)
    except CrystalError as exc:
        row.update(keep=False, error=dict(reason=str(exc), info=exc.info))
    except Exception as exc:
        print(f"ERROR: {csd_id} {exc}")
        print(traceback.format_exc())
        raise
    return crystal, row


def read_raw_data(metadata_path, cif_mol2_path):
    metadata = pl.read_parquet(metadata_path)
    cif_mol2 = pl.read_parquet(cif_mol2_path)
    return cif_mol2.join(metadata, on="id", how="left")


def split_dataset(metadata, crystals):
    test_fams = set(AVAILABLE_CSD_SUBSETS["test"])
    metadata = metadata.with_columns(
        pl.when(csd_fam(pl.col("id")).is_in(test_fams))
        .then(pl.lit("test"))
        .otherwise(pl.lit("train"))
        .alias("split")
    )

    # Drop train entries whose asu_smiles overlap with any test asu_smiles
    crystal_smiles = {c.csd_id: c.smiles for c in crystals}

    test_components = {}  # smiles -> test crystal id
    for row in metadata.filter((pl.col("split") == "test") & pl.col("keep")).iter_rows(named=True):
        for smi in crystal_smiles[row["id"]]:
            if CalcNumHeavyAtoms(Chem.MolFromSmiles(smi, sanitize=False)) > 7:
                test_components[smi] = row["id"]

    rows = metadata.to_dicts()
    for row in rows:
        if row["split"] != "train" or not row["keep"]:
            continue
        for smi in crystal_smiles[row["id"]]:
            if smi in test_components:
                row["keep"] = False
                row["error"] = {
                    "reason": "contains test component",
                    "info": test_components[smi],
                }
                break
    metadata = pl.DataFrame(rows, schema=metadata.schema)

    # Hold out 1000 random train families as validation
    val_fams = (
        metadata.filter((pl.col("split") == "train") & pl.col("keep"))
        .select(csd_fam(pl.col("id")).unique().sort())["id"]
        .to_list()
    )
    val_fams = set(random.Random(42).sample(val_fams, 1000))
    metadata = metadata.with_columns(
        pl.when((pl.col("split") == "train") & csd_fam(pl.col("id")).is_in(val_fams))
        .then(pl.lit("val"))
        .otherwise(pl.col("split"))
        .alias("split")
    )

    return metadata


def write_dataset(root, config, metadata, crystals, logging):
    root = pathlib.Path(root)
    root.mkdir(exist_ok=True)

    metadata.write_parquet(root / "metadata.parquet")

    # Inspect discarded
    discarded = metadata.filter(~pl.col("keep"))
    print(f"Dropped {len(discarded)} / {len(metadata)} samples")
    print(
        discarded.group_by("split", pl.col("error").struct.field("reason"))
        .agg(pl.len().alias("n"))
        .sort("split", "reason")
    )

    for split in ["train", "val", "test"]:
        split_ids = metadata.filter((pl.col("split") == split) & pl.col("keep"))["id"]
        split_ids = set(split_ids.to_list())
        split_data = [c for c in crystals if c.csd_id in split_ids]
        split_fams = set(csd_fam(crystal.csd_id) for crystal in split_data)
        dataset_path = root / f"{split}.pt"
        if split_data:
            pyg.data.InMemoryDataset.save(split_data, dataset_path)
        print(
            f"Saved {len(split_data)} {split} entries "
            f"across {len(split_fams)} fams "
            f"to {dataset_path}"
        )

    if logging:
        artifact = wandb.Artifact(name="csd", type="dataset")
        artifact.add_reference(uri=f"file://{root}")
        wandb.run.log_artifact(artifact)
        artifact.wait()
        config["artifact"] = artifact.qualified_name

    config_path = root / "config.json"
    config_path.write_text(json.dumps(config, indent=2))


def main(
    in_metadata: str = str(DATA_DIR / "raw" / "csd_metadata.parquet"),
    in_cif_mol2: str = str(DATA_DIR / "raw" / "csd_conquest.parquet"),
    out: str = str(DATA_DIR / "csd"),
    num_workers: int = 64,
    num_threads: int = 2,
    chunksize: int = 1,
    debug: int = -1,
    pbar: bool = True,
    logging: bool = True,
    max_r_factor: float = 10.0,
    max_deposition_date: date = date(2025, 5, 1),
    max_atoms: int = 512,
):
    config = dict(locals())
    config["max_deposition_date"] = str(config["max_deposition_date"])
    if logging:
        wandb.init(project="clari-data", dir=LOG_DIR, config=config)

    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    source = read_raw_data(in_metadata, in_cif_mol2)
    if debug > 0:
        source = source.sample(n=debug, seed=0)
    print(f"Loaded {len(source)} entries")

    process_fn = partial(
        crystal_from_cif_and_mol,
        max_r_factor=max_r_factor,
        max_deposition_date=max_deposition_date,
        max_atoms=max_atoms,
    )

    processed = pmap(
        process_fn,
        inputs=source.iter_rows(named=True),
        workers=num_workers,
        chunksize=chunksize,
        pbar=dict(total=len(source), desc="Processing", disable=(not pbar)),
    )

    metadata, crystals = [], []
    for c, row in processed:
        if c is not None:
            crystals.append(c)
        metadata.append(row)
    crystals = sorted(crystals, key=(lambda c: c.csd_id))

    metadata_schema = source.schema
    metadata_schema.pop("cif")
    metadata_schema.pop("mol2")
    metadata_schema.update(
        dict(
            keep=pl.Boolean,
            error=pl.Struct({"reason": pl.String, "info": pl.String}),
        )
    )
    metadata = pl.from_dicts(metadata, schema=metadata_schema).sort("id")

    # Split dataset
    metadata = split_dataset(metadata, crystals)

    # Save to disk
    write_dataset(
        root=out,
        config=config,
        metadata=metadata,
        crystals=crystals,
        logging=logging,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    auto_cli(main)
