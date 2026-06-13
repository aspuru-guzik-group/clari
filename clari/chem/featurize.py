import networkx as nx
import torch
import torch.nn.functional as F
import torch_geometric as pyg

from clari.chem.common import element_radii

ATOM_FEATURES = 4 + 5 + 10 + 5


def featurize(data: pyg.data.Data, ignore_Hs: bool = False):
    src, dst = data.edge_index.unbind(0)
    bonds = data.edge_attr.squeeze(-1)
    node_adj_bonds = torch.zeros([data.num_asu, 5]).float()

    G = nx.Graph()
    G.add_nodes_from(range(data.num_asu))
    for u, v, b in zip(src, dst, bonds):
        if ignore_Hs and (data.atom_nums[u] == 1 or data.atom_nums[v] == 1):
            continue
        G.add_edge(u.item(), v.item())
        node_adj_bonds[u][b - 1] = 1.0
        node_adj_bonds[v][b - 1] = 1.0

    radii_vdw = torch.tensor([element_radii(z, "vdw") for z in data.atom_nums]).float()
    radii_cov = torch.tensor([element_radii(z, "cov") for z in data.atom_nums]).float()
    charges = data.atom_charges.long()
    degrees = torch.tensor([G.degree(u) for u in sorted(G)]).long()

    f = [
        (radii_vdw / 2),
        (radii_cov / 2),
        (charges.float() / 5),
        torch.log(1 + degrees.float()),
    ]
    f = [
        torch.stack(f, dim=-1),
        F.one_hot(charges.clip(min=-2, max=2) + 2, num_classes=5),
        F.one_hot(degrees.clip(max=9), num_classes=10),
        node_adj_bonds,
    ]
    f = torch.cat(f, dim=-1)
    assert f.shape[-1] == ATOM_FEATURES

    B = torch.full([len(G), len(G)], -16).int()
    B.fill_diagonal_(0)
    B[src, dst] = bonds
    B[dst, src] = bonds
    for src, paths in nx.all_pairs_shortest_path_length(G, cutoff=16):
        for dst, L in paths.items():
            if L <= 1:
                continue
            B[src, dst] = -L

    return f, B
