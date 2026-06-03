import einops
import numpy as np
import py3Dmol

from clari.chem.common import xyzfile

UNIT_CUBE_EDGES = np.asarray(
    [
        [[0, 0, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 1, 0]],
        [[1, 0, 0], [1, 0, 1]],
        [[0, 1, 0], [1, 1, 0]],
        [[0, 1, 0], [0, 1, 1]],
        [[0, 0, 1], [1, 0, 1]],
        [[0, 0, 1], [0, 1, 1]],
        [[1, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [1, 0, 1]],
        [[1, 1, 1], [1, 1, 0]],
    ]
)  # (12, 2, 3)


def format_point(p):
    if isinstance(p, np.ndarray):
        p = p.tolist()
    return dict(zip("xyz", p, strict=False))


def draw_arrow(view, start, end, viewer=None, **kwargs):
    opts = {"start": format_point(start), "end": format_point(end), **kwargs}
    view.addArrow(opts, viewer=viewer)


def draw_line(view, start, end, viewer=None, **kwargs):
    opts = {"start": format_point(start), "end": format_point(end), **kwargs}
    view.addCylinder(opts, viewer=viewer)


def draw_box(view, lattice, viewer=None, **kwargs):
    for e in einops.einsum(UNIT_CUBE_EDGES - 0.5, lattice, "b e d, d i -> b e i"):
        draw_line(view, e[0], e[1], viewer=viewer, **kwargs)


def draw_overlay(mol1, mol2, view=None, colors=("blue", "red")):
    if view is None:
        view = py3Dmol.view(width=400, height=400)
    for mol, color in zip((mol1, mol2), colors, strict=False):
        atoms = [a.atomic_number for a in mol.atoms]
        coords = np.array([[a.coordinates.x, a.coordinates.y, a.coordinates.z] for a in mol.atoms])
        draw_crystal(atoms, coords, view=view, color=color)
    return view


def draw_crystal(
    atoms,
    coords,
    lattice=None,
    view=None,
    viewer=None,
    color=None,
    opacity=None,
):
    if py3Dmol is None:
        raise ValueError("install py3Dmol")
    if view is None:
        view = py3Dmol.view(width=400, height=400) if (view is None) else view
    view.addModel(xyzfile(atoms, coords), "xyz", viewer=viewer)

    if lattice is not None:
        assert tuple(lattice.shape) == (3, 3)
        o = -0.5 * np.sum(lattice, axis=0)
        for p, c in zip(lattice + o, ["red", "green", "blue"], strict=False):
            draw_arrow(view, o, p, viewer=viewer, radius=0.2, radiusRatio=2, mid=0.92, color=c)
        draw_box(view, lattice, viewer=viewer, radius=0.05)
        view.addUnitCell(viewer=viewer)

    if not isinstance(color, list):
        color = [color] * len(atoms)
    if not isinstance(opacity, list):
        opacity = [opacity] * len(atoms)
    for i, (c, o) in enumerate(zip(color, opacity, strict=False)):
        style_opts = {k: v for k, v in {"color": c, "opacity": o}.items() if v is not None}
        view.setStyle(
            {"model": -1, "serial": i},
            {
                "stick": {"radius": 0.2, **style_opts},
                "sphere": {"scale": 0.2, **style_opts},
            },
            viewer=viewer,
        )
    view.zoomTo()

    return view


def draw_crystal_trajectory(
    traj,
    lattice: bool = True,
    view=None,
    viewer=None,
    duration_play=20,
    duration_stop=10,
):
    if view is None:
        view = py3Dmol.view(width=400, height=400)

    def frame_xyz(C):
        atoms = C.atom_nums.numpy(force=True)
        coords = C.coords.numpy(force=True)
        if not lattice:
            return xyzfile(atoms, coords)
        # Embed unit cell edges as He dummy atoms so they animate per-frame.
        L = C.lattice.numpy(force=True)
        o = -0.5 * L.sum(axis=0)
        corners = np.array(
            [[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=float
        )
        edges = [
            (i, j)
            for i in range(8)
            for j in range(i + 1, 8)
            if np.abs(corners[i] - corners[j]).sum() == 1
        ]
        t = np.linspace(0, 1, 15)
        box_pts = np.array(
            [o + (corners[i] + ti * (corners[j] - corners[i])) @ L for i, j in edges for ti in t]
        )
        return xyzfile(
            np.concatenate([atoms, np.full(len(box_pts), 2)]),  # He = 2
            np.concatenate([coords, box_pts]),
        )

    interval = duration_play * 1000 / len(traj)
    hold_last = round(duration_stop / (interval / 1000))

    trajfile = ""
    for C in traj:
        trajfile += frame_xyz(C) + "\n"
    for _ in range(hold_last):
        trajfile += frame_xyz(traj[-1]) + "\n"

    view.addModelsAsFrames(trajfile, "xyz", viewer=viewer)
    view.setStyle(
        {"not": {"elem": "He"}}, {"stick": {"radius": 0.2}, "sphere": {"scale": 0.2}}, viewer=viewer
    )
    if lattice:
        view.setStyle({"elem": "He"}, {"sphere": {"radius": 0.08, "color": "gray"}}, viewer=viewer)
    view.animate({"interval": interval})
    view.zoomTo()
    return view
