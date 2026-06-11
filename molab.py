# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "clari",
#     "marimo",
#     "py3dmol",
#     "anywidget",
#     "traitlets",
#     "rdkit",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import subprocess, sys

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "clari",
            "py3dmol",
            "anywidget",
            "traitlets",
            "rdkit",
        ],
        check=True,
    )
    deps_ready = True
    return


@app.cell
def _():
    import base64
    import io
    import zipfile

    import anywidget
    import marimo as mo
    import py3Dmol
    import torch
    import traitlets

    from clari.chem.draw import draw_crystal, draw_crystal_trajectory_from_batch
    from clari.inference import ClariSampler
    from clari.inference.sample import sample_trajectory

    return (
        ClariSampler,
        anywidget,
        base64,
        io,
        mo,
        sample_trajectory,
        traitlets,
        zipfile,
    )


@app.cell
def _(mo):
    mo.md("""
    <style>
    :root { --ink: #1d2433; --muted: #657188; --line: #d7dee8; --panel: #f7f9fc; --accent: #1f8a70; }
    .hero {
        padding: 28px 30px 24px; border: 1px solid var(--line); border-radius: 8px;
        background: linear-gradient(135deg, rgba(31,138,112,.12), rgba(214,107,53,.08)), var(--panel);
    }
    .hero h1 { color: var(--ink); font-size: 1.95rem; line-height: 1.12; margin: 0 0 10px; }
    .hero p { color: var(--muted); font-size: 1rem; line-height: 1.55; margin: 0; max-width: 720px; }
    .authors { color: var(--muted); font-size: .95rem; margin: 10px 0 4px; }
    .authors sup, .contrib sup {
        font-size: 0.85em;
        position: relative;
        top: -0.15em;
        vertical-align: baseline;
    }
    .contrib { color: var(--muted); font-size: 0.8rem; margin: 0 0 14px; opacity: 0.85; }
    .badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 16px; align-items: center; }
    .badges img { display: block; height: 20px; margin: 0; }
    .dl { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; color: var(--ink); display: inline-block; width: auto; font-size: .9rem; font-weight: 500; line-height: 1; padding: 8px 12px; text-decoration: none; }
    .dl:hover { background: #eef2f8; }
    .step { color: var(--ink); font-weight: 700; font-size: 1.05rem; margin: 4px 0; }
    .molcard { border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; background: #fff; }
    .smiles { font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted); background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px; padding: 4px 8px; word-break: break-all; max-height: 46px; overflow-y: auto; }
    .candi { border: 1px solid var(--line); border-radius: 10px; padding: 10px; text-align: center; background: #fff; }
    </style>
    <section class="hero">
        <h1>Fast Organic Crystal Structure Prediction with Unit Cell Flow Matching</h1>
        <div class="authors">Alston Lo<sup>&ast;</sup>, Luka Mucko<sup>&ast;</sup>, Austin H. Cheng<sup>&ast;</sup>, Andy Cai, Alastair J. A. Price, Wojciech Matusik, and Alán Aspuru-Guzik</div>
        <div class="contrib"><sup>&ast;</sup>Equal contribution</div>
        <p>Draw molecular components in the Ketcher board, add them to the unit cell, set how many copies of each, then sample candidate crystal packings and inspect and download their CIFs.</p>
        <div class="badges">
            <a href="https://arxiv.org/abs/2606.03199"><img src="https://img.shields.io/badge/arXiv-2606.03199-b31b1b.svg" /></a>
            <a href="https://github.com/aspuru-guzik-group/clari"><img src="https://img.shields.io/badge/GitHub-aspuru--guzik--group%2Fclari-24292f.svg?logo=github" /></a>
            <a href="https://huggingface.co/the-matter-lab/clari"><img src="https://img.shields.io/badge/Hugging%20Face-Data-blue" /></a>
        </div>
    </section>
    """)
    return


@app.cell
def _(anywidget, traitlets):
    class KetcherWidget(anywidget.AnyWidget):
        """Embeds the standalone Ketcher editor in an iframe and mirrors the drawn
        structure's SMILES into a synced traitlet via Ketcher's postMessage API."""

        _esm = """
        function render({ model, el }) {
          const wrap = document.createElement("div");
          wrap.style.cssText = "width:100%;height:540px;border:1px solid #d7dee8;border-radius:8px;overflow:hidden;background:#fff;";
          const iframe = document.createElement("iframe");
          iframe.src = "https://ketcher.mireklzicar.com";
          iframe.style.cssText = "width:100%;height:100%;border:none;";
          wrap.appendChild(iframe);
          el.appendChild(wrap);

          function onMsg(event) {
            if (event.source !== iframe.contentWindow) return;
            const { type, payload, smiles } = event.data || {};
            if (type === "init") {
              iframe.contentWindow.postMessage({ type: "getSmiles" }, "*");
            } else if (type === "smiles") {
              model.set("smiles", payload || "");
              model.save_changes();
            } else if (type === "smiles-update") {
              model.set("smiles", smiles || "");
              model.save_changes();
            }
          }
          window.addEventListener("message", onMsg);
        }
        export default { render };
        """
        smiles = traitlets.Unicode("").tag(sync=True)

    return (KetcherWidget,)


@app.cell
def _(anywidget, traitlets):
    class Mol3DWidget(anywidget.AnyWidget):
        """Inline 3Dmol.js structure viewer (no iframe → not blocked by molab CSP).

        Mirrors clari.chem.draw: wrapped atoms as ball-and-stick, plus a per-frame
        lattice box and red/green/blue cell-axis arrows. Drive it with a concatenated
        multi-frame `frames_xyz` string and a matching `box_frames` JSON list (one box
        per frame); set `animate` to loop through them. The lattice is denoised too, so
        each frame carries its own box."""

        _esm = """
        let _loader;
        function load3Dmol() {
          if (window.$3Dmol) return Promise.resolve(window.$3Dmol);
          if (!_loader) {
            _loader = new Promise((resolve, reject) => {
              const s = document.createElement("script");
              s.src = "https://3Dmol.org/build/3Dmol-min.js";
              s.onload = () => resolve(window.$3Dmol);
              s.onerror = reject;
              document.head.appendChild(s);
            });
          }
          return _loader;
        }

        const pt = (p) => ({ x: p[0], y: p[1], z: p[2] });

        function render({ model, el }) {
          const h = model.get("height") || 520;
          const box = document.createElement("div");
          box.style.cssText = `position:relative;width:100%;height:${h}px;border:1px solid #d7dee8;border-radius:8px;overflow:hidden;background:#fff;`;
          el.appendChild(box);

          load3Dmol().then(($3Dmol) => {
            const viewer = $3Dmol.createViewer(box, { backgroundColor: "white" });
            viewer.addModelsAsFrames(model.get("frames_xyz"), "xyz");
            viewer.setStyle({}, { stick: { radius: 0.2 }, sphere: { scale: 0.2 } });

            let boxes = [];
            try { boxes = JSON.parse(model.get("box_frames") || "[]"); } catch (e) {}
            boxes.forEach((bf, i) => {
              (bf.lines || []).forEach((seg) =>
                viewer.addCylinder({ start: pt(seg[0]), end: pt(seg[1]), radius: 0.05, color: "#888888", frame: i })
              );
              (bf.arrows || []).forEach((a) =>
                viewer.addArrow({ start: pt(a[0]), end: pt(a[1]), radius: 0.2, radiusRatio: 2, mid: 0.92, color: a[2], frame: i })
              );
            });

            viewer.zoomTo();
            viewer.render();
            if (model.get("animate")) {
              viewer.animate({ loop: "forward", interval: model.get("interval") || 120 });
            }
            new ResizeObserver(() => viewer.resize()).observe(box);
          });
        }
        export default { render };
        """
        frames_xyz = traitlets.Unicode("").tag(sync=True)
        box_frames = traitlets.Unicode("[]").tag(sync=True)
        animate = traitlets.Bool(False).tag(sync=True)
        interval = traitlets.Float(120.0).tag(sync=True)
        height = traitlets.Int(520).tag(sync=True)

    return (Mol3DWidget,)


@app.cell
def _(KetcherWidget, mo):
    ketcher = mo.ui.anywidget(KetcherWidget())
    return (ketcher,)


@app.cell
def _(add_btn, ketcher, mo):
    _current = (ketcher.value or {}).get("smiles", "").strip()
    mo.vstack(
        [
            mo.md("<div class='step'>1. Draw a molecular component</div>"),
            mo.md(
                "<p style='color:#657188;font-size:.92rem'>Sketch a structure below, then add it to the unit cell. Draw and add several components for co-crystals.</p>"
            ),
            ketcher,
            add_btn,
            mo.md(
                f"<div class='smiles'>Current drawing: {_current or '— nothing drawn yet —'}</div>"
            ),
        ],
        gap=0.5,
    )
    return


@app.cell
def _(mo):
    # Each entry is one molecular component of the unit cell.
    get_comps, set_comps = mo.state([], allow_self_loops=True)
    return get_comps, set_comps


@app.cell
def _(mo):
    model = mo.ui.dropdown(
        options=["Clari Medium", "Clari Large", "Clari Huge"], value="Clari Medium", full_width=True
    )
    samples = mo.ui.number(start=1, stop=64, step=1, value=8, full_width=True)
    n_steps = mo.ui.number(start=1, stop=200, step=1, value=50, full_width=True)
    filter_clashing = mo.ui.checkbox(value=True, label="Filter clashing structures")
    run = mo.ui.run_button(label="Generate crystal packings", kind="success")
    return filter_clashing, model, n_steps, run, samples


@app.function
def mol_svg(smi, size=88):
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return "<div style='color:#94a3b8;font-size:11px'>no preview</div>"
    d = rdMolDraw2D.MolDraw2DSVG(size, size)
    d.drawOptions().clearBackground = False
    d.DrawMolecule(m)
    d.FinishDrawing()
    return d.GetDrawingText()


@app.cell
def _(get_comps, ketcher, mo, set_comps):
    comps = get_comps()
    copies_inputs = mo.ui.array(
        [mo.ui.number(start=1, stop=64, step=1, value=c["copies"]) for c in comps]
    )

    def snapshot():
        # Preserve in-flight copy edits when adding / deleting a row.
        return [
            {"smiles": comps[j]["smiles"], "copies": int(copies_inputs[j].value)}
            for j in range(len(comps))
        ]

    del_btns = mo.ui.array(
        [
            mo.ui.button(
                label="🗑",
                on_click=lambda _, i=i: set_comps([r for j, r in enumerate(snapshot()) if j != i]),
            )
            for i in range(len(comps))
        ]
    )

    def _add(_):
        smi = (ketcher.value or {}).get("smiles", "").strip()
        if smi:
            set_comps(snapshot() + [{"smiles": smi, "copies": 4}])

    add_btn = mo.ui.button(
        label="➕ Add drawn molecule to unit cell",
        kind="neutral",
        on_click=_add,
        full_width=True,
    )

    smiles_text = mo.ui.text(
        placeholder="Paste a SMILES, e.g. CC(=O)Oc1ccccc1C(=O)O", full_width=True
    )

    def _add_smiles(_):
        from rdkit import Chem

        smi = smiles_text.value.strip()
        if smi and Chem.MolFromSmiles(smi) is not None:
            set_comps(snapshot() + [{"smiles": smi, "copies": 4}])

    add_smiles_btn = mo.ui.button(label="➕ Add SMILES", kind="neutral", on_click=_add_smiles)

    if comps:
        _rows = [
            mo.hstack(
                [
                    mo.Html(
                        f"<div class='molcard' style='width:96px;height:96px;display:flex;align-items:center;justify-content:center'>{mol_svg(c['smiles'])}</div>"
                    ),
                    mo.vstack(
                        [
                            mo.Html(
                                f"<div class='smiles' title='{c['smiles']}'>{c['smiles']}</div>"
                            ),
                            mo.hstack(
                                [
                                    mo.md(
                                        "<span style='color:#657188;font-size:.85rem'>Number of copies in the unit cell</span>"
                                    ),
                                    copies_inputs[i],
                                ],
                                justify="start",
                                gap=0.5,
                                align="center",
                            ),
                        ],
                        gap=0.4,
                    ),
                    del_btns[i],
                ],
                justify="start",
                gap=0.6,
                align="center",
                widths=[1, 4, 1],
            )
            for i, c in enumerate(comps)
        ]
        _list = mo.vstack(_rows, gap=0.5)
    else:
        _list = mo.md(
            "<div style='border:2px dashed #d7dee8;border-radius:10px;padding:28px;text-align:center;color:#94a3b8'>🧪 No components yet.<br/>Draw a structure and click <b>Add drawn molecule to unit cell</b>.</div>"
        )

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md("<div class='step'>2. Unit cell composition</div>"),
                    mo.md(f"<span style='color:#657188'>{len(comps)} component(s)</span>"),
                ],
                justify="space-between",
            ),
            _list,
            mo.hstack(
                [smiles_text, add_smiles_btn],
                justify="start",
                gap=0.5,
                align="center",
                widths=[5, 1],
            ),
        ],
        gap=0.6,
    )
    return add_btn, copies_inputs


@app.cell
def _(filter_clashing, mo, model, n_steps, run, samples):
    def _field(lbl, el):
        return mo.vstack(
            [
                mo.md(f"<span style='font-size:.82rem;font-weight:600;color:#657188'>{lbl}</span>"),
                el,
            ],
            gap=0.25,
        )

    mo.vstack(
        [
            mo.md("<div class='step'>3. Model &amp; generation options</div>"),
            mo.hstack(
                [
                    _field("Model", model),
                    _field("Candidates to sample", samples),
                    _field("Denoising steps", n_steps),
                ],
                widths="equal",
                gap=1.0,
                align="start",
            ),
            filter_clashing,
            run,
        ],
        gap=0.6,
    )
    return


@app.cell
def _(mo):
    get_result, set_result = mo.state(None)
    get_sel, set_sel = mo.state(0, allow_self_loops=True)
    return get_result, get_sel, set_result, set_sel


@app.cell
def _(
    ClariSampler,
    copies_inputs,
    filter_clashing,
    get_comps,
    model,
    n_steps,
    run,
    sample_trajectory,
    samples,
    set_result,
    set_sel,
):
    if run.value:
        _model_ids = {"Clari Medium": "clari-m", "Clari Large": "clari-l", "Clari Huge": "clari-h"}
        _comps = get_comps()
        _smiles = [c["smiles"] for c in _comps if c["smiles"].strip()]
        _copies = [
            int(copies_inputs[i].value) for i, c in enumerate(_comps) if c["smiles"].strip()
        ]
        if _smiles:
            import traceback

            try:
                _sampler = ClariSampler(
                    _model_ids[model.value],
                    n_steps=int(n_steps.value),
                    torch_threads=1,
                    filter_clashing=bool(filter_clashing.value),
                )
                _trajectories = sample_trajectory(
                    _sampler,
                    _smiles,
                    copies=_copies,
                    samples=int(samples.value),
                    filter_clashing=bool(filter_clashing.value),
                )
                set_sel(0)
                set_result(
                    {
                        "crystals": [t.crystal for t in _trajectories],
                        "trajectories": _trajectories,
                        "smiles": " + ".join(
                            f"{_s} ×{_c}" for _s, _c in zip(_smiles, _copies, strict=False)
                        ),
                        "model": model.value,
                    }
                )
            except Exception as _exc:
                # Surface inference errors (e.g. a disconnected molecule) in the UI
                # instead of letting them crash the cell with no visible message.
                set_result(
                    {"error": f"{type(_exc).__name__}: {_exc}", "traceback": traceback.format_exc()}
                )
    return


@app.cell
def _(get_result, mo):
    result = get_result()
    mo.stop(
        result is None,
        mo.md(
            "<p style='color:#657188'>Ready — add components and generate to see candidate packings here.</p>"
        ),
    )
    mo.stop(
        result.get("error") is not None,
        mo.callout(
            mo.vstack(
                [
                    mo.md(f"**Sampling failed:** {result.get('error', '')}"),
                    mo.accordion(
                        {"Full traceback": mo.md(f"```\n{result.get('traceback', '')}\n```")}
                    ),
                ]
            ),
            kind="danger",
        ),
    )
    mo.stop(
        len(result["crystals"]) == 0,
        mo.md(
            "<p style='color:#b45309'>Every sampled candidate had atom clashes and was filtered out. Try more candidates, fewer copies, or uncheck <b>Filter clashing structures</b>.</p>"
        ),
    )
    crystals = result["crystals"]
    trajectories = result["trajectories"]
    sampled_smiles = result["smiles"]
    return crystals, sampled_smiles, trajectories


@app.cell
def _(crystals, get_sel, mo, set_sel):
    _sel = min(get_sel(), len(crystals) - 1)
    _cards = mo.ui.array(
        [
            mo.ui.button(
                label=f"Candidate #{i + 1}",
                kind="success" if i == _sel else "neutral",
                on_click=lambda _, i=i: set_sel(i),
            )
            for i in range(len(crystals))
        ]
    )
    mo.vstack(
        [
            mo.md("<div class='step'>4. Candidate packings</div>"),
            mo.md(
                "<p style='color:#657188;font-size:.9rem'>Click a candidate to load it into the 3D viewer below.</p>"
            ),
            mo.hstack(list(_cards), justify="start", gap=0.4, wrap=True),
        ],
        gap=0.5,
    )
    return


@app.cell
def _(
    Mol3DWidget,
    base64,
    crystals,
    get_sel,
    io,
    mo,
    sampled_smiles,
    trajectories,
    zipfile,
):
    _idx = min(get_sel(), len(crystals) - 1)
    _h = 520
    _crystal = crystals[_idx].wrapped(mode="com", bounds=(-0.5, 0.5))

    # Zip of every candidate CIF.
    _zip_buf = io.BytesIO()
    with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
        for _i, _c in enumerate(crystals):
            _zf.writestr(
                f"candidate_{_i + 1:03d}.cif", _c.wrapped(mode="com", bounds=(-0.5, 0.5)).to_cif()
            )
    _dl_all = mo.Html(
        f'<a class="dl" download="clari_candidates.zip" href="data:application/zip;base64,{base64.b64encode(_zip_buf.getvalue()).decode()}">⬇ Download all CIFs (ZIP)</a>'
    )

    _cif = _crystal.to_cif()
    _dl_one = mo.Html(
        f'<a class="dl" download="candidate_{_idx + 1:03d}.cif" href="data:chemical/x-cif;base64,{base64.b64encode(_cif.encode()).decode()}">⬇ Download current CIF</a>'
    )

    import json

    import numpy as np

    # Centered unit-cube edges (matches clari.chem.draw.UNIT_CUBE_EDGES).
    _CUBE = np.asarray(
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
        ],
        dtype=float,
    )

    def _frame_to_xyz(_frame):
        from ase.data import chemical_symbols

        _nums = _frame.atom_nums.detach().cpu().numpy()
        _xyz = _frame.coords.detach().cpu().numpy()
        _lines = [str(len(_nums)), ""]
        for _num, (_x, _y, _w) in zip(_nums, _xyz):
            _lines.append(f"{chemical_symbols[int(_num)]} {_x:.4f} {_y:.4f} {_w:.4f}")
        return "\n".join(_lines)

    def _box_geometry(_frame):
        # Lattice box + r/g/b cell-axis arrows, centered at the origin (matches draw.py).
        _lat = _frame.lattice.detach().cpu().numpy()
        _o = -0.5 * _lat.sum(axis=0)
        _edges = (_CUBE - 0.5) @ _lat
        _lines = [[seg[0].tolist(), seg[1].tolist()] for seg in _edges]
        _arrows = [
            [_o.tolist(), _p.tolist(), _col]
            for _p, _col in zip(_lat + _o, ["red", "green", "blue"])
        ]
        return {"lines": _lines, "arrows": _arrows}

    # Static viewer: the final wrapped structure with its lattice.
    _static_xyz = _frame_to_xyz(_crystal)
    _static_box = json.dumps([_box_geometry(_crystal)])

    # Trajectory: every denoising step, wrapped, each with its own (denoised) lattice.
    _traj = trajectories[_idx]
    _steps = [
        _traj.crystal.replace(x=_traj.trajectory[_i]).wrapped(mode="com", bounds=(-0.5, 0.5))
        for _i in range(_traj.trajectory.shape[0])
    ]
    _xyz_blocks = [_frame_to_xyz(_s) for _s in _steps]
    _boxes = [_box_geometry(_s) for _s in _steps]

    # Hold on the final frame for a beat before looping.
    _duration_play, _duration_stop = 2.0, 1.2
    _interval = _duration_play * 1000 / max(1, len(_steps))
    _hold = round(_duration_stop / (_interval / 1000))
    _xyz_blocks += [_xyz_blocks[-1]] * _hold
    _boxes += [_boxes[-1]] * _hold

    # Inline 3Dmol widgets (no iframe → not blocked by molab CSP).
    _viewer = mo.ui.anywidget(
        Mol3DWidget(frames_xyz=_static_xyz, box_frames=_static_box, animate=False, height=_h)
    )
    _traj_viewer = mo.ui.anywidget(
        Mol3DWidget(
            frames_xyz="\n".join(_xyz_blocks),
            box_frames=json.dumps(_boxes),
            animate=True,
            interval=_interval,
            height=_h,
        )
    )

    mo.vstack(
        [
            mo.md(f"### 3D unit cell viewer · candidate #{_idx + 1}"),
            mo.hstack([_dl_one, _dl_all], justify="start", gap=0.6),
            _viewer,
            mo.md(
                f"<p style='color:#657188;font-size:.9rem'>{sampled_smiles} · candidate {_idx + 1} of {len(crystals)}</p>"
            ),
            mo.accordion(
                {
                    "Sampling trajectory": _traj_viewer,
                    "CIF text": mo.md(f"```cif\n{_cif}\n```"),
                }
            ),
        ],
        gap=0.75,
    )
    return


if __name__ == "__main__":
    app.run()
