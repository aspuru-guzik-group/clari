# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "clari",
#     "marimo",
#     "py3dmol",
#     "anywidget",
#     "traitlets",
#     "rdkit",
#     "httpx[socks]",
# ]
#
# [tool.uv.sources]
# clari = { path = ".", editable = true }
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


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
        draw_crystal,
        draw_crystal_trajectory_from_batch,
        io,
        mo,
        py3Dmol,
        sample_trajectory,
        torch,
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
        .authors { color: var(--muted); font-size: .95rem; margin: 10px 0 14px; }
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
            <div class="authors">Alston Lo, Luka Mucko, Austin H. Cheng, Andy Cai, Alastair J. A. Price, Wojciech Matusik, and Alán Aspuru-Guzik</div>
            <p>Draw molecular components in the Ketcher board, add them to the asymmetric unit, set how many copies of each, then sample candidate crystal packings and inspect / download their CIFs.</p>
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
def _(KetcherWidget, mo):
    ketcher = mo.ui.anywidget(KetcherWidget())
    return (ketcher,)


@app.cell
def _(ketcher, mo):
    _current = (ketcher.value or {}).get("smiles", "").strip()
    mo.vstack(
        [
            mo.md("<div class='step'>1. Draw a molecular component</div>"),
            mo.md(
                "<p style='color:#657188;font-size:.92rem'>Sketch a structure below, then add it to the asymmetric unit on the right. Draw and add several components for co-crystals.</p>"
            ),
            ketcher,
            mo.md(
                f"<div class='smiles'>Current drawing: {_current or '— nothing drawn yet —'}</div>"
            ),
        ],
        gap=0.5,
    )
    return


@app.cell
def _(mo):
    # Each entry is one molecular component of the asymmetric unit.
    get_comps, set_comps = mo.state([], allow_self_loops=True)
    return get_comps, set_comps


@app.cell
def _(mo):
    model = mo.ui.dropdown(
        options=["Clari Medium", "Clari Large", "Clari Huge"], value="Clari Medium", full_width=True
    )
    samples = mo.ui.number(start=1, stop=64, step=1, value=8, full_width=True)
    n_steps = mo.ui.number(start=1, stop=200, step=1, value=50, full_width=True)
    z = mo.ui.number(start=1, stop=16, step=1, value=4, full_width=True)
    filter_clashing = mo.ui.checkbox(value=True, label="Filter clashing structures")
    run = mo.ui.run_button(label="Generate crystal packings", kind="success")
    return filter_clashing, model, n_steps, run, samples, z


@app.cell
def _(mo):
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

    return (mol_svg,)


@app.cell
def _(get_comps, ketcher, mo, mol_svg, set_comps):
    comps = get_comps()
    copies_inputs = mo.ui.array(
        [mo.ui.number(start=1, stop=16, step=1, value=c["copies"]) for c in comps]
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
            set_comps(snapshot() + [{"smiles": smi, "copies": 1}])

    add_btn = mo.ui.button(
        label="➕ Add drawn molecule to unit cell", kind="neutral", on_click=_add
    )

    smiles_text = mo.ui.text(
        placeholder="Paste a SMILES, e.g. CC(=O)Oc1ccccc1C(=O)O", full_width=True
    )

    def _add_smiles(_):
        from rdkit import Chem

        smi = smiles_text.value.strip()
        if smi and Chem.MolFromSmiles(smi) is not None:
            set_comps(snapshot() + [{"smiles": smi, "copies": 1}])

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
                                        "<span style='color:#657188;font-size:.85rem'>Copies in asym. unit</span>"
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
            add_btn,
            mo.hstack(
                [smiles_text, add_smiles_btn],
                justify="start", gap=0.5, align="center", widths=[5, 1],
            ),
        ],
        gap=0.6,
    )
    return (copies_inputs,)


@app.cell
def _(filter_clashing, model, mo, n_steps, run, samples, z):
    def _field(lbl, el):
        return mo.vstack(
            [mo.md(f"<span style='font-size:.82rem;font-weight:600;color:#657188'>{lbl}</span>"), el],
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
                    _field("Z", z),
                ],
                widths="equal", gap=1.0, align="start",
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
    z,
):
    if run.value:
        _model_ids = {"Clari Medium": "clari-m", "Clari Large": "clari-l", "Clari Huge": "clari-h"}
        _comps = get_comps()
        _z = int(z.value)
        _smiles = [c["smiles"] for c in _comps if c["smiles"].strip()]
        _copies = [
            int(copies_inputs[i].value) * _z for i, c in enumerate(_comps) if c["smiles"].strip()
        ]
        if _smiles:
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
    base64,
    crystals,
    draw_crystal,
    draw_crystal_trajectory_from_batch,
    get_sel,
    io,
    mo,
    py3Dmol,
    sampled_smiles,
    torch,
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

    _view = draw_crystal(
        _crystal.atom_nums.detach().cpu().numpy(),
        _crystal.coords.detach().cpu().numpy(),
        lattice=_crystal.lattice.detach().cpu().numpy(),
        view=py3Dmol.view(width="100%", height=_h),
    )

    _traj = trajectories[_idx]
    _frames = torch.stack(
        [
            _traj.crystal.replace(x=_traj.trajectory[i]).wrapped(mode="com", bounds=(-0.5, 0.5)).x
            for i in range(_traj.trajectory.shape[0])
        ]
    )
    _anim = draw_crystal_trajectory_from_batch(
        [_traj.__class__(crystal=_crystal, trajectory=_frames)],
        batch_idx=0,
        view=py3Dmol.view(width="100%", height=_h),
        duration_play=2,
        duration_stop=4,
    )
    _anim.setStyle({"stick": {"radius": 0.2}, "sphere": {"scale": 0.2}})
    _anim.setFrame(len(_frames) - 1)
    _anim.zoomTo()
    _anim.setFrame(0)
    _anim.animate({"interval": 2000 / len(_frames)})

    mo.vstack(
        [
            mo.md(f"### 3D unit cell viewer · candidate #{_idx + 1}"),
            mo.hstack([_dl_one, _dl_all], justify="start", gap=0.6),
            mo.iframe(_view.write_html(fullpage=True), height=_h + 20),
            mo.md(
                f"<p style='color:#657188;font-size:.9rem'>{sampled_smiles} · candidate {_idx + 1} of {len(crystals)}</p>"
            ),
            mo.accordion(
                {
                    "Sampling trajectory": mo.iframe(
                        _anim.write_html(fullpage=True), height=_h + 20
                    ),
                    "CIF text": mo.md(f"```cif\n{_cif}\n```"),
                }
            ),
        ],
        gap=0.75,
    )
    return


if __name__ == "__main__":
    app.run()
