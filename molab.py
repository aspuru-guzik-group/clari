# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "clari",
#     "marimo",
#     "py3dmol",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    import io
    import zipfile

    import marimo as mo
    import py3Dmol
    import torch

    from clari.chem.draw import draw_crystal, draw_crystal_trajectory_from_batch
    from clari.inference import ClariSampler
    from clari.inference.sample import sample_trajectory

    return ClariSampler, base64, draw_crystal, draw_crystal_trajectory_from_batch, io, mo, py3Dmol, sample_trajectory, torch, zipfile


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
        .dl { align-items: center; background: var(--accent); border-radius: 6px; color: white; display: inline-flex; font-weight: 650; line-height: 1; padding: 10px 13px; text-decoration: none; }
        </style>
        <section class="hero">
            <h1>Fast Organic Crystal Structure Prediction with Unit Cell Flow Matching</h1>
            <div class="authors">Alston Lo, Luka Mucko, Austin H. Cheng, Andy Cai, Alastair J. A. Price, Wojciech Matusik, and Alán Aspuru-Guzik</div>
            <p>Choose a model, set the unit-cell request, then sample and inspect generated CIF structures and their flow trajectories.</p>
            <div class="badges">
                <a href="https://arxiv.org/abs/2606.03199"><img src="https://img.shields.io/badge/arXiv-2606.03199-b31b1b.svg" /></a>
                <a href="https://github.com/aspuru-guzik-group/clari"><img src="https://img.shields.io/badge/GitHub-aspuru--guzik--group%2Fclari-24292f.svg?logo=github" /></a>
                <a href="https://huggingface.co/the-matter-lab/clari"><img src="https://img.shields.io/badge/Hugging%20Face-Data-blue" /></a>
            </div>
        </section>
    """)
    return


@app.cell
def _(mo):
    model = mo.ui.dropdown(options=["Clari Medium", "Clari Large", "Clari Huge"], value="Clari Medium", label="Model")
    samples = mo.ui.number(start=1, stop=64, step=1, value=8, label="Samples")
    n_steps = mo.ui.number(start=1, stop=200, step=1, value=50, label="Steps")
    z = mo.ui.number(start=1, stop=16, step=1, value=4, label="Z")
    run = mo.ui.run_button(label="Sample crystals")
    return model, n_steps, run, samples, z


@app.cell
def _(mo):
    get_rows, set_rows = mo.state(
        [
            {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "copies": 1},
            {"smiles": "O", "copies": 3},
        ],
        allow_self_loops=True,
    )
    return get_rows, set_rows


@app.cell
def _(get_rows, mo, set_rows):
    rows = get_rows()
    smiles_inputs = mo.ui.array([mo.ui.text(value=r["smiles"]) for r in rows])
    copies_inputs = mo.ui.array([mo.ui.number(start=1, stop=16, step=1, value=r["copies"]) for r in rows])

    def snapshot():
        # current typed values, so add/delete don't reset edits
        return [
            {"smiles": smiles_inputs[j].value, "copies": int(copies_inputs[j].value)}
            for j in range(len(smiles_inputs))
        ]

    del_btns = mo.ui.array([
        mo.ui.button(
            label="×",
            on_click=lambda _, i=i: set_rows(
                [r for j, r in enumerate(snapshot()) if j != i] or [{"smiles": "", "copies": 1}]
            ),
        )
        for i in range(len(rows))
    ])
    add_btn = mo.ui.button(
        label="+",
        on_click=lambda _: set_rows(snapshot() + [{"smiles": "", "copies": 1}]),
    )
    return add_btn, copies_inputs, del_btns, smiles_inputs


@app.cell
def _(add_btn, copies_inputs, del_btns, mo, model, n_steps, run, samples, smiles_inputs, z):
    _header = mo.hstack(
        [mo.Html("<div style='min-width:30px'></div>"), mo.Html("<strong>SMILES</strong>" + "&nbsp;" * 30), mo.Html("<div style='min-width:90px'><strong>Count</strong></div>")],
        justify="start", gap=0.5,
    )
    _rows = [
        mo.hstack([del_btns[i], smiles_inputs[i], copies_inputs[i]], justify="start", gap=0.5)
        for i in range(len(smiles_inputs))
    ]
    mo.vstack([
        mo.hstack([model, samples, n_steps, z], widths="equal"),
        mo.md("<p style='color:#657188;font-size:.92rem'>Each row is one molecular component; Count is its number in the asymmetric unit. Z multiplies the whole composition to form the unit cell.</p>"),
        mo.hstack([mo.vstack([_header] + _rows + [add_btn], gap=0.4)], justify="center"),
        run,
    ], gap=1.0)
    return


@app.cell
def _(mo):
    get_result, set_result = mo.state(None)
    return get_result, set_result


@app.cell
def _(ClariSampler, copies_inputs, model, n_steps, run, sample_trajectory, samples, set_result, smiles_inputs, z):
    if run.value:
        _model_ids = {"Clari Medium": "clari-m", "Clari Large": "clari-l", "Clari Huge": "clari-h"}
        _z = int(z.value)
        _smiles = [smiles_inputs[i].value.strip() for i in range(len(smiles_inputs)) if smiles_inputs[i].value.strip()]
        _copies = [int(copies_inputs[i].value) * _z for i in range(len(smiles_inputs)) if smiles_inputs[i].value.strip()]
        _sampler = ClariSampler(_model_ids[model.value], n_steps=int(n_steps.value), torch_threads=1)
        _trajectories = sample_trajectory(_sampler, _smiles, copies=_copies, samples=int(samples.value))
        set_result({
            "crystals": [t.crystal for t in _trajectories],
            "trajectories": _trajectories,
            "smiles": " + ".join(f"{_s} x{_c}" for _s, _c in zip(_smiles, _copies, strict=False)),
            "model": model.value,
        })
    return


@app.cell
def _(get_result, mo):
    result = get_result()
    mo.stop(result is None, mo.md("<p style='color:#657188'>Ready.</p>"))
    crystals = result["crystals"]
    trajectories = result["trajectories"]
    sampled_smiles = result["smiles"]
    return crystals, sampled_smiles, trajectories


@app.cell
def _(crystals, mo):
    sample = mo.ui.dropdown(
        options=[f"Sample {i}" for i in range(len(crystals))], value="Sample 0", label="Crystal"
    )
    sample
    return (sample,)


@app.cell
def _(base64, crystals, draw_crystal, draw_crystal_trajectory_from_batch, io, mo, py3Dmol, sample, sampled_smiles, torch, trajectories, zipfile):
    _idx = int(sample.value.split()[-1])
    _h = 520
    _crystal = crystals[_idx].wrapped(mode="com", bounds=(-0.5, 0.5))

    _zip_buf = io.BytesIO()
    with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
        for _i, _c in enumerate(crystals):
            _zf.writestr(f"sample_{_i:06d}.cif", _c.wrapped(mode="com", bounds=(-0.5, 0.5)).to_cif())
    _dl = mo.Html(f'<a class="dl" download="clari_samples.zip" href="data:application/zip;base64,{base64.b64encode(_zip_buf.getvalue()).decode()}">Download CIF ZIP</a>')

    _view = draw_crystal(
        _crystal.atom_nums.detach().cpu().numpy(),
        _crystal.coords.detach().cpu().numpy(),
        lattice=_crystal.lattice.detach().cpu().numpy(),
        view=py3Dmol.view(width="100%", height=_h),
    )

    _traj = trajectories[_idx]
    _frames = torch.stack([
        _traj.crystal.replace(x=_traj.trajectory[i]).wrapped(mode="com", bounds=(-0.5, 0.5)).x
        for i in range(_traj.trajectory.shape[0])
    ])
    _anim = draw_crystal_trajectory_from_batch(
        [_traj.__class__(crystal=_crystal, trajectory=_frames)],
        batch_idx=0, view=py3Dmol.view(width="100%", height=_h), duration_play=2, duration_stop=4,
    )
    _anim.setStyle({"stick": {"radius": 0.2}, "sphere": {"scale": 0.2}})
    _anim.setFrame(len(_frames) - 1)
    _anim.zoomTo()
    _anim.setFrame(0)
    _anim.animate({"interval": 2000 / len(_frames)})

    mo.vstack([
        mo.md("### Final crystal"),
        _dl,
        mo.iframe(_view.write_html(fullpage=True), height=_h + 20),
        mo.md(f"<p style='color:#657188;font-size:.9rem'>{sampled_smiles} · sample {_idx}</p>"),
        mo.md("### Sampling trajectory"),
        mo.iframe(_anim.write_html(fullpage=True), height=_h + 20),
        mo.accordion({"CIF": mo.md(f"```cif\n{_crystal.to_cif()}\n```")}),
    ], gap=0.75)
    return


if __name__ == "__main__":
    app.run()
