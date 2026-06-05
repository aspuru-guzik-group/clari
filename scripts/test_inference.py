from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import tempfile
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import polars as pl

import clari.inference.core as core
import clari.inference.sample as sample_mod
from clari.inference.export import export_cifs
from clari.inference.inputs import (
    SampleRequest,
    build_request,
    parse_cli_request,
    parse_config_requests,
    request_components,
    sanitize_id,
)
from clari.inference.sample import build_run_config, request_to_crystal, save


def assert_eq(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value: bool, label: str) -> None:
    if not value:
        raise AssertionError(label)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n")


def test_no_hydrogen_public_surface() -> None:
    assert_true("add_hs" not in {field.name for field in fields(SampleRequest)}, "SampleRequest has add_hs")
    checked = [
        build_request,
        parse_cli_request,
        parse_config_requests,
        sample_mod.ClariSampler.sample,
        sample_mod.sample_trajectory,
    ]
    for func in checked:
        params = inspect.signature(func).parameters
        assert_true("add_hs" not in params, f"{func.__qualname__} has add_hs")
        assert_true("no_add_hs" not in params, f"{func.__qualname__} has no_add_hs")

    parser = core.build_parser()
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            parser.parse_args(["--no_add_hs"])
        except SystemExit:
            pass
        else:
            raise AssertionError("CLI accepted removed --no_add_hs flag")


def test_request_building_and_cli_parsing() -> None:
    assert_eq(sanitize_id(" C C/O? x "), "C_C_O_x", "sanitize_id")

    single = build_request("CCO", copies=2, samples=3)
    assert_eq(single.smiles, "CCO", "single smiles")
    assert_eq(single.copies, 2, "single copies")
    assert_eq(single.samples, 3, "single samples")
    assert_eq(request_components(single), [("CCO", 2)], "single components")

    cocrystal = build_request([("CCO", 1), ("O", 3)], id="hydrate", samples=5)
    assert_eq(cocrystal.smiles, [("CCO", 1), ("O", 3)], "cocrystal smiles")
    assert_eq(cocrystal.copies, 1, "cocrystal request copies placeholder")
    assert_eq(request_components(cocrystal), [("CCO", 1), ("O", 3)], "cocrystal components")

    parsed = parse_cli_request(
        pos_args=[],
        smiles_flags=["CCO", "O"],
        copies_flags=["1", "3"],
        request_id="mix",
        samples=7,
    )
    assert_eq(len(parsed), 1, "parsed request count")
    assert_eq(parsed[0].id, "mix", "parsed id")
    assert_eq(request_components(parsed[0]), [("CCO", 1), ("O", 3)], "parsed components")
    assert_eq(parsed[0].samples, 7, "parsed samples")

    try:
        parse_cli_request(["4"], None, None, None, 1)
    except ValueError as exc:
        assert_true("copy count" in str(exc), "copy-count error message")
    else:
        raise AssertionError("copy count without SMILES was accepted")


def test_config_parsing_ignores_hydrogen_keys() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "batch.json"
        write_json(
            config_path,
            {
                "checkpoint_path": "local.ckpt",
                "output_dir": "results/batch",
                "add_hs": False,
                "no_add_hs": True,
                "requests": [
                    {
                        "id": "ethanol",
                        "smiles": "CCO",
                        "copies": 4,
                        "samples": 2,
                        "batch_size": 8,
                        "add_hs": False,
                    },
                    {
                        "id": "hydrate",
                        "smiles": [["CCO", 1], ["O", 3]],
                        "samples": 3,
                        "no_add_hs": True,
                    },
                ],
            },
        )

        requests, options = parse_config_requests(config_path)
        assert_eq(len(requests), 2, "config request count")
        assert_eq(options["checkpoint_path"], "local.ckpt", "config checkpoint passthrough")
        assert_eq(options["output_dir"], "results/batch", "config output passthrough")
        assert_eq(requests[0].id, "ethanol", "config single id")
        assert_eq(requests[0].batch_size, 8, "config batch size")
        assert_eq(request_components(requests[1]), [("CCO", 1), ("O", 3)], "config cocrystal")


def test_cli_requires_output_dir_and_passes_options() -> None:
    original_sample = core.sample
    calls = []

    def fake_sample(requests, **kwargs):
        calls.append((requests, kwargs))
        return Path(kwargs["output_dir"])

    core.sample = fake_sample
    try:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = core.main(["--smiles", "CCO", "--samples", "1"])
        assert_eq(code, 1, "missing output_dir exit code")
        assert_true("output_dir" in stderr.getvalue(), "missing output_dir error")

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = core.main(
                [
                    "--smiles",
                    "CCO",
                    "--id",
                    "ethanol",
                    "--samples",
                    "2",
                    "--output_dir",
                    "results/ethanol",
                    "--no_pbar",
                ]
            )
        assert_eq(code, 0, "successful CLI exit code")
        assert_eq(stdout.getvalue().strip(), "results/ethanol/predictions.parquet", "CLI output")
        assert_eq(len(calls), 1, "sample call count")
        requests, kwargs = calls[0]
        assert_eq(requests[0].id, "ethanol", "CLI request id")
        assert_eq(requests[0].samples, 2, "CLI request samples")
        assert_eq(kwargs["output_dir"], "results/ethanol", "CLI output_dir")
        assert_eq(kwargs["pbar"], False, "CLI pbar")
    finally:
        core.sample = original_sample


def test_request_to_crystal_does_not_pass_hydrogen_kwarg() -> None:
    original_crystal = sample_mod.Crystal
    calls = []

    class FakeCrystal:
        @staticmethod
        def from_smiles(components, **kwargs):
            calls.append((components, kwargs))
            return "crystal"

    sample_mod.Crystal = FakeCrystal
    try:
        result = request_to_crystal(build_request([("CCO", 1), ("O", 3)], id="mix"))
        assert_eq(result, "crystal", "request_to_crystal return")
        components, kwargs = calls[0]
        assert_eq(components, [("CCO", 1), ("O", 3)], "request_to_crystal components")
        assert_eq(kwargs, {"csd_id": "mix"}, "request_to_crystal kwargs")
    finally:
        sample_mod.Crystal = original_crystal


def test_run_config_save_and_export_contracts() -> None:
    requests = [build_request("CCO", id="ethanol", samples=2)]
    config = build_run_config(
        requests,
        checkpoint_path="clari-m",
        device="cpu",
        num_gpus=1,
        batch_size=None,
        n_steps=50,
        use_ema=True,
        use_bf16=False,
        compile=False,
        overwrite=False,
    )
    assert_true("add_hs" not in json.dumps(config), "run config contains hydrogen options")

    class FakeCrystal:
        def __init__(self, csd_id: str, cif: str):
            self.csd_id = csd_id
            self._cif = cif

        def to_cif(self) -> str:
            return self._cif

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        predictions_path = save(
            [FakeCrystal("ethanol", "data_ethanol_0"), FakeCrystal("ethanol", "data_ethanol_1")],
            root / "results",
        )
        df = pl.read_parquet(predictions_path)
        assert_eq(df.columns, ["id", "sample_idx", "cif"], "saved columns")
        assert_eq(df["sample_idx"].to_list(), [0, 1], "saved sample_idx")

        export_cifs(root / "results", output_dir=root / "all_cifs")
        assert_true((root / "all_cifs" / "ethanol" / "sample_000000.cif").is_file(), "all CIF 0")
        assert_true((root / "all_cifs" / "ethanol" / "sample_000001.cif").is_file(), "all CIF 1")

        pl.DataFrame(
            {
                "sample_idx": [0, 1],
                "id": ["ethanol", "ethanol"],
                "energies": [2.0, 1.0],
                "rank": [1, 0],
            }
        ).write_csv(root / "results" / "rankings.csv")
        export_cifs(root / "results", output_dir=root / "top_cifs", top_k=1)
        assert_true(
            (root / "top_cifs" / "ethanol" / "rank_0000_sample_000001.cif").is_file(),
            "top-ranked CIF",
        )
        assert_true(
            not (root / "top_cifs" / "ethanol" / "rank_0001_sample_000000.cif").exists(),
            "excluded lower-ranked CIF",
        )


def main() -> None:
    tests = [
        test_no_hydrogen_public_surface,
        test_request_building_and_cli_parsing,
        test_config_parsing_ignores_hydrogen_keys,
        test_cli_requires_output_dir_and_passes_options,
        test_request_to_crystal_does_not_pass_hydrogen_kwarg,
        test_run_config_save_and_export_contracts,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} inference tests")


if __name__ == "__main__":
    main()
