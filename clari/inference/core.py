from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Sample organic crystal structures from unit cell SMILES strings."
    )
    parser.add_argument("pos_args", nargs="*", metavar="SMILES [copies]", default=[])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--smiles", action="append", default=None)
    parser.add_argument("--copies", action="append", default=None)
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--model", type=str, default="clari-m")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--no_pbar", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = vars(parser.parse_args(argv))
    from clari.inference.inputs import parse_cli_request, parse_config_requests

    options = {}
    if args["config"]:
        if args["pos_args"] or args["smiles"]:
            raise ValueError("`--config` cannot be combined with direct SMILES input.")
        requests, options = parse_config_requests(args["config"])
        known_top_level = set(args) | {"use_ema", "use_bf16", "pbar"}
        for key, value in options.items():
            if key not in known_top_level:
                from clari.inference.inputs import _warn

                _warn(f"ignoring unknown top-level config key {key!r}")
            elif key in args and args[key] == parser.get_default(key):
                args[key] = value
    else:
        requests = parse_cli_request(
            args["pos_args"], args["smiles"], args["copies"], args["id"], args["samples"]
        )
    if args["output_dir"] is None:
        if args["config"] and len(requests) > 1:
            args["output_dir"] = f"results/{Path(args['config']).stem}"
        else:
            args["output_dir"] = f"results/{requests[0].id}"

    from clari.inference.sample import (
        ClariSampler,
        sample,
        sample_batch_to_directories,
        validate_requests,
    )

    validate_requests(requests)

    use_ema = False if args["no_ema"] else bool(options.get("use_ema", True))
    use_bf16 = False if args["no_bf16"] else bool(options.get("use_bf16", True))
    pbar = False if args["no_pbar"] else bool(options.get("pbar", True))

    if args["config"] and len(requests) > 1:
        base_dir = Path(args["output_dir"])
        base_dir.mkdir(parents=True, exist_ok=True)
        sampler = ClariSampler(
            args["model"],
            device=args["device"],
            use_ema=use_ema,
            use_bf16=use_bf16,
            n_steps=args["n_steps"],
            compile=args["compile"],
            torch_threads=args["torch_threads"],
            num_gpus=args["num_gpus"],
            seed=args["seed"],
        )
        output_dirs = [base_dir / str(request.id) for request in requests]
        results = sample_batch_to_directories(
            sampler,
            requests=requests,
            output_dirs=output_dirs,
            base_dir=base_dir,
            batch_size=args["batch_size"],
            num_gpus=args["num_gpus"],
            overwrite=args["overwrite"],
            pbar=pbar,
            seed=args["seed"],
        )
        (base_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "config": args["config"],
                    "requests": [
                        {
                            "id": request.id,
                            "output_dir": str(result),
                            "predictions": str(result / "predictions.parquet"),
                        }
                        for request, result in zip(requests, results)
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        for result in results:
            print(result / "predictions.parquet")
        return 0

    result = sample(
        requests,
        model=args["model"],
        output_dir=args["output_dir"],
        batch_size=args["batch_size"],
        num_gpus=args["num_gpus"],
        device=args["device"],
        n_steps=args["n_steps"],
        use_ema=use_ema,
        use_bf16=use_bf16,
        compile=args["compile"],
        torch_threads=args["torch_threads"],
        overwrite=args["overwrite"],
        pbar=pbar,
        seed=args["seed"],
    )
    print(result / "predictions.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
