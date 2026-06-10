from __future__ import annotations

import sys
from argparse import ArgumentParser


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
    try:
        from clari.inference.inputs import parse_cli_request, parse_config_requests

        options = {}
        if args["config"]:
            if args["pos_args"] or args["smiles"]:
                raise ValueError("`--config` cannot be combined with direct SMILES input.")
            requests, options = parse_config_requests(args["config"])
            for key, value in options.items():
                if key in args and args[key] == parser.get_default(key):
                    args[key] = value
        else:
            requests = parse_cli_request(
                args["pos_args"], args["smiles"], args["copies"], args["id"], args["samples"]
            )
        if args["output_dir"] is None:
            args["output_dir"] = f"results/{requests[0].id}"

        from clari.inference.sample import sample

        result = sample(
            requests,
            model=args["model"],
            output_dir=args["output_dir"],
            batch_size=args["batch_size"],
            num_gpus=args["num_gpus"],
            device=args["device"],
            n_steps=args["n_steps"],
            use_ema=False if args["no_ema"] else bool(options.get("use_ema", True)),
            use_bf16=False if args["no_bf16"] else bool(options.get("use_bf16", True)),
            compile=args["compile"],
            torch_threads=args["torch_threads"],
            overwrite=args["overwrite"],
            pbar=False if args["no_pbar"] else bool(options.get("pbar", True)),
            seed=args["seed"],
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(result / "predictions.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
