from __future__ import annotations

import sys

from jsonargparse import ArgumentParser

from clari.inference.inputs import (
    parse_cli_request,
    parse_config_requests,
)
from clari.inference.sample import sample


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Sample organic crystal structures from unit cell SMILES strings."
    )
    parser.add_argument("pos_args", nargs="*", default=[])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--smiles", action="append", default=None)
    parser.add_argument("--copies", action="append", default=None)
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, default="clari-m")
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
    try:
        args = vars(parser.parse_args(argv))
        config_path = args.pop("config")
        config_options = {}
        if config_path:
            requests, config_options = parse_config_requests(config_path)
            for key, value in config_options.items():
                if key in args and args[key] == parser.get_default(key):
                    args[key] = value
        else:
            requests = parse_cli_request(
                args.pop("pos_args"),
                args.pop("smiles"),
                args.pop("copies"),
                args.pop("id"),
                args.pop("samples"),
            )
        use_ema = False if args["no_ema"] else bool(config_options.get("use_ema", True))
        use_bf16 = False if args["no_bf16"] else bool(config_options.get("use_bf16", True))
        pbar = False if args["no_pbar"] else bool(config_options.get("pbar", True))
        if args["output_dir"] is None:
            raise ValueError("`--output_dir` is required for CLI sampling.")
        result = sample(
            requests,
            checkpoint_path=args["checkpoint_path"],
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
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(result / "predictions.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
