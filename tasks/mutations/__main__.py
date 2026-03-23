"""Common CLI for training-data and evaluation-signal mutations.

Examples:
    python -m tasks.mutations data label_noise <workspace> --noise-rate 0.3
    python -m tasks.mutations eval eval_leakage <workspace> --leak-rate 0.25
"""

from __future__ import annotations

import argparse

from .data import VALID_DATA_MUTATIONS, contaminate
from .eval import VALID_EVAL_MUTATIONS, contaminate_eval_signal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tasks.mutations")
    subparsers = parser.add_subparsers(dest="surface", required=True)

    data_parser = subparsers.add_parser("data")
    data_parser.add_argument("mutation", choices=sorted(VALID_DATA_MUTATIONS))
    data_parser.add_argument("workspace")
    data_parser.add_argument("--train-file", default="data/scifact.jsonl")
    data_parser.add_argument("--val-file", default="data/val.jsonl")
    data_parser.add_argument("--noise-rate", type=float, default=0.3)
    data_parser.add_argument("--leak-rate", type=float, default=0.2)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("mutation", choices=sorted(VALID_EVAL_MUTATIONS))
    eval_parser.add_argument("workspace")
    eval_parser.add_argument("--eval-file", default="data/val.jsonl")
    eval_parser.add_argument("--train-file", default="data/scifact.jsonl")
    eval_parser.add_argument("--leak-rate", type=float, default=0.25)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.surface == "data":
        contaminate(
            args.workspace,
            args.mutation,
            train_file=args.train_file,
            val_file=args.val_file,
            noise_rate=args.noise_rate,
            leak_rate=args.leak_rate,
        )
        return

    contaminate_eval_signal(
        args.workspace,
        args.mutation,
        eval_file=args.eval_file,
        train_file=args.train_file,
        leak_rate=args.leak_rate,
    )


if __name__ == "__main__":
    main()
