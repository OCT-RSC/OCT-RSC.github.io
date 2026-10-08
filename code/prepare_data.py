"""Preprocess OCT demonstrations and create a trajectory-level split."""

import argparse
from pathlib import Path

from oct_policy.config import load_config
from oct_policy.prepare import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-profile", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = prepare_dataset(
        args.data_root, load_config(args.config), args.output,
        args.data_profile, args.overwrite, device=args.device,
    )
    print(f"Prepared dataset: {output}")


if __name__ == "__main__":
    main()
