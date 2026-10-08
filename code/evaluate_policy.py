"""Evaluate a checkpoint on its held-out demonstration trajectories."""

import argparse
from pathlib import Path

from oct_policy.prepare import open_prepared
from oct_policy.evaluation import export_offline_predictions, validate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="Also save per-state action chunks")
    args = parser.parse_args()
    manifest, split = open_prepared(args.prepared)
    result = validate_checkpoint(args.checkpoint, manifest, split, args.output)
    if args.predictions:
        export_offline_predictions(args.checkpoint, manifest, split, args.predictions)
    print(f"Validation score: {result['val_selection_score']:.6f}")


if __name__ == "__main__":
    main()
