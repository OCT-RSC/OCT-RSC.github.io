"""Train an OCT policy and save the best and last EMA checkpoints."""

import argparse
from pathlib import Path

from oct_policy.config import config_digest, load_config
from oct_policy.prepare import open_prepared
from oct_policy.training import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="Override the 100-epoch limit")
    parser.add_argument("--fresh", action="store_true", help="Do not resume an existing run")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        config["maximum_epochs"] = args.epochs
        config["minimum_epochs"] = min(config["minimum_epochs"], args.epochs)
        config["config_digest"] = config_digest({
            key: value for key, value in config.items()
            if key not in ("config_path", "config_digest")
        })
    manifest, split = open_prepared(args.prepared)
    result = train_experiment(manifest, split, config, args.output, resume=not args.fresh)
    print(f"Best epoch: {result['best_epoch']}; validation score: {result['best_validation_score']:.6f}")


if __name__ == "__main__":
    main()
