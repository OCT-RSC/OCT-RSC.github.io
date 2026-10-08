"""Predict actions from every saved OCT state in one trajectory; no robot commands are sent."""

import argparse
from pathlib import Path

import numpy as np

from oct_policy import load_policy
from oct_policy.config import write_json_atomic
from oct_policy.data import load_tool_trajectory, target_volume_path
from oct_policy.geometry import recorded_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    trajectory = args.trajectory.resolve()
    target = target_volume_path(trajectory)
    predeform = target.parent if target.parent.name == "predeform" else trajectory / "predeform"
    poses, _ = load_tool_trajectory(trajectory)
    scans = [predeform / "ascans.npy"] + [
        step / "ascans.npy" for step in recorded_steps(trajectory)
    ]
    policy = load_policy(args.checkpoint, args.device)
    if args.seed is not None:
        policy.config["inference_seed"] = args.seed
    policy.reset(target, poses[0])
    predictions = []
    for state, (scan, pose) in enumerate(zip(scans, poses)):
        result = policy.step(scan, pose)
        predictions.append({
            "state": state,
            **{key: value.tolist() if isinstance(value, np.ndarray) else value
               for key, value in result.items()},
        })
        print(f"State {state}: {result['action_6dof_mm_deg']}", flush=True)
    write_json_atomic(args.output, {"predictions": predictions})


if __name__ == "__main__":
    main()
