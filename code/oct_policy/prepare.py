import json
from pathlib import Path

from .config import write_json_atomic
from .data import (
    build_cache,
    create_frozen_split,
)


def open_prepared(directory: str | Path) -> tuple[Path, Path]:
    root = Path(directory).resolve()
    metadata = json.loads((root / "prepared.json").read_text())
    manifest_path = root / metadata["manifest"]
    split_path = root / metadata["split"]
    return manifest_path, split_path


def prepare_dataset(
    data_root: Path,
    config: dict,
    output: Path,
    data_profile: Path | None = None,
    overwrite: bool = False,
    device: str | None = None,
) -> Path:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = json.loads(data_profile.read_text()) if data_profile else {}
    prefixes = profile.get("include_prefixes", {}).get(config["material"])
    if data_profile and prefixes is None:
        raise ValueError(f"Data profile has no {config['material']} trajectories")
    split_path = output / "split.json"
    split = create_frozen_split(
        data_root, config["material"], split_path,
        seed=config["split_seed"], include_prefixes=prefixes,
    )
    if not split["train"] or not split["validation"]:
        raise ValueError("Need at least two demonstrations per target condition")
    counts = {"train": len(split["train"]), "validation": len(split["validation"])}
    manifest = build_cache(
        data_root, split_path, config, output / "cache", overwrite, device=device
    )
    cached = json.loads(manifest.read_text())
    actions = sum(len(trajectory["actions"]) for trajectory in cached["trajectories"])
    write_json_atomic(output / "prepared.json", {
        "schema": "oct_policy_prepared_data_v1",
        "manifest": manifest.relative_to(output).as_posix(),
        "split": "split.json",
        "material": config["material"],
        "representation": config["representation"],
        "cache_sha256": cached["cache_sha256"],
        "split_sha256": split["split_sha256"],
        "train_trajectories": counts["train"],
        "validation_trajectories": counts["validation"],
        "actions": actions,
    })
    return output
