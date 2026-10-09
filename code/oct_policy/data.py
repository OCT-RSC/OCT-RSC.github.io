import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CACHE_SCHEMA, canonical_json, config_digest, write_json_atomic
from .geometry import action_array, action_transform, recorded_steps, relative_tool_pose
from .representations import RepresentationProcessor


def trajectory_id(trajectory: Path, data_root: Path) -> str:
    return trajectory.resolve().relative_to(data_root.resolve()).as_posix()


def discover_trajectories(
    data_root: Path,
    material: str | None = None,
    include_prefixes: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    root = Path(data_root).resolve()
    prefixes = [prefix.replace("\\", "/").strip("/") for prefix in include_prefixes or []]
    search_roots = [root / prefix for prefix in prefixes] if prefixes else [root]
    trajectories = []
    for search_root in search_roots:
        for selection_path in search_root.rglob("target_selection.json"):
            if selection_path.parent.name != "predeform":
                continue
            trajectory = selection_path.parent.parent
            selection = json.loads(selection_path.read_text())
            if material is not None and selection["material"] != material:
                continue
            if (selection_path.parent / "ascans.npy").is_file() and recorded_steps(trajectory):
                trajectories.append(trajectory)
    return sorted(set(trajectories), key=lambda path: str(path.relative_to(root)))


def target_volume_path(trajectory: Path) -> Path:
    for path in (
        trajectory / "predeform" / "target_volume.npy",
        trajectory / "target_volume.npy",
        trajectory.parent / "predeform" / "target_volume.npy",
        trajectory.parent.parent / "predeform" / "target_volume.npy",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No target_volume.npy in {trajectory}")


def load_tool_trajectory(trajectory: Path) -> tuple[list[np.ndarray], np.ndarray]:
    records = [json.loads((step / "step_info.json").read_text()) for step in recorded_steps(trajectory)]
    actions = np.stack([action_array(record) for record in records]).astype(np.float32)
    poses = [np.asarray(record["tooltip_transform_4x4"], dtype=np.float64) for record in records]
    initial_pose = poses[0] @ np.linalg.inv(action_transform(actions[0]))
    return [initial_pose, *poses], actions


def split_digest(split: dict) -> str:
    value = {key: item for key, item in split.items() if key != "split_sha256"}
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def validate_split_payload(split: dict) -> None:
    if set(split["train"]) & set(split["validation"]):
        raise ValueError("Training and validation trajectories overlap")


def create_frozen_split(
    data_root: Path,
    material: str,
    output: Path,
    validation_fraction: float = 0.2,
    seed: int = 42,
    include_prefixes: list[str] | tuple[str, ...] | None = None,
) -> dict:
    root = Path(data_root).resolve()
    grouped = defaultdict(list)
    for trajectory in discover_trajectories(root, material, include_prefixes):
        grouped[trajectory.parent.name].append(trajectory_id(trajectory, root))
    if not grouped:
        raise ValueError(f"No {material} demonstrations found in {root}")
    rng = random.Random(int(seed))
    train_ids, validation_ids, groups = [], [], {}
    for group, identifiers in sorted(grouped.items()):
        shuffled = sorted(identifiers)
        rng.shuffle(shuffled)
        count = max(1, int(round(len(shuffled) * validation_fraction)))
        train = sorted(shuffled[count:])
        validation = sorted(shuffled[:count])
        train_ids.extend(train)
        validation_ids.extend(validation)
        groups[group] = {"train": train, "validation": validation}
    split = {
        "schema": "trajectory_split_v1",
        "data_root": str(root),
        "material": material,
        "split_seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "include_prefixes": list(include_prefixes or []),
        "train": sorted(train_ids),
        "validation": sorted(validation_ids),
        "groups": groups,
    }
    split["split_sha256"] = split_digest(split)
    write_json_atomic(output, split)
    return split


def representation_cache_spec(config: dict) -> dict:
    names = (
        "material", "representation", "dense_resolution", "sparse_resolution",
        "point_count", "tissue_threshold", "air_sigma_multiplier",
        "maximum_visible_tissue_depth_mm", "mask_top_margin", "mask_bottom_margin",
        "mask_gap_fill_voxels", "minimum_visible_tissue_depth_voxels",
        "target_representation_source", "representation_seed",
    )
    return {name: config[name] for name in names if name in config}


def build_cache(
    data_root: Path,
    split_path: Path,
    config: dict,
    output_root: Path,
    overwrite: bool = False,
    device: str | None = None,
) -> Path:
    root = Path(data_root).resolve()
    split = json.loads(split_path.read_text())
    specification = representation_cache_spec(config)
    cache_specification = dict(specification)
    if config["representation"] == "full_volume" and device is not None:
        cache_specification["preprocessing_device"] = str(torch.device(device))
    digest = config_digest(specification)
    cache_digest = config_digest(cache_specification)
    key = f"{config['material']}__{config['representation']}__{cache_digest}"
    cache = Path(output_root).resolve() / key
    cache.mkdir(parents=True, exist_ok=True)
    extension = ".npz" if config["representation"] == "tissue_only" else ".npy"
    manifest = {
        "schema": CACHE_SCHEMA,
        "cache_key": key,
        "representation_config": specification,
        "representation_config_digest": digest,
        "split_sha256": split["split_sha256"],
        "material": config["material"],
        "representation": config["representation"],
        "trajectories": [],
    }
    if config["representation"] == "full_volume":
        manifest["preprocessing_device"] = cache_specification.get("preprocessing_device", "auto")
    for identifier in sorted(set(split["train"]) | set(split["validation"])):
        trajectory = root / identifier
        processor = RepresentationProcessor(trajectory, config)
        if device is not None:
            processor.device = torch.device(device)
        folder = cache / hashlib.sha256(identifier.encode()).hexdigest()[:12]
        folder.mkdir(exist_ok=True)
        sources = [trajectory / "predeform" / "ascans.npy"] + [
            step / "ascans.npy" for step in recorded_steps(trajectory)
        ]
        observations = []
        for state, source in enumerate(sources):
            path = folder / f"observation_{state:03d}{extension}"
            if overwrite or not path.is_file():
                processor.save_representation(source, path, is_target=False, state=state)
            observations.append(path.relative_to(cache).as_posix())
        target = folder / f"target{extension}"
        if overwrite or not target.is_file():
            processor.save_representation(
                target_volume_path(trajectory), target, is_target=True, state=-1
            )
        poses, actions = load_tool_trajectory(trajectory)
        scale = np.asarray(config["tool_scale"], dtype=np.float32)
        manifest["trajectories"].append({
            "id": identifier,
            "group": trajectory.parent.name,
            "observations": observations,
            "target": target.relative_to(cache).as_posix(),
            "tool_poses": [(relative_tool_pose(poses[0], pose) / scale).tolist() for pose in poses],
            "actions": actions.tolist(),
        })
        print(f"Cached {identifier}: {len(actions)} actions", flush=True)
    manifest["cache_sha256"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    path = cache / "manifest.json"
    write_json_atomic(path, manifest)
    return path


def _load_representation(path: Path, representation: str):
    if representation != "tissue_only":
        return torch.from_numpy(np.asarray(np.load(path), dtype=np.float32))
    with np.load(path) as saved:
        return {
            "coordinates": torch.from_numpy(saved["coordinates"].astype(np.int32)),
            "features": torch.from_numpy(saved["features"].astype(np.float32)),
        }


class PolicyDataset(Dataset):
    def __init__(self, manifest_path: Path, trajectory_ids: set[str], config: dict):
        manifest_path = Path(manifest_path).resolve()
        self.directory = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        if manifest["representation_config_digest"] != config_digest(representation_cache_spec(config)):
            raise ValueError("Prepared representation does not match the model configuration")
        self.cache_sha256 = manifest["cache_sha256"]
        self.representation = config["representation"]
        self.history = int(config["observation_window"])
        self.horizon = int(config["prediction_window"])
        self.action_scale = torch.tensor(config["action_scale"], dtype=torch.float32)
        self.trajectories = {
            item["id"]: item for item in manifest["trajectories"] if item["id"] in trajectory_ids
        }
        if set(self.trajectories) != trajectory_ids:
            raise ValueError("Prepared dataset is missing requested trajectories")
        self.loaded = {}
        self.samples = [
            (identifier, state)
            for identifier, trajectory in self.trajectories.items()
            for state in range(len(trajectory["actions"]))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str):
        if path not in self.loaded:
            self.loaded[path] = _load_representation(self.directory / path, self.representation)
        return self.loaded[path]

    def __getitem__(self, index: int) -> dict:
        identifier, state = self.samples[index]
        trajectory = self.trajectories[identifier]
        history = [max(0, state - self.history + 1 + offset) for offset in range(self.history)]
        observations = [self._load(trajectory["observations"][item]) for item in history]
        tool_poses = torch.tensor([trajectory["tool_poses"][item] for item in history], dtype=torch.float32)
        actions = torch.zeros(self.horizon, 6, dtype=torch.float32)
        mask = torch.zeros(self.horizon, dtype=torch.float32)
        future = trajectory["actions"][state:state + self.horizon]
        if future:
            actions[:len(future)] = torch.tensor(future, dtype=torch.float32) / self.action_scale
            mask[:len(future)] = 1.0
        return {
            "observations": observations,
            "target": self._load(trajectory["target"]),
            "tool_poses": tool_poses,
            "actions": actions,
            "mask": mask,
            "trajectory_id": identifier,
            "state": state,
        }


def collate_policy_samples(samples: list[dict]) -> dict:
    if isinstance(samples[0]["target"], dict):
        observations = [sample["observations"] for sample in samples]
        target = [sample["target"] for sample in samples]
    else:
        observations = torch.stack([torch.stack(sample["observations"]) for sample in samples])
        target = torch.stack([sample["target"] for sample in samples])
    return {
        "observations": observations,
        "target": target,
        "tool_poses": torch.stack([sample["tool_poses"] for sample in samples]),
        "actions": torch.stack([sample["actions"] for sample in samples]),
        "mask": torch.stack([sample["mask"] for sample in samples]),
        "trajectory_id": [sample["trajectory_id"] for sample in samples],
        "state": torch.tensor([sample["state"] for sample in samples]),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
