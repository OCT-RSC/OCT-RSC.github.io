from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


EXPERIMENT_SCHEMA = "oct_policy_three_input_v1"
CACHE_SCHEMA = "oct_policy_representation_cache_v2"
REPRESENTATIONS = ("full_volume", "tissue_only", "point_cloud")
HEADS = ("diffusion", "deterministic_bc")
MATERIALS = ("phantom", "exvivo")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def config_digest(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]


def validate_config(config: dict) -> dict:
    config = dict(config)
    if config.get("schema") != EXPERIMENT_SCHEMA:
        raise ValueError(f"Expected schema {EXPERIMENT_SCHEMA!r}")
    choices_by_name = {
        "material": MATERIALS,
        "representation": REPRESENTATIONS,
        "head": HEADS,
    }
    for name, choices in choices_by_name.items():
        if config[name] not in choices:
            raise ValueError(f"{name} must be one of {choices}")
    for name in ("observation_window", "prediction_window", "execute_window"):
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(config["execute_window"]) > int(config["prediction_window"]):
        raise ValueError("execute_window cannot exceed prediction_window")
    for name in ("action_scale", "tool_scale"):
        values = config[name]
        if len(values) != 6 or any(float(value) <= 0 for value in values):
            raise ValueError(f"{name} must contain six positive values")
    return config


def _load_config_dict(path: Path) -> dict:
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    base_name = value.pop("extends", None)
    if base_name is not None:
        base = _load_config_dict(path.parent / base_name)
        base.update(value)
        value = base
    return value


def load_config(path: Path) -> dict:
    path = Path(path).resolve()
    value = validate_config(_load_config_dict(path))
    value["config_path"] = str(path)
    value["config_digest"] = config_digest(
        {key: item for key, item in value.items() if key not in ("config_path", "config_digest")}
    )
    return value


def write_json_atomic(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def torch_load(path: Path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)
