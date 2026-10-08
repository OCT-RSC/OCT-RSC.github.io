import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import torch_load, write_json_atomic
from .data import (
    PolicyDataset,
    collate_policy_samples,
    move_batch,
    validate_split_payload,
)
from .encoders import CosineSchedule, DIFFUSION_STEPS
from .models import build_policy, predict_actions


def masked_mse(
    prediction: torch.Tensor, action: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    error = F.mse_loss(prediction, action, reduction="none").mean(dim=-1)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


@torch.inference_mode()
def evaluate_model(
    policy, config: dict, loader: DataLoader, device: torch.device
) -> dict:
    policy.eval()
    schedule = (
        CosineSchedule(int(config.get("diffusion_steps", DIFFUSION_STEPS)), device)
        if config["head"] == "diffusion"
        else None
    )
    generator = torch.Generator(device=device).manual_seed(
        int(config.get("validation_seed", 42))
    )
    action_scale = torch.tensor(config["action_scale"], device=device)
    by_trajectory = defaultdict(list)
    first_translation = []
    first_rotation = []
    masked_error_sum = 0.0
    masked_valid_steps = 0.0
    latencies = []
    latency_samples = 0
    horizon_translation_error_sum = 0.0
    horizon_rotation_error_sum = 0.0
    horizon_component_count = 0.0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            prediction = predict_actions(
                policy, schedule, batch, int(config.get("inference_steps", 20)), generator
            )
            loss = masked_mse(prediction, batch["actions"], batch["mask"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
        latency_samples += int(batch["actions"].shape[0])
        valid_steps = float(batch["mask"].sum().item())
        masked_error_sum += float(loss.item()) * valid_steps
        masked_valid_steps += valid_steps
        normalized_squared = (
            (prediction[:, 0] - batch["actions"][:, 0]).square().mean(dim=-1)
        )
        for identifier, value in zip(
            raw_batch["trajectory_id"], normalized_squared.cpu().tolist()
        ):
            by_trajectory[identifier].append(float(value))
        physical_error = (prediction - batch["actions"]) * action_scale
        first_translation.extend(
            physical_error[:, 0, :3].square().mean(dim=-1).cpu().tolist()
        )
        first_rotation.extend(
            physical_error[:, 0, 3:].square().mean(dim=-1).cpu().tolist()
        )
        valid = batch["mask"][:, :, None]
        horizon_translation_error_sum += float(
            (physical_error[..., :3].square() * valid).sum().item()
        )
        horizon_rotation_error_sum += float(
            (physical_error[..., 3:].square() * valid).sum().item()
        )
        horizon_component_count += valid_steps * 3.0
    trajectory_scores = {
        identifier: float(np.mean(values)) for identifier, values in by_trajectory.items()
    }
    return {
        "val_selection_score": float(np.mean(list(trajectory_scores.values()))),
        "val_selection_definition": "first-action normalized MSE, mean per trajectory then macro mean",
        "trajectory_first_action_normalized_mse": trajectory_scores,
        "masked_horizon_normalized_mse": masked_error_sum / max(masked_valid_steps, 1.0),
        "first_action_translation_rmse_mm": float(np.sqrt(np.mean(first_translation))),
        "first_action_rotation_rmse_deg": float(np.sqrt(np.mean(first_rotation))),
        "horizon_translation_rmse_mm": float(
            np.sqrt(horizon_translation_error_sum / max(horizon_component_count, 1.0))
        ),
        "horizon_rotation_rmse_deg": float(
            np.sqrt(horizon_rotation_error_sum / max(horizon_component_count, 1.0))
        ),
        "mean_inference_latency_ms": float(sum(latencies) / max(latency_samples, 1)),
        "inference_latency_definition": "model sampling latency per validation sample",
    }


def _load_validation(
    checkpoint: Path,
    manifest: Path,
    split_path: Path,
    batch_size: int | None = None,
):
    saved = torch_load(checkpoint)
    config = saved["config"]
    split = json.loads(split_path.read_text())
    validate_split_payload(split)
    if saved.get("split") and saved["split"]["validation"] != split["validation"]:
        raise ValueError("Checkpoint and prepared data use different validation trajectories")
    dataset = PolicyDataset(manifest, set(split["validation"]), config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or int(config["validation_batch_size"]),
        shuffle=False,
        collate_fn=collate_policy_samples,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_policy(config).to(device).eval()
    policy.load_state_dict(saved["model_ema"])
    return config, loader, policy, device


def validate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    split_path: Path,
    output_path: Path | None = None,
) -> dict:
    config, loader, policy, device = _load_validation(
        checkpoint_path, manifest_path, split_path
    )
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "weights": "ema",
        **evaluate_model(policy, config, loader, device),
    }
    if output_path is not None:
        write_json_atomic(output_path, result)
    return result


@torch.inference_mode()
def export_offline_predictions(
    checkpoint_path: Path,
    manifest_path: Path,
    split_path: Path,
    output_path: Path,
) -> dict:
    config, loader, policy, device = _load_validation(
        checkpoint_path, manifest_path, split_path, batch_size=1
    )
    schedule = (
        CosineSchedule(int(config["diffusion_steps"]), device)
        if config["head"] == "diffusion"
        else None
    )
    generator = torch.Generator(device=device).manual_seed(int(config["validation_seed"]))
    scale = torch.tensor(config["action_scale"], device=device)
    predictions = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            prediction = predict_actions(
                policy, schedule, batch, int(config["inference_steps"]), generator
            )
        predictions.append(
            {
                "trajectory_id": raw_batch["trajectory_id"][0],
                "state": int(raw_batch["state"][0]),
                "predicted_action_chunk_mm_deg": (prediction[0] * scale).cpu().tolist(),
                "predicted_normalized_action_chunk": prediction[0].cpu().tolist(),
                "recorded_action_chunk_mm_deg": (batch["actions"][0] * scale).cpu().tolist(),
                "valid_action_mask": batch["mask"][0].cpu().tolist(),
            }
        )
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "evaluation_weights": "ema",
        "output_shape_per_state": [int(config["prediction_window"]), 6],
        "predictions": predictions,
    }
    write_json_atomic(output_path, result)
    return result
