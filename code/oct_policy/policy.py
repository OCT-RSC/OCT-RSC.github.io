from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import torch_load
from .encoders import CosineSchedule
from .geometry import relative_tool_pose
from .models import build_policy, predict_actions
from .preprocessing import trajectory_from_target
from .representations import RepresentationProcessor


class UnifiedPolicyInterface:
    def __init__(self, checkpoint: Path, device: str = "cuda"):
        self.checkpoint = Path(checkpoint)
        saved = torch_load(self.checkpoint, map_location="cpu")
        self.config = saved["config"]
        self.device = torch.device(device)
        self.policy = build_policy(self.config).to(self.device).eval()
        self.policy.load_state_dict(saved["model_ema"])
        self.schedule = (
            CosineSchedule(int(self.config.get("diffusion_steps", 100)), self.device)
            if self.config["head"] == "diffusion"
            else None
        )
        self.action_scale = np.asarray(self.config["action_scale"], dtype=np.float32)
        self.tool_scale = np.asarray(self.config["tool_scale"], dtype=np.float32)
        self.processor: RepresentationProcessor | None = None
        self.target = None
        self.initial_pose = None
        self.observations = []
        self.tool_poses = []
        self.generator = torch.Generator(device=self.device)

    def _runtime_representation(self, value):
        if self.config["representation"] != "tissue_only":
            return torch.as_tensor(value, dtype=torch.float32)
        coordinates = torch.as_tensor(value["coordinates"], dtype=torch.int32)
        features = torch.as_tensor(value["features"], dtype=torch.float32)
        return {
            "coordinates": coordinates.contiguous(),
            "features": features.contiguous(),
        }

    def reset(self, target_volume: Path, initial_tooltip_pose: np.ndarray) -> None:
        target_volume = Path(target_volume).resolve()
        trajectory = trajectory_from_target(target_volume)
        self.processor = RepresentationProcessor(trajectory, self.config)
        self.target = self._runtime_representation(
            self.processor.make_representation(
                target_volume, is_target=True, state=-1
            )
        )
        self.initial_pose = np.asarray(initial_tooltip_pose, dtype=np.float64)
        self.observations.clear()
        self.tool_poses.clear()
        self.generator.manual_seed(int(self.config.get("inference_seed", 42)))

    @staticmethod
    def _repeat_oldest(values: list, length: int) -> list:
        values = values[-length:]
        return [values[0]] * max(0, length - len(values)) + values

    def _batch(self) -> dict:
        history = int(self.config["observation_window"])
        observations = self._repeat_oldest(self.observations, history)
        poses = self._repeat_oldest(self.tool_poses, history)
        if self.config["representation"] == "tissue_only":
            observation_batch = [observations]
            target_batch = [self.target]
        else:
            observation_batch = torch.stack(observations)[None].to(self.device)
            target_batch = torch.as_tensor(self.target, dtype=torch.float32)[None].to(self.device)
        return {
            "observations": observation_batch,
            "target": target_batch,
            "tool_poses": torch.stack(poses)[None].to(self.device),
        }

    @torch.inference_mode()
    def step(self, scanned_oct: Path, tooltip_pose: np.ndarray) -> dict:
        if self.processor is None or self.target is None or self.initial_pose is None:
            raise RuntimeError("Call reset() before step()")
        started = time.perf_counter()
        representation = self._runtime_representation(
            self.processor.make_representation(
                Path(scanned_oct), is_target=False, state=len(self.observations)
            )
        )
        pose = relative_tool_pose(
            self.initial_pose, np.asarray(tooltip_pose, dtype=np.float64)
        )
        self.observations.append(representation)
        self.tool_poses.append(torch.from_numpy(pose / self.tool_scale).float())
        batch = self._batch()
        model_started = time.perf_counter()
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            prediction = predict_actions(
                self.policy,
                self.schedule,
                batch,
                int(self.config.get("inference_steps", 20)),
                self.generator,
            )[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        model_latency_ms = (time.perf_counter() - model_started) * 1000.0
        physical = prediction.float().cpu().numpy() * self.action_scale
        execute_count = int(self.config["execute_window"])
        return {
            "objective": (
                "diffusion_clean_action"
                if self.config["head"] == "diffusion"
                else "deterministic_action_chunk"
            ),
            "representation": self.config["representation"],
            "use_residual": bool(self.config["use_residual"]),
            "action_6dof_mm_deg": physical[0],
            "executed_action_sequence_6dof_mm_deg": physical[:execute_count],
            "predicted_action_sequence_6dof_mm_deg": physical,
            "model_latency_ms": model_latency_ms,
            "total_latency_ms": (time.perf_counter() - started) * 1000.0,
        }


def load_policy(checkpoint: str | Path, device: str | None = None) -> UnifiedPolicyInterface:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return UnifiedPolicyInterface(Path(checkpoint), device=device)
