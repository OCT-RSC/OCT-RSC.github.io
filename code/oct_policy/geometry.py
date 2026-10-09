from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from .preprocessing import (
    FloorAwareVolumePreprocessor,
    open_raw_hwd,
)


OCT_MEAN = 10.377689
OCT_STD = 6.639139


def recorded_steps(trajectory: Path) -> list[Path]:
    return sorted(
        [path for path in Path(trajectory).iterdir()
         if path.is_dir() and path.name.removeprefix("rollout_step_").isdigit()],
        key=lambda path: int(path.name.removeprefix("rollout_step_")),
    )


def load_raw_volume(path: Path) -> np.ndarray:
    raw = open_raw_hwd(path)
    return np.ascontiguousarray(raw.transpose(2, 0, 1)[:, :, ::-1])


def preprocess_volume(
    path: Path,
    resolution: int,
    device: torch.device | str,
    preprocessor: FloorAwareVolumePreprocessor | None = None,
    is_target: bool = False,
) -> np.ndarray:
    if preprocessor is not None:
        return preprocessor.preprocess(path, resolution, device, is_target=is_target)
    volume = torch.from_numpy(load_raw_volume(path)).to(device=device, dtype=torch.float32)
    volume = F.interpolate(
        volume[None, None],
        size=(resolution, resolution, resolution),
        mode="trilinear",
        align_corners=True,
    )[0, 0]
    volume = ((volume - OCT_MEAN) / OCT_STD).clamp(-5, 5)
    return volume.to(torch.float16).cpu().numpy()


def action_array(record: dict) -> np.ndarray:
    values = record["inputs_6dof_mm_deg"]
    return np.asarray(
        [
            values["dx_mm"],
            values["dy_mm"],
            values["dz_mm"],
            values["d_roll_deg"],
            values["d_pitch_deg"],
            values["d_yaw_deg"],
        ],
        dtype=np.float32,
    )


def action_transform(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", action[3:], degrees=True).as_matrix()
    transform[:3, 3] = action[:3] / 1000.0
    return transform


def relative_tool_pose(initial_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    relative = np.linalg.inv(initial_pose) @ current_pose
    translation_mm = relative[:3, 3] * 1000.0
    rotation_deg = np.rad2deg(Rotation.from_matrix(relative[:3, :3]).as_rotvec())
    return np.concatenate((translation_mm, rotation_deg)).astype(np.float32)
