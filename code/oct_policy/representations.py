from __future__ import annotations


import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .segmentation import (
    adaptive_upper_surface,
    measured_air_statistics,
)
from .geometry import preprocess_volume
from .preprocessing import (
    FLOOR_REMOVED_FULL,
    OCT_MEAN,
    OCT_STD,
    FloorAwareVolumePreprocessor,
    open_raw_hwd,
)


def _dhw_to_xyz(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError(f"Expected a final DHW axis of length 3, got {values.shape}")
    return np.take(values, (2, 1, 0), axis=-1)


class RepresentationProcessor:

    def __init__(self, trajectory: Path, config: dict):
        self.trajectory = Path(trajectory).resolve()
        self.predeform = self.trajectory / "predeform"
        self.config = config
        self.representation = config["representation"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.full_preprocessor = FloorAwareVolumePreprocessor(
            self.trajectory, FLOOR_REMOVED_FULL
        )
        self.selection = json.loads(
            (self.predeform / "target_selection.json").read_text(encoding="utf-8")
        )
        self.pre_raw = np.asarray(open_raw_hwd(self.predeform / "ascans.npy"))

    def _clean(self, raw_hwd: np.ndarray) -> np.ndarray:
        return np.asarray(self.full_preprocessor._clean_raw(raw_hwd))

    def _complete_tissue_mask(self, clean_hwd: np.ndarray) -> np.ndarray:
        surface, _, reliable = adaptive_upper_surface(clean_hwd)
        support = ndimage.binary_fill_holes(reliable)
        support = ndimage.binary_closing(support, iterations=2)
        air = measured_air_statistics(clean_hwd, surface, reliable)
        threshold = max(
            float(self.config.get("tissue_threshold", 20.0)),
            air["median"] + float(self.config.get("air_sigma_multiplier", 3.0)) * max(air["std"], 1.0),
        )
        smoothed = ndimage.gaussian_filter(
            np.asarray(clean_hwd, dtype=np.float32),
            sigma=(0.6, 0.6, 1.0),
            mode="nearest",
        )
        depth = clean_hwd.shape[2]
        depth_index = np.arange(depth)[None, None, :]
        start = np.floor(surface).astype(np.int32) - int(self.config.get("mask_top_margin", 2))
        maximum_depth_voxels = int(
            round(
                float(self.config.get("maximum_visible_tissue_depth_mm", 4.0))
                / float(self.selection.get("depth_spacing_mm", 0.01337))
            )
        )
        stop_limit = np.minimum(depth - 1, np.ceil(surface).astype(np.int32) + maximum_depth_voxels)
        candidate = (
            support[:, :, None]
            & (smoothed >= threshold)
            & (depth_index >= start[:, :, None])
            & (depth_index <= stop_limit[:, :, None])
        )
        candidate = ndimage.binary_closing(
            candidate,
            structure=np.ones((1, 1, int(self.config.get("mask_gap_fill_voxels", 7))), dtype=bool),
        )
        reverse = np.argmax(candidate[:, :, ::-1], axis=2)
        has_candidate = candidate.any(axis=2)
        last = depth - 1 - reverse
        minimum_stop = np.ceil(surface).astype(np.int32) + int(
            self.config.get("minimum_visible_tissue_depth_voxels", 6)
        )
        last = np.where(has_candidate, last, minimum_stop)
        last = ndimage.median_filter(last.astype(np.float32), size=5, mode="nearest")
        last = np.minimum(last + int(self.config.get("mask_bottom_margin", 2)), stop_limit)
        mask = (
            support[:, :, None]
            & (depth_index >= start[:, :, None])
            & (depth_index <= last[:, :, None])
        )
        return np.ascontiguousarray(mask, dtype=bool)


    @staticmethod
    def _dhw(array_hwd: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(array_hwd.transpose(2, 0, 1)[:, :, ::-1])

    def _resampled_tissue(
        self, path: Path, is_target: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(open_raw_hwd(path))
        clean = self._clean(raw)
        mask = self._complete_tissue_mask(clean)
        resolution = tuple(int(value) for value in self.config.get("sparse_resolution", (96, 96, 96)))
        normalized = np.clip((clean.astype(np.float32) - OCT_MEAN) / OCT_STD, -5.0, 5.0)
        volume = torch.from_numpy(self._dhw(normalized))[None, None]
        mask_tensor = torch.from_numpy(self._dhw(mask).copy()).float()[None, None]
        volume = F.interpolate(volume, size=resolution, mode="trilinear", align_corners=True)[0, 0]
        mask_tensor = F.interpolate(mask_tensor, size=resolution, mode="nearest")[0, 0] >= 0.5
        return volume.numpy(), mask_tensor.numpy()

    def make_representation(self, path: Path, is_target: bool, state: int):
        path = Path(path)
        if self.representation == "full_volume":
            resolution = tuple(int(value) for value in self.config["dense_resolution"])
            if len(set(resolution)) != 1:
                raise ValueError("Existing full-volume preprocessor requires an isotropic output grid")
            return preprocess_volume(
                path,
                resolution[0],
                self.device,
                preprocessor=self.full_preprocessor,
                is_target=is_target,
            )

        values, mask = self._resampled_tissue(path, is_target)
        coordinates = np.argwhere(mask).astype(np.int32)
        if not len(coordinates):
            raise RuntimeError(f"No tissue voxels survived preprocessing: {path}")
        if self.representation == "tissue_only":
            features = values[tuple(coordinates.T)][:, None].astype(np.float32)
            return {"coordinates": coordinates, "features": features}

        point_count = int(self.config.get("point_count", 1024))
        seed = int(self.config.get("representation_seed", 42)) + max(state, 0)
        rng = np.random.default_rng(seed)
        selected = rng.choice(
            len(coordinates),
            size=point_count,
            replace=len(coordinates) < point_count,
        )
        points_dhw = coordinates[selected].astype(np.float32)
        shape_dhw = np.asarray(mask.shape, dtype=np.float32)
        lateral_fov = float(self.selection.get("lateral_fov_mm", 18.0))
        depth_extent = (self.pre_raw.shape[2] - 1) * float(
            self.selection.get("depth_spacing_mm", 0.01337)
        )
        points_xyz = _dhw_to_xyz(points_dhw)
        shape_xyz = _dhw_to_xyz(shape_dhw)
        extent_xyz_mm = np.asarray(
            (lateral_fov, lateral_fov, depth_extent), dtype=np.float32
        )
        points_mm = points_xyz / np.maximum(shape_xyz - 1.0, 1.0) * extent_xyz_mm
        points = (points_mm - 0.5 * extent_xyz_mm) / (0.5 * extent_xyz_mm)
        return points.astype(np.float32)

    def save_representation(
        self, path: Path, destination: Path, is_target: bool, state: int
    ) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        value = self.make_representation(path, is_target=is_target, state=state)
        temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
        if isinstance(value, dict):
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **value)
        else:
            with temporary.open("wb") as stream:
                np.save(stream, value)
        temporary.replace(destination)
