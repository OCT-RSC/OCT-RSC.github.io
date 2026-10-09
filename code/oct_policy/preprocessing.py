
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .segmentation import (
    SURFACE_MARGIN as ADAPTIVE_SURFACE_MARGIN,
    adaptive_upper_surface,
    oct_artifact_mask,
    open_raw_hwd as open_saved_oct,
    replace_artifacts_with_measured_air,
)


TISSUE_THRESHOLD = 25
SURFACE_MARGIN = ADAPTIVE_SURFACE_MARGIN
BAND_BEFORE_SURFACE = 2
BAND_AFTER_SURFACE = 24
DEPTH_SPACING_MM = 0.01337
OCT_MEAN = 10.377689
OCT_STD = 6.639139

RAW_FULL = "raw_full"
FLOOR_REMOVED_FULL = "floor_removed_full"
TISSUE_ONLY = "tissue_only"
VOLUME_INPUT_MODES = (RAW_FULL, FLOOR_REMOVED_FULL, TISSUE_ONLY)


def open_raw_hwd(path: Path) -> np.ndarray:
    return open_saved_oct(Path(path))[0]


def extract_tissue_surface(
    path: Path,
    tissue_support_yx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = open_raw_hwd(path)
    surface, _, reliable = adaptive_upper_surface(raw)
    reliable &= tissue_support_yx
    return surface.astype(np.float32), reliable


def _selected_displacement(predeform: Path) -> np.ndarray:
    selection_path = predeform / "target_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    relative_path = selection.get("displacement_path")
    if relative_path:
        path = predeform / Path(str(relative_path).replace("\\", "/"))
        if not path.exists():
            raise FileNotFoundError(f"Selected target displacement is missing: {path}")
        return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    percentage = int(selection["deformation_percent"])
    path = (
        predeform
        / "targets"
        / f"deformation_{percentage}pct"
        / "gaussian_displacement_yx_mm.npy"
    )
    if not path.exists():
        raise FileNotFoundError(f"Selected target displacement is missing: {path}")
    return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)


class FloorAwareVolumePreprocessor:

    def __init__(self, trajectory: Path, mode: str):
        if mode not in VOLUME_INPUT_MODES:
            raise ValueError(f"Unknown volume input mode {mode!r}")
        self.trajectory = Path(trajectory)
        self.predeform = self.trajectory / "predeform"
        self.mode = mode
        self.air_cleaned = False
        self.artifact_mask_hwd = None
        self.reference_clean_hwd = None

        if mode == RAW_FULL:
            self.tissue_support_yx = None
            self.pre_surface_yx = None
            self.pre_mask_yx = None
            self.target_surface_yx = None
            return

        saved_surface_path = self.predeform / "targets" / "pre_surface_yx_vox.npy"
        if not saved_surface_path.exists():
            raise FileNotFoundError(
                f"Floor removal requires the saved predeform surface: {saved_surface_path}"
            )
        saved_surface = np.load(saved_surface_path, mmap_mode="r")
        displacement = _selected_displacement(self.predeform)
        selection_path = self.predeform / "target_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        artifact_processing = selection.get("artifact_processing", {})
        self.air_cleaned = artifact_processing.get("version") == "geometric_measured_air_v1"

        if self.air_cleaned:
            reference_raw = open_raw_hwd(self.predeform / "ascans.npy")
            initial_surface, _, initial_reliable = adaptive_upper_surface(reference_raw)
            self.artifact_mask_hwd, _ = oct_artifact_mask(
                reference_raw,
                selection["material"],
                bright_threshold=float(artifact_processing["bright_threshold"]),
                occupancy_threshold=float(artifact_processing["occupancy_threshold"]),
            )
            self.reference_clean_hwd = replace_artifacts_with_measured_air(
                reference_raw,
                reference_raw,
                initial_surface,
                initial_reliable,
                self.artifact_mask_hwd,
            )
            self.pre_surface_yx, _, self.pre_mask_yx = adaptive_upper_surface(
                self.reference_clean_hwd
            )
            self.pre_mask_yx = ndimage.binary_fill_holes(self.pre_mask_yx)
            self.tissue_support_yx = np.asarray(self.pre_mask_yx, dtype=bool)
        else:
            visible = displacement >= max(0.05, 0.05 * float(np.max(displacement)))
            tissue_depth_limit = float(np.percentile(saved_surface[visible], 99.9)) + 25.0
            self.tissue_support_yx = np.asarray(
                saved_surface < tissue_depth_limit,
                dtype=bool,
            )
            self.pre_surface_yx, self.pre_mask_yx = extract_tissue_surface(
                self.predeform / "ascans.npy",
                self.tissue_support_yx,
            )
        self.target_surface_yx = (
            self.pre_surface_yx + displacement / DEPTH_SPACING_MM
        ).astype(np.float32)

    def _surface(
        self,
        path: Path,
        raw_hwd: np.ndarray,
        is_target: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == RAW_FULL:
            raise RuntimeError("Raw volumes do not require a tissue surface")
        if is_target:
            return self.target_surface_yx, self.pre_mask_yx
        if Path(path).resolve() == (self.predeform / "ascans.npy").resolve():
            return self.pre_surface_yx, self.pre_mask_yx
        if self.air_cleaned:
            surface, _, reliable = adaptive_upper_surface(raw_hwd)
            reliable = ndimage.binary_fill_holes(reliable)
            return surface.astype(np.float32), reliable & self.tissue_support_yx
        return extract_tissue_surface(path, self.tissue_support_yx)

    def _clean_raw(self, raw_hwd: np.ndarray) -> np.ndarray:
        if not self.air_cleaned:
            return raw_hwd
        return replace_artifacts_with_measured_air(
            raw_hwd,
            self.reference_clean_hwd,
            self.pre_surface_yx,
            self.pre_mask_yx,
            self.artifact_mask_hwd,
        )

    @staticmethod
    def _filled_surface(surface_yx: np.ndarray, mask_yx: np.ndarray) -> np.ndarray:
        nearest = ndimage.distance_transform_edt(
            ~mask_yx,
            return_distances=False,
            return_indices=True,
        )
        return surface_yx[tuple(nearest)]

    def preprocess(
        self,
        path: Path,
        resolution: int,
        device: torch.device | str,
        is_target: bool = False,
    ) -> np.ndarray:
        raw = open_raw_hwd(path)
        raw = self._clean_raw(raw)
        volume = torch.tensor(
            np.asarray(raw),
            device=device,
            dtype=torch.float32,
        )
        if not self.air_cleaned:
            volume[:, :, :SURFACE_MARGIN] = OCT_MEAN

        valid = None
        if self.mode != RAW_FULL:
            surface_yx, surface_mask_yx = self._surface(path, raw, is_target)
            depth = torch.arange(raw.shape[2], device=device)[None, None, :]
            if self.mode == FLOOR_REMOVED_FULL:
                if not self.air_cleaned:
                    filled_surface = self._filled_surface(surface_yx, surface_mask_yx)
                    cutoff = torch.as_tensor(
                        np.rint(filled_surface).astype(np.int32),
                        device=device,
                    )[:, :, None] + BAND_AFTER_SURFACE
                    valid = depth <= cutoff
            else:
                surface = torch.as_tensor(
                    np.rint(surface_yx).astype(np.int32),
                    device=device,
                )[:, :, None]
                surface_mask = torch.as_tensor(surface_mask_yx, device=device)[:, :, None]
                valid = (
                    surface_mask
                    & (volume >= TISSUE_THRESHOLD)
                    & (depth >= surface - BAND_BEFORE_SURFACE)
                    & (depth <= surface + BAND_AFTER_SURFACE)
                )

        volume.sub_(OCT_MEAN).div_(OCT_STD).clamp_(-5.0, 5.0)
        if valid is not None:
            volume.masked_fill_(~valid, 0.0)

        volume = volume.permute(2, 0, 1).flip(2)
        if self.mode == TISSUE_ONLY:
            volume = F.adaptive_max_pool3d(
                volume[None, None],
                output_size=(resolution, resolution, resolution),
            )[0, 0]
        else:
            volume = F.interpolate(
                volume[None, None],
                size=(resolution, resolution, resolution),
                mode="trilinear",
                align_corners=True,
            )[0, 0]

        if valid is not None:
            valid_dhw = valid.permute(2, 0, 1).flip(2).to(torch.float32)
            if self.mode == TISSUE_ONLY:
                valid_dhw = F.adaptive_max_pool3d(
                    valid_dhw[None, None],
                    output_size=(resolution, resolution, resolution),
                )[0, 0]
            else:
                valid_dhw = F.interpolate(
                    valid_dhw[None, None],
                    size=(resolution, resolution, resolution),
                    mode="nearest",
                )[0, 0]
            volume.masked_fill_(valid_dhw < 0.5, 0.0)

        return volume.to(torch.float16).cpu().numpy()


def trajectory_from_target(target_volume: Path) -> Path:
    target_volume = Path(target_volume).resolve()
    if target_volume.parent.name == "predeform":
        return target_volume.parent.parent
    return target_volume.parent
