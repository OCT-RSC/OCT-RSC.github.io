from __future__ import annotations


from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph


SURFACE_MARGIN = 16
NOISE_MULTIPLIER = 2.8
PEAK_FRACTION = 0.23
PERSISTENCE = 3
MAXIMUM_NEIGHBOR_JUMP = 18
AIR_GUARD_VOXELS = 12
MINIMUM_AIR_SAMPLES = 16
MATERIAL_TYPES = ("phantom", "exvivo")
ARTIFACT_BRIGHT_THRESHOLD = 20.0
ARTIFACT_OCCUPANCY_THRESHOLD = 0.35


def open_raw_hwd(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    saved = np.load(Path(path), mmap_mode="r")
    saved_shape = tuple(saved.shape)
    raw = saved[0, :, :, :, 0] if saved.ndim == 5 else saved
    if raw.ndim != 3:
        raise ValueError(f"Expected a 3D or 5D OCT volume, got {saved_shape}")
    return raw, saved_shape


def _largest_smooth_component(
    surface_yx: np.ndarray,
    valid_yx: np.ndarray,
) -> np.ndarray:
    node_index = np.full(valid_yx.shape, -1, dtype=np.int32)
    node_count = int(np.count_nonzero(valid_yx))
    if node_count == 0:
        raise RuntimeError("No direct tissue surface candidates were found")
    node_index[valid_yx] = np.arange(node_count, dtype=np.int32)

    starts: list[np.ndarray] = []
    stops: list[np.ndarray] = []
    for first, second, first_depth, second_depth in (
        (node_index[:-1], node_index[1:], surface_yx[:-1], surface_yx[1:]),
        (node_index[:, :-1], node_index[:, 1:], surface_yx[:, :-1], surface_yx[:, 1:]),
    ):
        connected = (
            (first >= 0)
            & (second >= 0)
            & (np.abs(first_depth - second_depth) <= MAXIMUM_NEIGHBOR_JUMP)
        )
        starts.append(first[connected])
        stops.append(second[connected])

    edge_start = np.concatenate(starts)
    edge_stop = np.concatenate(stops)
    graph = sparse.coo_matrix(
        (
            np.ones(2 * len(edge_start), dtype=np.uint8),
            (
                np.concatenate((edge_start, edge_stop)),
                np.concatenate((edge_stop, edge_start)),
            ),
        ),
        shape=(node_count, node_count),
    ).tocsr()
    _, labels = csgraph.connected_components(graph, directed=False)
    selected_label = int(np.argmax(np.bincount(labels)))
    selected = np.zeros(valid_yx.shape, dtype=bool)
    selected[valid_yx] = labels == selected_label
    return selected


def adaptive_upper_surface(
    raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width, depth = raw.shape
    if depth <= 2 * SURFACE_MARGIN + PERSISTENCE:
        raise ValueError(f"OCT depth {depth} is too small for surface detection")

    direct_surface = np.zeros((height, width), dtype=np.int32)
    direct_detection = np.zeros((height, width), dtype=bool)
    for start in range(0, height, 16):
        stop = min(height, start + 16)
        halo_start = max(0, start - 2)
        halo_stop = min(height, stop + 2)
        detection = np.asarray(raw[halo_start:halo_stop], dtype=np.float32)
        detection = ndimage.gaussian_filter(
            detection,
            sigma=(0.8, 0.8, 1.0),
            mode="nearest",
        )[start - halo_start : stop - halo_start]
        search = detection[:, :, SURFACE_MARGIN : depth - SURFACE_MARGIN]
        baseline = np.median(search, axis=2)
        mad = np.median(np.abs(search - baseline[:, :, None]), axis=2)
        robust_scale = np.maximum(1.4826 * mad, 1.0)
        peak = np.max(search, axis=2)
        threshold = baseline + np.maximum(
            NOISE_MULTIPLIER * robust_scale,
            PEAK_FRACTION * (peak - baseline),
        )
        above = search > threshold[:, :, None]
        candidate_length = search.shape[2] - PERSISTENCE + 1
        candidates = np.logical_and.reduce(
            tuple(
                above[:, :, offset : offset + candidate_length]
                for offset in range(PERSISTENCE)
            )
        )
        direct_detection[start:stop] = np.any(candidates, axis=2)
        direct_surface[start:stop] = (
            np.argmax(candidates, axis=2).astype(np.int32) + SURFACE_MARGIN
        )

    reliable = _largest_smooth_component(direct_surface, direct_detection)
    fill = float(np.median(direct_surface[reliable]))
    filled = np.where(reliable, direct_surface, fill).astype(np.float32)
    local_median = ndimage.median_filter(filled, size=5, mode="nearest")
    reliable &= np.abs(filled - local_median) <= MAXIMUM_NEIGHBOR_JUMP
    reliable = _largest_smooth_component(direct_surface, reliable)

    numerator = ndimage.gaussian_filter(
        np.where(reliable, direct_surface, 0.0).astype(np.float32),
        sigma=1.0,
    )
    denominator = ndimage.gaussian_filter(reliable.astype(np.float32), sigma=1.0)
    surface = np.where(
        denominator > 1e-4,
        numerator / np.maximum(denominator, 1e-4),
        fill,
    ).astype(np.float32)
    surface = np.clip(surface, SURFACE_MARGIN, depth - SURFACE_MARGIN - 1)
    return surface, direct_detection, reliable


def measured_air_statistics(
    raw: np.ndarray,
    surface_yx: np.ndarray,
    reliable_yx: np.ndarray,
) -> dict[str, float]:
    samples: list[np.ndarray] = []
    for row in range(0, raw.shape[0], 8):
        for column in range(0, raw.shape[1], 8):
            if not reliable_yx[row, column]:
                continue
            stop = int(np.floor(surface_yx[row, column])) - AIR_GUARD_VOXELS
            if stop - SURFACE_MARGIN < MINIMUM_AIR_SAMPLES:
                continue
            samples.append(
                np.asarray(raw[row, column, SURFACE_MARGIN:stop:2], dtype=np.float32)
            )
    if not samples:
        return {"median": 10.0, "mean": 10.0, "std": 0.0}
    values = np.concatenate(samples)
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def measured_air_chunk(
    source_hwd: np.ndarray,
    surface_yx: np.ndarray,
    first_row: int,
    fallback: float,
) -> np.ndarray:
    rows, width, depth = source_hwd.shape
    air_stop = np.floor(surface_yx).astype(np.int32) - AIR_GUARD_VOXELS
    air_length = air_stop - SURFACE_MARGIN
    safe_length = np.maximum(air_length, 1)[:, :, None]
    target_z = np.arange(depth, dtype=np.int32)[None, None, :]
    row_index = np.arange(first_row, first_row + rows, dtype=np.int32)[:, None, None]
    column_index = np.arange(width, dtype=np.int32)[None, :, None]
    phase = 37 * row_index + 17 * column_index
    source_z = SURFACE_MARGIN + np.mod(target_z + phase, safe_length)
    source_z = np.clip(source_z, 0, depth - 1)
    sampled = np.take_along_axis(source_hwd, source_z, axis=2).astype(np.float32)
    insufficient = air_length < MINIMUM_AIR_SAMPLES
    if np.any(insufficient):
        sampled[insufficient] = float(fallback)
    return sampled


def _normalized_material(material: str) -> str:
    value = str(material).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "phantom": "phantom",
        "exvivo": "exvivo",
        "porcine": "exvivo",
        "pork": "exvivo",
    }
    if value not in aliases:
        raise ValueError(
            f"Unknown OCT material {material!r}; expected phantom or exvivo."
        )
    return aliases[value]


def _coarse_bright_fraction(
    raw_hwd: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    height, width, depth = raw_hwd.shape
    block_shape = (
        max(1, int(np.ceil(height / 64))),
        max(1, int(np.ceil(width / 64))),
        4,
    )
    coarse_shape = tuple(
        int(np.ceil(size / block))
        for size, block in zip(raw_hwd.shape, block_shape)
    )
    padded_shape = tuple(
        size * block for size, block in zip(coarse_shape, block_shape)
    )
    bright = np.zeros(padded_shape, dtype=np.uint8)
    bright[:height, :width, :depth] = raw_hwd >= threshold
    fraction = bright.reshape(
        coarse_shape[0], block_shape[0],
        coarse_shape[1], block_shape[1],
        coarse_shape[2], block_shape[2],
    ).mean(axis=(1, 3, 5))
    return fraction.astype(np.float32), block_shape


def _component_summary(labels: np.ndarray, component_id: int) -> dict[str, float | int | bool]:
    rows, columns, depths = np.where(labels == component_id)
    height, width, _ = labels.shape
    footprint = np.any(labels == component_id, axis=2)
    center = footprint[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    outer = np.array(footprint, copy=True)
    outer[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = False
    outer_area = footprint.size - center.size
    return {
        "id": int(component_id),
        "voxel_count": int(len(rows)),
        "z_min": int(depths.min()),
        "z_max": int(depths.max()),
        "lateral_faces_touched": int(sum((
            bool(np.any(rows == 0)),
            bool(np.any(rows == height - 1)),
            bool(np.any(columns == 0)),
            bool(np.any(columns == width - 1)),
        ))),
        "row_coverage": float((rows.max() - rows.min() + 1) / height),
        "column_coverage": float((columns.max() - columns.min() + 1) / width),
        "center_occupancy": float(np.mean(center)),
        "outer_occupancy": float(np.count_nonzero(outer) / max(outer_area, 1)),
        "touches_top": bool(np.any(depths == 0)),
    }


def oct_artifact_mask(
    raw_hwd: np.ndarray,
    material: str,
    bright_threshold: float = ARTIFACT_BRIGHT_THRESHOLD,
    occupancy_threshold: float = ARTIFACT_OCCUPANCY_THRESHOLD,
) -> tuple[np.ndarray, dict[str, object]]:
    material_use = _normalized_material(material)
    fraction, block_shape = _coarse_bright_fraction(raw_hwd, bright_threshold)
    occupied = fraction >= occupancy_threshold
    labels, component_count = ndimage.label(
        occupied,
        structure=np.ones((3, 3, 3), dtype=bool),
    )
    counts = np.bincount(labels.ravel())
    minimum_size = max(16, int(0.00005 * labels.size))
    components = [
        _component_summary(labels, component_id)
        for component_id in range(1, component_count + 1)
        if counts[component_id] >= minimum_size
    ]
    components.sort(key=lambda item: int(item["voxel_count"]), reverse=True)

    top_candidates = [
        item for item in components
        if item["touches_top"]
        and item["row_coverage"] >= 0.8
        and item["column_coverage"] >= 0.8
        and item["z_max"] <= max(4, int(0.08 * labels.shape[2]))
    ]
    top = max(top_candidates, key=lambda item: int(item["voxel_count"]), default=None)
    non_top = [item for item in components if top is None or item["id"] != top["id"]]

    if material_use == "phantom":
        tissue_candidates = [
            item for item in non_top
            if item["lateral_faces_touched"] == 0
            and item["z_min"] < int(0.65 * labels.shape[2])
        ]
    else:
        tissue_candidates = [
            item for item in non_top
            if item["z_min"] < int(0.65 * labels.shape[2])
        ]
    tissue = max(
        tissue_candidates,
        key=lambda item: int(item["voxel_count"]),
        default=None,
    )

    floor = None
    if material_use == "phantom" and tissue is not None:
        floor_candidates = [
            item for item in non_top
            if item["id"] != tissue["id"]
            and item["z_min"] > tissue["z_max"]
            and item["lateral_faces_touched"] >= 2
            and item["row_coverage"] >= 0.8
            and item["column_coverage"] >= 0.8
            and item["outer_occupancy"] > item["center_occupancy"]
        ]
        floor = max(
            floor_candidates,
            key=lambda item: int(item["voxel_count"]),
            default=None,
        )

    height, width, depth = raw_hwd.shape
    block_y, block_x, block_z = block_shape
    top_stop = SURFACE_MARGIN
    if top is not None:
        top_stop = max(top_stop, min(depth, (int(top["z_max"]) + 1) * block_z))
    artifact_mask = np.zeros(raw_hwd.shape, dtype=bool)
    artifact_mask[:, :, :top_stop] = True
    if floor is not None:
        coarse_floor = ndimage.binary_dilation(labels == floor["id"], iterations=1)
        floor_mask = np.repeat(
            np.repeat(np.repeat(coarse_floor, block_y, axis=0), block_x, axis=1),
            block_z,
            axis=2,
        )[:height, :width, :depth]
        artifact_mask |= floor_mask

    info = {
        "version": "geometric_measured_air_v1",
        "material": material_use,
        "bright_threshold": float(bright_threshold),
        "occupancy_threshold": float(occupancy_threshold),
        "top_stop_voxel": int(top_stop),
        "top_component_id": None if top is None else int(top["id"]),
        "tissue_component_id": None if tissue is None else int(tissue["id"]),
        "floor_component_id": None if floor is None else int(floor["id"]),
        "artifact_fraction": float(np.mean(artifact_mask)),
    }
    return artifact_mask, info


def replace_artifacts_with_measured_air(
    source_hwd: np.ndarray,
    reference_hwd: np.ndarray,
    reference_surface_yx: np.ndarray,
    reference_reliable_yx: np.ndarray,
    artifact_mask_hwd: np.ndarray,
) -> np.ndarray:
    if source_hwd.shape != reference_hwd.shape or source_hwd.shape != artifact_mask_hwd.shape:
        raise ValueError("OCT source, reference, and artifact mask shapes must match.")
    statistics = measured_air_statistics(
        reference_hwd,
        reference_surface_yx,
        reference_reliable_yx,
    )
    output = np.empty_like(source_hwd)
    for start in range(0, source_hwd.shape[0], 8):
        stop = min(source_hwd.shape[0], start + 8)
        source = np.asarray(source_hwd[start:stop], dtype=np.float32)
        air = measured_air_chunk(
            reference_hwd[start:stop],
            reference_surface_yx[start:stop],
            start,
            statistics["median"],
        )
        cleaned = np.where(artifact_mask_hwd[start:stop], air, source)
        if np.issubdtype(output.dtype, np.integer):
            limits = np.iinfo(output.dtype)
            cleaned = np.rint(cleaned).clip(limits.min, limits.max)
        output[start:stop] = cleaned.astype(output.dtype, copy=False)
    return output
