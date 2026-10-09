import itertools

import torch
import torch.nn as nn

from .encoders import (
    ConditionEncoder as DenseConditionEncoder,
    CosineSchedule,
    TimeEmbedding,
    TransformerBlock,
)


class PointNetEncoder(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.output(self.point_mlp(points).amax(dim=1))


def _batched_sparse_frames(
    frames: list[dict], resolution: tuple[int, int, int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = []
    features = []
    bounds = torch.tensor(resolution, device=device, dtype=torch.long)
    for batch_index, frame in enumerate(frames):
        spatial = frame["coordinates"].to(device=device, dtype=torch.long)
        feature = frame["features"].to(device=device, dtype=torch.float32)
        if spatial.ndim != 2 or spatial.shape[1] != 3:
            raise ValueError(f"Expected sparse DHW coordinates [N,3], got {spatial.shape}")
        if feature.ndim != 2 or feature.shape[0] != spatial.shape[0]:
            raise ValueError("Sparse coordinates and features have different lengths")
        if not bool(((spatial >= 0) & (spatial < bounds)).all()):
            raise ValueError("Sparse coordinate lies outside the configured DHW lattice")
        batch = torch.full((len(spatial), 1), batch_index, device=device, dtype=torch.long)
        coordinates.append(torch.cat((batch, spatial), dim=1))
        features.append(feature)
    return torch.cat(coordinates, dim=0), torch.cat(features, dim=0)


def _linear_coordinate_keys(
    coordinates: torch.Tensor, resolution: tuple[int, int, int]
) -> torch.Tensor:
    depth, height, width = resolution
    batch, d_coord, h_coord, w_coord = coordinates.unbind(dim=1)
    return ((batch * depth + d_coord) * height + h_coord) * width + w_coord


def _voxel_mean_pool(
    coordinates: torch.Tensor, features: torch.Tensor, stride: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if stride == 1:
        return coordinates, features
    output_coordinates = coordinates.clone()
    output_coordinates[:, 1:] = torch.div(
        output_coordinates[:, 1:], stride, rounding_mode="floor"
    )
    output_coordinates, inverse = torch.unique(
        output_coordinates, dim=0, sorted=True, return_inverse=True
    )
    order = torch.argsort(inverse, stable=True)
    counts = torch.bincount(inverse, minlength=len(output_coordinates))
    return output_coordinates, torch.segment_reduce(features[order], "mean", lengths=counts)


class NativeSparseConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int, normalization=True):
        super().__init__()
        self.stride = int(stride)
        self.input_projection = nn.Linear(in_channels, out_channels, bias=False)
        self.kernel = nn.Parameter(torch.full((27, out_channels), 1.0 / 27.0))
        self.channel_mixing = nn.Linear(out_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.normalization = nn.BatchNorm1d(out_channels) if normalization else nn.Identity()
        self.activation = nn.ReLU(inplace=False)
        self.register_buffer(
            "offsets",
            torch.tensor(list(itertools.product((-1, 0, 1), repeat=3)), dtype=torch.long),
            persistent=False,
        )

    def _neighborhood(self, coordinates, features, resolution):
        keys = _linear_coordinate_keys(coordinates, resolution)
        order = torch.argsort(keys)
        sorted_keys = keys[order]
        sorted_features = features[order]
        query_spatial = coordinates[:, None, 1:] + self.offsets[None]
        bounds = torch.tensor(resolution, device=coordinates.device, dtype=torch.long)
        valid = ((query_spatial >= 0) & (query_spatial < bounds)).all(dim=-1)
        query_batch = coordinates[:, None, :1].expand(-1, 27, -1)
        query_coordinates = torch.cat((query_batch, query_spatial), dim=-1)
        query_keys = _linear_coordinate_keys(
            query_coordinates.reshape(-1, 4), resolution
        ).reshape(len(coordinates), 27)
        positions = torch.searchsorted(sorted_keys, query_keys)
        safe_positions = positions.clamp(max=max(len(sorted_keys) - 1, 0))
        matches = valid & (positions < len(sorted_keys))
        matches &= sorted_keys[safe_positions] == query_keys
        neighbors = sorted_features[safe_positions] * matches[..., None].to(features.dtype)
        return (neighbors * self.kernel[None].to(features.dtype)).sum(dim=1)

    def forward(self, coordinates, features, resolution):
        coordinates, features = _voxel_mean_pool(coordinates, features, self.stride)
        if self.stride > 1:
            resolution = tuple((size + self.stride - 1) // self.stride for size in resolution)
        projected = self.input_projection(features)
        aggregated = self._neighborhood(coordinates, projected, resolution)
        output = projected + self.channel_mixing(aggregated) + self.bias
        return coordinates, self.activation(self.normalization(output)), resolution


class NativeSparseVolumeEncoder(nn.Module):

    def __init__(self, in_channels: int, embedding_dim: int, resolution: tuple[int, int, int]):
        super().__init__()
        self.resolution = tuple(int(value) for value in resolution)
        channels = [in_channels + 3, 32, 64, 128, embedding_dim]
        self.blocks = nn.ModuleList(
            [
                NativeSparseConvBlock(channels[0], channels[1], stride=1),
                NativeSparseConvBlock(channels[1], channels[2], stride=2),
                NativeSparseConvBlock(channels[2], channels[3], stride=2),
                NativeSparseConvBlock(channels[3], channels[4], stride=2, normalization=False),
            ]
        )

    def forward(self, frames: list[dict], device: torch.device) -> torch.Tensor:
        coordinates, features = _batched_sparse_frames(frames, self.resolution, device)
        scale = torch.tensor(self.resolution, device=device, dtype=features.dtype).clamp_min(2.0)
        position = coordinates[:, 1:].to(features.dtype) / (scale - 1.0)
        features = torch.cat((features, position.mul(2.0).sub(1.0)), dim=1)
        resolution = self.resolution
        for block in self.blocks:
            coordinates, features, resolution = block(coordinates, features, resolution)
        batch_indices = coordinates[:, 0]
        return torch.stack([features[batch_indices == index].mean(dim=0) for index in range(len(frames))])


def sparse_residual_frame(current: dict, target: dict) -> dict:
    current_coordinates = current["coordinates"].int()
    target_coordinates = target["coordinates"].int()
    coordinates = torch.cat((target_coordinates, current_coordinates), dim=0)
    unique, inverse = torch.unique(coordinates, dim=0, return_inverse=True)
    features = torch.zeros((len(unique), 2), dtype=torch.float32)
    target_count = len(target_coordinates)
    target_index = inverse[:target_count]
    current_index = inverse[target_count:]
    features[:, 0].scatter_add_(0, target_index, target["features"][:, 0].float())
    features[:, 0].scatter_add_(0, current_index, -current["features"][:, 0].float())
    features[:, 1].scatter_add_(0, target_index, torch.ones(target_count))
    features[:, 1].scatter_add_(0, current_index, -torch.ones(len(current_coordinates)))
    return {"coordinates": unique, "features": features}


class CompactConditionEncoder(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        self.representation = config["representation"]
        self.history = int(config["observation_window"])
        self.embedding_dim = int(config.get("embedding_dim", 128))
        self.use_residual = bool(config["use_residual"])
        if self.representation == "point_cloud":
            self.encoder = PointNetEncoder(self.embedding_dim)
            self.residual_encoder = None
        elif self.representation == "tissue_only":
            if config.get("sparse_backend") != "native_pytorch":
                raise ValueError("tissue_only requires sparse_backend=native_pytorch")
            resolution = tuple(config["sparse_resolution"])
            self.encoder = NativeSparseVolumeEncoder(1, self.embedding_dim, resolution)
            self.residual_encoder = (
                NativeSparseVolumeEncoder(2, self.embedding_dim, resolution)
                if self.use_residual
                else None
            )
        else:
            raise ValueError(f"CompactConditionEncoder cannot encode {self.representation}")
        self.temporal = nn.Embedding(self.history, self.embedding_dim)
        self.kind = nn.Embedding(4 if self.use_residual else 3, self.embedding_dim)
        self.tool = nn.Sequential(
            nn.Linear(6, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def _encode_sparse_history(self, observations, device):
        batch = len(observations)
        encoded = []
        for time_index in range(self.history):
            frames = [observations[item][time_index] for item in range(batch)]
            encoded.append(self.encoder(frames, device))
        return torch.stack(encoded, dim=1)

    def forward(self, observations, target, tool_poses):
        device = tool_poses.device
        batch = tool_poses.shape[0]
        if self.representation == "point_cloud":
            observation_features = self.encoder(observations.flatten(0, 1)).reshape(
                batch, self.history, self.embedding_dim
            )
            target_features = self.encoder(target)
        else:
            observation_features = self._encode_sparse_history(observations, device)
            target_features = self.encoder(target, device)
        temporal = self.temporal(torch.arange(self.history, device=device))
        observation_tokens = observation_features + temporal[None] + self.kind.weight[0]
        target_tokens = target_features[:, None] + self.kind.weight[1]
        tool_tokens = self.tool(tool_poses) + temporal[None] + self.kind.weight[2]
        tokens = [observation_tokens, target_tokens, tool_tokens]
        if self.use_residual:
            if self.representation == "point_cloud":
                residual_features = target_features[:, None] - observation_features
            else:
                residual_by_time = []
                for time_index in range(self.history):
                    frames = [
                        sparse_residual_frame(observations[item][time_index], target[item])
                        for item in range(batch)
                    ]
                    residual_by_time.append(self.residual_encoder(frames, device))
                residual_features = torch.stack(residual_by_time, dim=1)
            tokens.append(residual_features + temporal[None] + self.kind.weight[3])
        return torch.cat(tokens, dim=1)


class DiffusionActionHead(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        embedding = int(config.get("embedding_dim", 128))
        self.horizon = int(config["prediction_window"])
        self.action_dim = 6
        self.action = nn.Linear(6, embedding)
        self.position = nn.Parameter(torch.randn(1, self.horizon, embedding) * 0.02)
        self.time = TimeEmbedding(embedding)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding,
                    int(config.get("attention_heads", 4)),
                    int(config.get("feedforward_dim", 512)),
                )
                for _ in range(int(config.get("transformer_layers", 4)))
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(embedding), nn.Linear(embedding, 6))

    def forward(self, noisy_action, timestep, condition):
        value = self.action(noisy_action) + self.position[:, : noisy_action.shape[1]]
        time = self.time(timestep)
        for block in self.blocks:
            value = block(value, condition, time)
        return self.output(value)


class DeterministicBCHead(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        embedding = int(config.get("embedding_dim", 128))
        self.horizon = int(config["prediction_window"])
        self.queries = nn.Parameter(torch.randn(1, self.horizon, embedding) * 0.02)
        self.attention = nn.MultiheadAttention(
            embedding, int(config.get("attention_heads", 4)), batch_first=True
        )
        self.net = nn.Sequential(
            nn.LayerNorm(embedding),
            nn.Linear(embedding, int(config.get("feedforward_dim", 512))),
            nn.SiLU(),
            nn.Linear(int(config.get("feedforward_dim", 512)), 6),
        )

    def forward(self, condition):
        queries = self.queries.expand(condition.shape[0], -1, -1)
        attended = queries + self.attention(queries, condition, condition, need_weights=False)[0]
        return self.net(attended)


class UnifiedPolicy(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.head_name = config["head"]
        self.horizon = int(config["prediction_window"])
        if config["representation"] == "full_volume":
            self.condition = DenseConditionEncoder(
                history=int(config["observation_window"]),
                tool_dim=6,
                embedding_dim=int(config.get("embedding_dim", 128)),
                include_mask_channel=False,
                use_residual_history=bool(config["use_residual"]),
                spatial_grid=tuple(config.get("spatial_grid", (4, 4, 4))),
            )
        else:
            self.condition = CompactConditionEncoder(config)
        self.head = (
            DiffusionActionHead(config)
            if self.head_name == "diffusion"
            else DeterministicBCHead(config)
        )

    def encode_condition(self, batch: dict) -> torch.Tensor:
        return self.condition(
            batch["observations"], batch["target"], batch["tool_poses"]
        )

    def forward(
        self,
        batch: dict,
        noisy_action: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self.encode_condition(batch)
        if self.head_name == "deterministic_bc":
            if noisy_action is not None or timestep is not None:
                raise ValueError("Deterministic BC does not accept diffusion inputs")
            return self.head(condition)
        if noisy_action is None or timestep is None:
            raise ValueError("Diffusion requires noisy_action and timestep")
        return self.head(noisy_action, timestep, condition)


def build_policy(config: dict) -> UnifiedPolicy:
    return UnifiedPolicy(config)


@torch.inference_mode()
def predict_actions(
    policy: UnifiedPolicy,
    schedule: CosineSchedule | None,
    batch: dict,
    inference_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    condition = policy.encode_condition(batch)
    if policy.head_name == "deterministic_bc":
        return policy.head(condition).clamp(-1.0, 1.0)
    if schedule is None:
        raise ValueError("Diffusion inference requires a noise schedule")
    batch_size = batch["tool_poses"].shape[0]
    action = torch.randn(
        batch_size,
        policy.horizon,
        6,
        device=batch["tool_poses"].device,
        generator=generator,
    )
    timesteps = torch.linspace(
        schedule.steps - 1,
        0,
        int(inference_steps),
        device=action.device,
    ).long()
    for index, step in enumerate(timesteps):
        timestep = torch.full((batch_size,), int(step), device=action.device, dtype=torch.long)
        clean = policy.head(action, timestep, condition).clamp(-1.0, 1.0)
        if index == len(timesteps) - 1:
            action = clean
            break
        alpha = schedule.alpha_bar[timestep].to(action.dtype)[:, None, None]
        noise = (action - alpha.sqrt() * clean) / (1.0 - alpha).sqrt()
        next_alpha = schedule.alpha_bar[timesteps[index + 1]].to(action.dtype)
        action = next_alpha.sqrt() * clean + (1.0 - next_alpha).sqrt() * noise
    return action
