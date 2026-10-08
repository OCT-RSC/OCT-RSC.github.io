from __future__ import annotations


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DIFFUSION_STEPS = 100
INFERENCE_STEPS = 20


def _groups(channels: int, maximum: int = 8) -> int:
    for value in range(min(channels, maximum), 0, -1):
        if channels % value == 0:
            return value
    return 1


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return F.silu(volume + self.net(volume))


class VolumeEncoder(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        include_mask_channel: bool = False,
        spatial_grid: tuple[int, int, int] = (2, 4, 4),
    ):
        super().__init__()
        self.include_mask_channel = bool(include_mask_channel)

        def down(in_channels: int, out_channels: int, stride: tuple[int, int, int]):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1),
                nn.GroupNorm(_groups(out_channels), out_channels),
                nn.SiLU(),
                ResBlock3D(out_channels),
            )

        self.net = nn.Sequential(
            down(2 if self.include_mask_channel else 1, 16, (1, 2, 2)),
            down(16, 32, (2, 2, 2)),
            down(32, 64, (2, 2, 2)),
            down(64, embedding_dim, (2, 2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool3d(spatial_grid)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim == 4:
            volume = volume[:, None]
        if self.include_mask_channel:
            occupancy = (volume.abs() > 1e-6).to(volume.dtype)
            volume = torch.cat((volume, occupancy), dim=1)
        encoded = self.pool(self.net(volume))
        return encoded.flatten(2).transpose(1, 2)


class SpatialPosition3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        spatial_grid: tuple[int, int, int],
    ):
        super().__init__()
        depth, height, width = spatial_grid
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, depth),
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing="ij",
        )
        coordinates = torch.stack((z, y, x), dim=-1).reshape(-1, 3)
        frequencies = torch.tensor((1.0, 2.0, 4.0, 8.0)) * math.pi
        angles = coordinates[:, :, None] * frequencies[None, None]
        fourier = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(1)
        self.register_buffer("fourier", fourier)
        self.projection = nn.Linear(fourier.shape[1], embedding_dim, bias=False)

    def forward(self) -> torch.Tensor:
        return self.projection(self.fourier)


class ConditionEncoder(nn.Module):

    def __init__(
        self,
        history: int,
        tool_dim: int,
        embedding_dim: int,
        include_mask_channel: bool = False,
        use_residual_history: bool = False,
        spatial_grid: tuple[int, int, int] = (2, 4, 4),
    ):
        super().__init__()
        self.history = history
        self.embedding_dim = embedding_dim
        self.spatial_tokens = math.prod(spatial_grid)
        self.use_residual_history = bool(use_residual_history)
        self.observation_encoder = VolumeEncoder(
            embedding_dim, include_mask_channel, spatial_grid
        )
        self.target_encoder = VolumeEncoder(
            embedding_dim, include_mask_channel, spatial_grid
        )
        self.residual_encoder = (
            VolumeEncoder(embedding_dim, include_mask_channel, spatial_grid)
            if self.use_residual_history
            else None
        )
        self.spatial = SpatialPosition3D(embedding_dim, spatial_grid)
        self.temporal = nn.Embedding(history, embedding_dim)
        self.kind = nn.Embedding(4 if self.use_residual_history else 3, embedding_dim)
        self.tool = nn.Sequential(
            nn.Linear(tool_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        observation_history: torch.Tensor,
        target_volume: torch.Tensor,
        tool_pose_history: torch.Tensor,
    ) -> torch.Tensor:
        batch, history = observation_history.shape[:2]
        if history != self.history:
            raise ValueError(f"Expected {self.history} observations, received {history}")

        observation_tokens = self.observation_encoder(
            observation_history.flatten(0, 1)
        ).reshape(batch, history, self.spatial_tokens, self.embedding_dim)
        target_tokens = self.target_encoder(target_volume)

        spatial = self.spatial()
        observation_tokens = observation_tokens + spatial[None, None]
        target_tokens = target_tokens + spatial[None]

        time = self.temporal(torch.arange(history, device=observation_history.device))
        observation_tokens = (
            observation_tokens + time[None, :, None] + self.kind.weight[0]
        )
        target_tokens = target_tokens + self.kind.weight[1]
        tool_tokens = self.tool(tool_pose_history) + time[None] + self.kind.weight[2]
        token_groups = [observation_tokens.flatten(1, 2), target_tokens, tool_tokens]
        if self.use_residual_history:
            residual_history = target_volume[:, None] - observation_history
            residual_tokens = self.residual_encoder(
                residual_history.flatten(0, 1)
            ).reshape(batch, history, self.spatial_tokens, self.embedding_dim)
            residual_tokens = (
                residual_tokens
                + spatial[None, None]
                + time[None, :, None]
                + self.kind.weight[3]
            )
            token_groups.append(residual_tokens.flatten(1, 2))
        return torch.cat(token_groups, dim=1)


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.SiLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.embedding_dim // 2
        frequency = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timestep.device)
            / max(half - 1, 1)
        )
        angle = timestep.float()[:, None] * frequency[None]
        embedding = torch.cat((angle.sin(), angle.cos()), dim=-1)
        return self.mlp(F.pad(embedding, (0, self.embedding_dim - embedding.shape[-1])))


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, heads: int, feedforward_dim: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embedding_dim, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embedding_dim, heads, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.SiLU(),
            nn.Linear(feedforward_dim, embedding_dim),
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.time = nn.Linear(embedding_dim, 3 * embedding_dim)

    def forward(self, actions, condition, time_embedding):
        shift1, shift2, shift3 = self.time(time_embedding).chunk(3, dim=-1)
        actions = actions + shift1[:, None]
        normalized = self.norm1(actions)
        actions = actions + self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        actions = actions + shift2[:, None]
        actions = actions + self.cross_attention(
            self.norm2(actions), condition, condition, need_weights=False
        )[0]
        actions = actions + shift3[:, None]
        return actions + self.feedforward(self.norm3(actions))


class CosineSchedule:
    def __init__(self, steps: int, device: torch.device | str):
        self.steps = int(steps)
        time = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((time / steps) + 0.008) / 1.008 * math.pi / 2).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999).float()
        self.alpha_bar = torch.cumprod(1 - beta, dim=0).to(device)

    def add_noise(self, clean_action, timestep, noise):
        alpha = self.alpha_bar[timestep].to(clean_action.dtype)[:, None, None]
        return alpha.sqrt() * clean_action + (1 - alpha).sqrt() * noise
