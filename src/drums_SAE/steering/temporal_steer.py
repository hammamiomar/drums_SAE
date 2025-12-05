"""
Temporal Steering Module
========================

Core steering logic for the Drums SAE demo.
Implements 4 temporal steering modes with clean interfaces.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
import torch


class TemporalMode(Enum):
    """Steering modes for temporal control."""

    UNIFORM = "uniform"
    SEGMENT = "segment"
    ENVELOPE = "envelope"
    TEMPORAL_FEATURES = "temporal_features"


@dataclass(frozen=True)
class TemporalConfig:
    """Configuration for temporal phases."""

    n_timesteps: int = 32
    attack_end: int = 8  # Timesteps 0-7: Attack (~0-370ms)
    body_end: int = 24  # Timesteps 8-23: Body (~370-1100ms)
    # Timesteps 24-31: Tail (~1100-1500ms)

    @property
    def attack_slice(self) -> slice:
        return slice(0, self.attack_end)

    @property
    def body_slice(self) -> slice:
        return slice(self.attack_end, self.body_end)

    @property
    def tail_slice(self) -> slice:
        return slice(self.body_end, self.n_timesteps)

    @property
    def sustain_slice(self) -> slice:
        """Everything after attack (body + tail)."""
        return slice(self.attack_end, self.n_timesteps)


@dataclass
class SteeringParams:
    """Parameters for a single property's steering."""

    alpha: float = 0.0  # Uniform mode
    alpha_attack: float = 0.0  # Segment/temporal modes
    alpha_body: float = 0.0  # Segment mode
    alpha_tail: float = 0.0  # Segment mode
    alpha_start: float = 0.0  # Envelope mode (attack)
    alpha_end: float = 0.0  # Envelope mode (tail)
    alpha_sustain: float = 0.0  # Temporal features mode

    @property
    def is_active(self) -> bool:
        """Check if any steering is applied."""
        return any(
            abs(v) > 1e-6
            for v in [
                self.alpha,
                self.alpha_attack,
                self.alpha_body,
                self.alpha_tail,
                self.alpha_start,
                self.alpha_end,
                self.alpha_sustain,
            ]
        )


class ControlVector(Protocol):
    """Protocol for control vectors."""

    direction: torch.Tensor


@dataclass
class TemporalControlVector:
    """Control vector with separate attack/sustain directions."""

    attack_direction: torch.Tensor
    sustain_direction: torch.Tensor

    @classmethod
    def from_uniform(cls, direction: torch.Tensor) -> "TemporalControlVector":
        """Create temporal CV from a uniform direction."""
        return cls(attack_direction=direction, sustain_direction=direction)


def steer_uniform(
    z: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Uniform steering: same alpha across all timesteps.

    Args:
        z: Latent tensor, shape (T, D) where T=timesteps, D=latent_dim
        direction: Unit control vector, shape (D,)
        alpha: Steering strength

    Returns:
        Steered latent, shape (T, D)
    """
    return z + alpha * direction.unsqueeze(0)


def steer_segment(
    z: torch.Tensor,
    direction: torch.Tensor,
    params: SteeringParams,
    config: TemporalConfig,
) -> torch.Tensor:
    """
    Segment-based steering: different alpha for attack/body/tail.

    This allows precise control over transient (attack) vs sustained (body/tail)
    characteristics of the drum sound.
    """
    z_steered = z.clone()

    z_steered[config.attack_slice] += params.alpha_attack * direction
    z_steered[config.body_slice] += params.alpha_body * direction
    z_steered[config.tail_slice] += params.alpha_tail * direction

    return z_steered


def steer_envelope(
    z: torch.Tensor,
    direction: torch.Tensor,
    params: SteeringParams,
    config: TemporalConfig,
) -> torch.Tensor:
    """
    Smooth envelope steering: linear interpolation from start to end.

    Creates gradual transitions in the steered property over time.
    """
    device = z.device
    alphas = torch.linspace(
        params.alpha_start, params.alpha_end, config.n_timesteps, device=device
    )
    return z + alphas.unsqueeze(1) * direction.unsqueeze(0)


def steer_temporal_features(
    z: torch.Tensor,
    temporal_cv: TemporalControlVector,
    params: SteeringParams,
    config: TemporalConfig,
) -> torch.Tensor:
    """
    Temporal feature targeting: separate control vectors for attack vs sustain.

    This mode uses features that naturally fire during attack (high attack_ratio)
    for the attack phase, and sustain-heavy features for body/tail.
    This can produce more natural-sounding results than uniform steering.
    """
    z_steered = z.clone()

    # Attack phase uses attack-heavy feature direction
    z_steered[config.attack_slice] += params.alpha_attack * temporal_cv.attack_direction

    # Body and tail use sustain-heavy feature direction
    z_steered[config.sustain_slice] += (
        params.alpha_sustain * temporal_cv.sustain_direction
    )

    return z_steered


def apply_steering(
    z: torch.Tensor,
    direction: torch.Tensor,
    mode: TemporalMode,
    params: SteeringParams,
    config: TemporalConfig,
    temporal_cv: TemporalControlVector | None = None,
) -> torch.Tensor:
    """
    Apply steering based on mode.

    Args:
        z: Original latent, shape (T, D)
        direction: Uniform control direction, shape (D,)
        mode: Steering mode
        params: Steering parameters
        config: Temporal configuration
        temporal_cv: Temporal control vectors (required for TEMPORAL_FEATURES mode)

    Returns:
        Steered latent, shape (T, D)
    """
    if not params.is_active:
        return z

    if mode == TemporalMode.UNIFORM:
        return steer_uniform(z, direction, params.alpha)

    elif mode == TemporalMode.SEGMENT:
        return steer_segment(z, direction, params, config)

    elif mode == TemporalMode.ENVELOPE:
        return steer_envelope(z, direction, params, config)

    elif mode == TemporalMode.TEMPORAL_FEATURES:
        if temporal_cv is None:
            temporal_cv = TemporalControlVector.from_uniform(direction)
        return steer_temporal_features(z, temporal_cv, params, config)

    return z


def compute_alpha_envelope(
    mode: TemporalMode,
    params: SteeringParams,
    config: TemporalConfig,
) -> np.ndarray:
    """
    Compute the alpha envelope for visualization.

    Returns:
        Array of alpha values, shape (n_timesteps,)
    """
    if mode == TemporalMode.UNIFORM:
        return np.full(config.n_timesteps, params.alpha)

    elif mode == TemporalMode.SEGMENT:
        alphas = np.zeros(config.n_timesteps)
        alphas[: config.attack_end] = params.alpha_attack
        alphas[config.attack_end : config.body_end] = params.alpha_body
        alphas[config.body_end :] = params.alpha_tail
        return alphas

    elif mode == TemporalMode.ENVELOPE:
        return np.linspace(params.alpha_start, params.alpha_end, config.n_timesteps)

    elif mode == TemporalMode.TEMPORAL_FEATURES:
        alphas = np.zeros(config.n_timesteps)
        alphas[: config.attack_end] = params.alpha_attack
        alphas[config.attack_end :] = params.alpha_sustain
        return alphas

    return np.zeros(config.n_timesteps)


# =============================================================================
# Control Vector Building
# =============================================================================


def build_control_vector(
    correlations: np.ndarray,
    decoder_weights: torch.Tensor,
    top_k: int = 20,
    min_correlation: float = 0.05,
) -> torch.Tensor | None:
    """
    Build a control vector from feature correlations and decoder weights.

    The control vector is a weighted sum of decoder columns, where weights
    are the correlations between SAE features and the target property.

    Args:
        correlations: Per-feature correlations with target property, shape (n_features,)
        decoder_weights: SAE decoder weight matrix, shape (d_input, n_features)
        top_k: Number of top-correlated features to use
        min_correlation: Minimum |correlation| to consider the label viable

    Returns:
        Unit-normalized control vector, shape (d_input,), or None if not viable
    """
    if np.abs(correlations).max() < min_correlation:
        return None

    # Top-k features by absolute correlation
    top_idx = np.argsort(np.abs(correlations))[-top_k:]
    weights = torch.tensor(correlations[top_idx], dtype=torch.float32)

    # Weighted sum of decoder directions
    vec = decoder_weights[:, top_idx] @ weights

    # Unit normalize
    norm = vec.norm()
    if norm < 1e-8:
        return None

    return vec / norm


def build_temporal_control_vectors(
    correlations: np.ndarray,
    attack_ratios: np.ndarray,
    decoder_weights: torch.Tensor,
    top_k: int = 10,
) -> TemporalControlVector:
    """
    Build separate control vectors for attack vs sustain phases.

    Uses the attack_ratio (mean activation in attack / mean activation in tail)
    to separate features that fire early vs late in the drum sound.

    Args:
        correlations: Per-feature correlations, shape (n_features,)
        attack_ratios: Per-feature attack/sustain ratio, shape (n_features,)
        decoder_weights: SAE decoder, shape (d_input, n_features)
        top_k: Features to use per direction

    Returns:
        TemporalControlVector with separate attack/sustain directions
    """
    # Compute threshold (median of positive attack ratios)
    positive_ratios = attack_ratios[attack_ratios > 0]
    if len(positive_ratios) == 0:
        # Fallback: use uniform direction
        uniform = build_control_vector(correlations, decoder_weights, top_k * 2)
        if uniform is None:
            uniform = torch.zeros(decoder_weights.shape[0])
        return TemporalControlVector.from_uniform(uniform)

    threshold = np.median(positive_ratios)

    # Attack-heavy features: high attack_ratio AND correlated
    attack_mask = (attack_ratios > threshold) & (np.abs(correlations) > 0.05)
    sustain_mask = (
        (attack_ratios <= threshold)
        & (attack_ratios > 0)
        & (np.abs(correlations) > 0.05)
    )

    def build_from_mask(mask: np.ndarray) -> torch.Tensor:
        if mask.sum() < 5:
            return build_control_vector(
                correlations, decoder_weights, top_k
            ) or torch.zeros(decoder_weights.shape[0])

        indices = np.where(mask)[0]
        top_indices = indices[np.argsort(np.abs(correlations[indices]))[-top_k:]]
        weights = torch.tensor(correlations[top_indices], dtype=torch.float32)
        vec = decoder_weights[:, top_indices] @ weights
        norm = vec.norm()
        return vec / norm if norm > 1e-8 else torch.zeros_like(vec)

    return TemporalControlVector(
        attack_direction=build_from_mask(attack_mask),
        sustain_direction=build_from_mask(sustain_mask),
    )
