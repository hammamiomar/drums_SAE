"""Steering module for controllable drum generation via SAE features."""

from drums_SAE.steering.steer import (
    ControlVectors,
    create_steering_grid,
    interpolate_latents,
    steer_latent,
    steer_latent_normalized,
    steer_multi_features_with_residual,
    steer_with_residual,
)
from drums_SAE.steering.temporal_steer import (
    SteeringParams,
    TemporalConfig,
    TemporalControlVector,
    TemporalMode,
    apply_steering,
    build_control_vector,
    build_temporal_control_vectors,
    compute_alpha_envelope,
    steer_envelope,
    steer_segment,
    steer_temporal_features,
    steer_uniform,
)
from drums_SAE.steering.probe_steer import (
    ProbeSteeringVectors,
    steer_with_probe,
    steer_with_probe_bidirectional,
    steer_to_target_bin,
    create_steered_triplet,
)

__all__ = [
    # Probe-based steering (preferred method)
    "ProbeSteeringVectors",
    "steer_with_probe",
    "steer_with_probe_bidirectional",
    "steer_to_target_bin",
    "create_steered_triplet",
    # Core steering with residual (Gytis trick)
    "steer_with_residual",
    "steer_multi_features_with_residual",
    # Control vector steering (legacy, correlation-based)
    "ControlVectors",
    "steer_latent",
    "steer_latent_normalized",
    "interpolate_latents",
    "create_steering_grid",
    # Temporal steering
    "TemporalMode",
    "TemporalConfig",
    "SteeringParams",
    "TemporalControlVector",
    "steer_uniform",
    "steer_segment",
    "steer_envelope",
    "steer_temporal_features",
    "apply_steering",
    "compute_alpha_envelope",
    "build_control_vector",
    "build_temporal_control_vectors",
]
