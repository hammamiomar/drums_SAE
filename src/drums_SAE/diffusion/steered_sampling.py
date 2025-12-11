"""Steered sampling callback for Stable Audio Open diffusion.

This module provides a callback factory that injects SAE-based steering
into the diffusion sampling loop. The callback modifies latent tensors
in-place at each step to guide generation toward desired acoustic properties.

Key Insight:
    The sampling functions in stable_audio_tools pass `x` by reference to
    callbacks. By using `x.copy_(x_steered)`, we modify the actual tensor
    the loop uses—no modifications to stable_audio_tools required.

Example:
    callback = create_steering_callback(
        sae=sae,
        steering_vectors=vectors,
        property_alphas={"bass": 1.5, "spectral_centroid": -0.5},
        apply_at_steps=list(range(30, 70)),  # Middle steps only
    )

    # Pass to sample_rf via generate_diffusion_cond's **sampler_kwargs
    audio = generate_diffusion_cond(..., callback=callback)
"""

import logging
from typing import Callable, Optional

import torch
from einops import rearrange

from drums_SAE.sae.model import AudioSae
from drums_SAE.steering.probe_steer import ProbeSteeringVectors

logger = logging.getLogger(__name__)


def create_steering_callback(
    sae: AudioSae,
    steering_vectors: ProbeSteeringVectors,
    property_alphas: dict[str, float],
    apply_at_steps: Optional[list[int]] = None,
) -> Callable:
    """Create a callback that applies SAE steering at diffusion steps.

    The callback modifies x in-place so the sampling loop uses steered values.
    Uses the Gytis residual trick to preserve reconstruction quality.

    Args:
        sae: Trained SAE model (should be on same device as diffusion model)
        steering_vectors: Loaded probe steering vectors from run_eval
        property_alphas: Dict of {property_name: alpha} for steering.
            Positive alpha = more of property, negative = less.
            Typical range: [-2.0, 2.0]
        apply_at_steps: Which step indices to apply steering (None = all steps).
            For middle-step steering, use list(range(int(steps*0.3), int(steps*0.7)))

    Returns:
        Callable that can be passed as `callback` to sample_rf/sample_k

    Raises:
        ValueError: If property_name not found in steering_vectors

    Note:
        Shape transformation:
        - Diffusion latent x: (B, 64, T) — batch, channels, time
        - SAE expects: (B*T, 64) — flattened vectors
        We use einops.rearrange for clean shape handling.
    """
    device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype

    # Validate properties
    available = steering_vectors.properties
    for prop_name in property_alphas:
        if prop_name not in available:
            raise ValueError(
                f"Unknown property: '{prop_name}'. "
                f"Available: {available}"
            )

    # Pre-compute combined steering direction for efficiency
    # This avoids repeated numpy->torch conversion at each step
    combined_direction = torch.zeros(sae.d_hidden, device=device, dtype=sae_dtype)
    for prop_name, alpha in property_alphas.items():
        direction = steering_vectors.get_direction(prop_name)
        combined_direction += alpha * torch.from_numpy(direction).to(device=device, dtype=sae_dtype)

    # Log steering configuration
    props_str = ", ".join(f"{k}={v:+.2f}" for k, v in property_alphas.items())
    logger.info(f"Created steering callback: {props_str}")
    if apply_at_steps is not None:
        logger.info(f"Applying at steps: {min(apply_at_steps)}-{max(apply_at_steps)}")

    # Track steering statistics for debugging
    stats = {"calls": 0, "steered": 0}

    def steering_callback(callback_info: dict) -> None:
        """Apply steering in-place at each diffusion step.

        Args:
            callback_info: Dict with keys:
                - 'x': Current noisy latent (B, C, T)
                - 'i': Step index (1-indexed from sample_discrete_euler)
                - 't': Current timestep value
                - 'sigma': Noise level (same as t for rectified flow)
                - 'denoised': Model's estimate of clean latent
        """
        stats["calls"] += 1
        step_idx = callback_info["i"]

        # Skip if not in target steps
        if apply_at_steps is not None and step_idx not in apply_at_steps:
            return

        stats["steered"] += 1

        # Get sigma for logging (handle both float and tensor)
        sigma = callback_info.get("sigma", "N/A")
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.item()
        logger.debug(f"Steering step {step_idx}, sigma={sigma:.4f}")

        # Get the tensor to steer
        x = callback_info["x"]  # (B, C, T) = (B, 64, T)
        batch_size, channels, time_steps = x.shape
        original_dtype = x.dtype

        # Reshape for SAE: (B, C, T) -> (B*T, C)
        x_flat = rearrange(x, "b c t -> (b t) c")

        with torch.no_grad():
            # Auto-cast to SAE dtype (typically fp32 for precision)
            x_flat_sae = x_flat.to(dtype=sae_dtype)

            # Encode through SAE
            enc = sae.encode(x_flat_sae, return_aux=False)
            f = enc["f"]  # (B*T, d_hidden) RMS-normalized features

            # Compute residual BEFORE steering (Gytis trick)
            # This preserves information the SAE can't capture (~6-12% of signal)
            x_recon = sae.decode(f)
            residual = x_flat_sae - x_recon

            # Apply steering direction
            f_steered = f + combined_direction
            f_steered = sae.rms_norm(f_steered)

            # Decode with residual preservation
            x_steered_flat = sae.decode(f_steered) + residual

            # Cast back to original dtype
            x_steered_flat = x_steered_flat.to(dtype=original_dtype)

        # Reshape back: (B*T, C) -> (B, C, T)
        x_steered = rearrange(
            x_steered_flat, "(b t) c -> b c t", b=batch_size, t=time_steps
        )

        # CRITICAL: Modify x in-place to affect next diffusion step
        # This works because the sampling loop holds a reference to this tensor
        x.copy_(x_steered)

    # Attach stats to callback for inspection
    steering_callback.stats = stats  # type: ignore

    return steering_callback


def get_step_schedule(
    steps: int,
    schedule: str = "middle",
) -> Optional[list[int]]:
    """Get list of step indices for a named schedule.

    Args:
        steps: Total number of diffusion steps
        schedule: One of:
            - "all": Apply at all steps (returns None)
            - "early": First 30% of steps
            - "middle": Steps 30-70% (recommended by Smule paper)
            - "late": Last 30% of steps

    Returns:
        List of step indices, or None for "all"
    """
    if schedule == "all":
        return None
    elif schedule == "early":
        return list(range(int(steps * 0.3)))
    elif schedule == "middle":
        start = int(steps * 0.3)
        end = int(steps * 0.7)
        return list(range(start, end))
    elif schedule == "late":
        return list(range(int(steps * 0.7), steps))
    else:
        raise ValueError(f"Unknown schedule: {schedule}. Use: all, early, middle, late")
