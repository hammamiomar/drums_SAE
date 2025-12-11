"""Property tracking callback for evolution visualization.

This callback tracks property scores at each diffusion step AND optionally
applies steering. By combining both functions, we avoid the complexity of
managing separate callbacks or running generation twice.
"""

import torch
from einops import rearrange
from typing import Callable, Optional

from drums_SAE.sae.model import AudioSae
from drums_SAE.steering.probe_steer import ProbeSteeringVectors

from .presets import PROPERTY_NAMES


def create_tracking_callback(
    sae: AudioSae,
    steering_vectors: ProbeSteeringVectors,
    property_alphas: Optional[dict[str, float]] = None,
    apply_at_steps: Optional[list[int]] = None,
) -> tuple[Callable, list[dict]]:
    """Create a callback that tracks properties AND optionally steers.

    This is the "tracking mode" callback — it records property scores at every
    step for visualization, and also applies steering if requested.

    Args:
        sae: Trained SAE model (on same device as diffusion model)
        steering_vectors: Loaded probe steering vectors
        property_alphas: If provided, apply steering with these alphas.
            Dict of {property_name: alpha}. None = tracking only.
        apply_at_steps: Which step indices to apply steering (None = all).
            Only used if property_alphas is provided.

    Returns:
        Tuple of (callback_function, step_data_list).
        The step_data_list is populated during generation with entries like:
        {"step": 1, "bass": 0.42, "spectral_centroid": 0.31, ...}
    """
    device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype

    step_data: list[dict] = []

    # Pre-load probe directions for property scoring
    # Shape: (d_hidden,) for each property
    probe_directions = {}
    for prop in PROPERTY_NAMES:
        direction = steering_vectors.get_direction(prop)
        probe_directions[prop] = torch.from_numpy(direction).to(
            device=device, dtype=sae_dtype
        )

    # Pre-compute combined steering direction if steering
    combined_direction = None
    if property_alphas:
        combined_direction = torch.zeros(sae.d_hidden, device=device, dtype=sae_dtype)
        for prop_name, alpha in property_alphas.items():
            if abs(alpha) > 0.01 and prop_name in probe_directions:
                combined_direction += alpha * probe_directions[prop_name]

    def tracking_callback(callback_info: dict) -> None:
        """Track properties and optionally steer at each diffusion step.

        Args:
            callback_info: Dict with keys:
                - 'x': Current noisy latent (B, C, T)
                - 'i': Step index (1-indexed)
                - 't': Current timestep value
                - 'sigma': Noise level
                - 'denoised': Model's estimate of clean latent
        """
        step_idx = callback_info["i"]
        x = callback_info["x"]  # (B, C, T) = (B, 64, T)
        batch_size, channels, time_steps = x.shape
        original_dtype = x.dtype

        # Reshape for SAE: (B, C, T) -> (B*T, C)
        x_flat = rearrange(x, "b c t -> (b t) c")

        with torch.no_grad():
            # Auto-cast to SAE dtype
            x_flat_sae = x_flat.to(dtype=sae_dtype)

            # Encode through SAE
            enc = sae.encode(x_flat_sae, return_aux=False)
            f = enc["f"]  # (B*T, d_hidden) RMS-normalized

            # === TRACKING: Score each property ===
            # Score = mean projection of features onto probe direction
            props = {"step": step_idx}
            for prop, direction in probe_directions.items():
                # Dot product of features with probe direction, averaged
                score = (f @ direction).mean().item()
                props[prop] = score

            step_data.append(props)

            # === STEERING: Apply if requested ===
            if combined_direction is not None:
                should_steer = (
                    apply_at_steps is None or step_idx in apply_at_steps
                )

                if should_steer:
                    # Compute residual BEFORE steering (Gytis trick)
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
                        x_steered_flat,
                        "(b t) c -> b c t",
                        b=batch_size,
                        t=time_steps,
                    )

                    # CRITICAL: Modify x in-place to affect next diffusion step
                    x.copy_(x_steered)

    return tracking_callback, step_data


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
            - "middle": Steps 30-70% (recommended)
            - "late": Last 30% of steps

    Returns:
        List of step indices, or None for "all"
    """
    if schedule == "all":
        return None
    elif schedule == "early":
        return list(range(1, int(steps * 0.3) + 1))
    elif schedule == "middle":
        start = int(steps * 0.3)
        end = int(steps * 0.7)
        return list(range(start + 1, end + 1))
    elif schedule == "late":
        return list(range(int(steps * 0.7) + 1, steps + 1))
    else:
        return None
