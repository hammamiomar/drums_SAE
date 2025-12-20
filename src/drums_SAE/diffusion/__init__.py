"""
⚠️  DEPRECATED: Diffusion-time steering for Stable Audio Open.

This module attempted SAE-based steering DURING diffusion sampling.
**This approach FAILED** because:
- SAE was trained on CLEAN VAE latents
- Diffusion intermediates are NOISY (completely different distribution)
- Steering in the wrong distribution produces unpredictable results

USE INSTEAD:
    from drums_SAE.generation import generate_latents
    from drums_SAE.steering import steer_with_probe
    from drums_SAE.vae import decode_latents_to_audio

    # Post-hoc steering (correct approach):
    latents = generate_latents(model, prompt="kick drum")
    z_steered = steer_with_probe(z_norm, sae, direction, alpha)
    audio = decode_latents_to_audio(z_steered, vae, mean, std)

See demo/generation/app.py for the complete working example.
"""

import warnings

warnings.warn(
    "drums_SAE.diffusion is deprecated. Use drums_SAE.generation for post-hoc steering.",
    DeprecationWarning,
    stacklevel=2,
)

from drums_SAE.diffusion.steered_sampling import create_steering_callback
from drums_SAE.diffusion.generate import generate_steered_audio, load_sae, load_stable_audio

__all__ = [
    "create_steering_callback",
    "generate_steered_audio",
    "load_sae",
    "load_stable_audio",
]
