"""Text-to-latent generation for Stable Audio Open.

This module provides utilities for GENERATING latents from text prompts.
For STEERING and DECODING, use the shared modules:
- `drums_SAE.steering` - SAE-based steering (probe_steer)
- `drums_SAE.vae` - VAE decoding with normalization

Usage:
    from drums_SAE.generation import load_models, load_sae, generate_latents
    from drums_SAE.steering import steer_with_probe, ProbeSteeringVectors
    from drums_SAE.vae import decode_latents_to_audio, load_latent_stats

    # Load models
    model, model_config, sample_rate = load_models("mps")
    sae = load_sae("experiments/v2_main/checkpoints/sae_latest.pt", "mps")
    vectors = ProbeSteeringVectors.load("experiments/v2_main/eval/steering_vectors.npz")

    # Generate clean latents
    latents = generate_latents(model, prompt="kick drum", ...)

    # Steer using shared steering module (handles normalization)
    # See demo/generation/app.py for the full pattern
"""

from drums_SAE.generation.generate import (
    generate_latents,
    load_models,
    load_sae,
)

__all__ = [
    "generate_latents",
    "load_models",
    "load_sae",
]
