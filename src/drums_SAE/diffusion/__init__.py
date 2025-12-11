"""Diffusion-time steering for Stable Audio Open.

This module enables SAE-based property steering during the diffusion
sampling loop, allowing controlled generation of drum sounds with
specific acoustic properties (brightness, bass, punchiness, etc.).

Key Components:
    create_steering_callback: Factory for diffusion step callbacks
    generate_steered_audio: High-level generation API
    load_sae: Utility to load trained SAE checkpoints

Usage:
    from drums_SAE.diffusion import generate_steered_audio

    audio = generate_steered_audio(
        prompt="punchy kick drum",
        property_steering={"bass": 1.5, "spectral_centroid": -0.5},
        steps=100,
        seed=42,
    )
"""

from drums_SAE.diffusion.steered_sampling import create_steering_callback
from drums_SAE.diffusion.generate import generate_steered_audio, load_sae, load_stable_audio

__all__ = [
    "create_steering_callback",
    "generate_steered_audio",
    "load_sae",
    "load_stable_audio",
]
