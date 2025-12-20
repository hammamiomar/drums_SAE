"""VAE loading and audio decoding utilities.

This module provides shared utilities for:
- Loading the Stable Audio Open VAE
- Decoding latents to audio with proper normalization handling
- Loading normalization statistics from NPZ files
"""

from drums_SAE.vae.decode import (
    DEVICE,
    decode_latents_to_audio,
    load_latent_stats,
    load_vae,
)

__all__ = [
    "DEVICE",
    "decode_latents_to_audio",
    "load_latent_stats",
    "load_vae",
]
