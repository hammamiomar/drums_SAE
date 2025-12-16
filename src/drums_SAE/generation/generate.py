"""Post-hoc generation with SAE steering.

This module provides utilities for TEXT-TO-LATENT generation:

1. `load_models()` - Load Stable Audio Open diffusion model
2. `load_sae()` - Load trained SAE from checkpoint
3. `generate_latents()` - Generate clean latents from text prompts

For STEERING and DECODING, use the shared modules:
- `drums_SAE.steering.probe_steer` - SAE-based steering
- `drums_SAE.vae` - VAE decoding with normalization

Example:
    from drums_SAE.generation import load_models, load_sae, generate_latents
    from drums_SAE.steering import steer_with_probe, ProbeSteeringVectors
    from drums_SAE.vae import decode_latents_to_audio, load_latent_stats

    # Load models
    model, model_config, sample_rate = load_models("mps")
    sae = load_sae("experiments/v2_main/checkpoints/sae_latest.pt", "mps")
    vectors = ProbeSteeringVectors.load("experiments/v2_main/eval/steering_vectors.npz")
    mean, std = load_latent_stats("data/latents_v2.npz", "mps")

    # Generate latents
    latents = generate_latents(model, prompt="kick drum", seed=42, device="mps")

    # Steer (normalize → steer → denormalize)
    # See demo/generation/app.py for the full pattern
"""

import logging
from typing import Any

import torch

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

from drums_SAE.sae.model import AudioSae

logger = logging.getLogger(__name__)


def load_models(device: str = "cuda") -> tuple[Any, dict, int]:
    """Load Stable Audio Open model.

    Note: Does NOT return VAE separately. For VAE decode, use:
        from drums_SAE.vae import load_vae

    Args:
        device: Device to load models to ("cuda", "mps", or "cpu")

    Returns:
        Tuple of (model, model_config, sample_rate)
    """
    logger.info("Loading Stable Audio Open from HuggingFace...")
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)
    model.requires_grad_(False)

    sample_rate = model_config.get("sample_rate", 44100)

    logger.info(f"Stable Audio loaded on {device}, sample_rate={sample_rate}")
    return model, model_config, sample_rate


def load_sae(
    checkpoint_path: str,
    device: str = "cuda",
) -> AudioSae:
    """Load trained SAE from checkpoint.

    Handles both raw state_dict and wrapped checkpoints (with config).

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model to

    Returns:
        Loaded AudioSae model in inference mode
    """
    logger.info(f"Loading SAE from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both raw state_dict and wrapped checkpoints
    if "model_state_dict" in state:
        state_dict = state["model_state_dict"]
        config = state.get("config", {})
    else:
        state_dict = state
        config = {}

    # Infer dimensions from state_dict
    d_hidden = state_dict["encoder.weight"].shape[0]
    d_input = state_dict["encoder.weight"].shape[1]
    expansion_factor = d_hidden // d_input

    # Get topk from config or use default
    topk = config.get("topk", 32)

    logger.info(f"SAE config: d_input={d_input}, expansion={expansion_factor}, topk={topk}")

    sae = AudioSae(
        d_input=d_input,
        expansion_factor=expansion_factor,
        topk=topk,
    )

    # Load state dict (handle potential key mismatches)
    model_keys = set(sae.state_dict().keys())
    filtered_state = {k: v for k, v in state_dict.items() if k in model_keys}
    sae.load_state_dict(filtered_state, strict=False)

    sae = sae.to(device)
    sae.requires_grad_(False)
    sae.training = False

    return sae


def generate_latents(
    model: Any,
    prompt: str = "",
    negative_prompt: str = "",
    seconds: float = 2.0,
    steps: int = 100,
    cfg_scale: float = 7.0,
    seed: int = -1,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate clean latents from a text prompt.

    Uses Stable Audio's diffusion with return_latents=True to get
    the final clean latents before VAE decoding.

    These latents are in the SAME space as pre-encoded latents from
    `latents_v2.npz`, so they can be:
    1. Normalized: (latents - mean) / std
    2. Steered via SAE
    3. Denormalized: latents * std + mean
    4. Decoded via VAE

    Args:
        model: Stable Audio model
        prompt: Text prompt for generation
        negative_prompt: Negative prompt (optional)
        seconds: Duration in seconds
        steps: Number of diffusion steps
        cfg_scale: Classifier-free guidance scale
        seed: Random seed (-1 for random)
        device: Device for generation

    Returns:
        Clean latents tensor, shape (B, 64, T) where T = seconds * sample_rate / 2048
    """
    sample_rate = model.sample_rate
    sample_size = int(seconds * sample_rate)

    # Build conditioning dict (list of dicts, one per batch item)
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": seconds,
    }]

    negative_conditioning = None
    if negative_prompt:
        negative_conditioning = [{
            "prompt": negative_prompt,
            "seconds_start": 0,
            "seconds_total": seconds,
        }]

    logger.info(f"Generating: prompt='{prompt}', steps={steps}, cfg={cfg_scale}, seed={seed}")

    # Generate with return_latents=True to get clean latents (SAE's domain!)
    latents = generate_diffusion_cond(
        model=model,
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        sample_size=sample_size,
        device=device,
        return_latents=True,  # KEY: Returns clean latents, not audio
    )

    logger.info(f"Generated latents shape: {latents.shape}")
    return latents  # Shape: (B, 64, T)
