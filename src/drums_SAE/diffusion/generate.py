"""High-level API for steered audio generation.

This module provides the main entry point for generating audio with
SAE-based property steering during the diffusion process.

Example:
    from drums_SAE.diffusion import generate_steered_audio

    # Simple usage - models loaded automatically
    audio = generate_steered_audio(
        prompt="punchy kick drum",
        property_steering={"bass": 1.5, "spectral_centroid": -0.5},
    )

    # Efficient batch usage - pre-load models
    model, model_config = load_stable_audio()
    sae = load_sae("experiments/v2_main/checkpoints/sae_latest.pt")
    vectors = ProbeSteeringVectors.load("experiments/v2_main/eval/steering_vectors.npz")

    for seed in range(10):
        audio = generate_steered_audio(
            prompt="hi-hat",
            property_steering={"spectral_centroid": 1.0},
            seed=seed,
            model=model,
            model_config=model_config,
            sae=sae,
            steering_vectors=vectors,
        )
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Any

import torch

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

from drums_SAE.sae.model import AudioSae
from drums_SAE.steering.probe_steer import ProbeSteeringVectors
from drums_SAE.diffusion.steered_sampling import create_steering_callback, get_step_schedule

logger = logging.getLogger(__name__)


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
    # Filter out non-model keys like 'steps_since_fired' buffer if needed
    model_keys = set(sae.state_dict().keys())
    filtered_state = {k: v for k, v in state_dict.items() if k in model_keys}
    sae.load_state_dict(filtered_state, strict=False)

    sae = sae.to(device)
    sae.requires_grad_(False)
    sae.training = False  # Set to inference mode

    return sae


def load_stable_audio(
    device: str = "cuda",
) -> Tuple[Any, dict]:
    """Load Stable Audio Open model from HuggingFace.

    Args:
        device: Device to load model to

    Returns:
        Tuple of (model, model_config)
    """
    logger.info("Loading Stable Audio Open from HuggingFace...")
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)
    model.requires_grad_(False)
    logger.info("Stable Audio Open loaded successfully")
    return model, model_config


def generate_steered_audio(
    prompt: str = "",
    property_steering: Optional[dict[str, float]] = None,
    steps: int = 100,
    cfg_scale: float = 7.0,
    seed: int = -1,
    sample_size: int = 65536,
    device: str = "cuda",
    apply_steering_at: str = "middle",
    # Pre-loaded models (optional, for efficiency)
    model: Optional[Any] = None,
    model_config: Optional[dict] = None,
    sae: Optional[AudioSae] = None,
    steering_vectors: Optional[ProbeSteeringVectors] = None,
    # Model paths
    sae_checkpoint: str = "experiments/v2_main/checkpoints/sae_latest.pt",
    vectors_path: str = "experiments/v2_main/eval/steering_vectors.npz",
) -> torch.Tensor:
    """Generate audio with SAE-based property steering.

    This function generates audio from noise using Stable Audio Open's
    diffusion model, with optional SAE-based steering to control acoustic
    properties during generation.

    Args:
        prompt: Text prompt for generation (e.g., "kick drum", "snare hit")
        property_steering: Dict of {property: alpha} for steering.
            Available properties (V2): spectral_centroid, rms, crest_factor, bass
            Typical alpha range: [-2.0, 2.0]
            Examples:
                {"bass": 1.5} — More bass/body
                {"spectral_centroid": -1.0} — Darker sound
                {"crest_factor": 0.5, "rms": 0.5} — Punchier and louder
        steps: Number of diffusion sampling steps (50-150 typical)
        cfg_scale: Classifier-free guidance scale (higher = more prompt adherence)
        seed: Random seed for reproducibility (-1 for random)
        sample_size: Output length in audio samples.
            65536 = ~1.5s at 44.1kHz, 32768 = ~0.74s
        device: Device for computation ("cuda", "mps", or "cpu")
        apply_steering_at: When to apply steering during diffusion:
            - "all": Every step
            - "early": First 30% of steps
            - "middle": Steps 30-70% (default, recommended)
            - "late": Last 30% of steps
        model: Pre-loaded Stable Audio model (optional, for efficiency)
        model_config: Model config dict (required if model provided)
        sae: Pre-loaded SAE (optional, for efficiency)
        steering_vectors: Pre-loaded steering vectors (optional)
        sae_checkpoint: Path to SAE checkpoint (if sae not provided)
        vectors_path: Path to steering vectors (if steering_vectors not provided)

    Returns:
        Generated audio tensor, shape (channels, samples) or (samples,)

    Example:
        # Basic generation
        audio = generate_steered_audio(
            prompt="punchy kick drum",
            property_steering={"bass": 1.5},
            seed=42,
        )

        # Save to file
        import torchaudio
        torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), 44100)
    """
    property_steering = property_steering or {}

    # Load Stable Audio model if not provided
    if model is None:
        model, model_config = load_stable_audio(device)
    elif model_config is None:
        raise ValueError("model_config required when model is provided")

    # Load SAE if steering requested and not provided
    if property_steering and sae is None:
        sae = load_sae(sae_checkpoint, device)

    # Load steering vectors if steering requested and not provided
    if property_steering and steering_vectors is None:
        logger.info(f"Loading steering vectors from {vectors_path}")
        steering_vectors = ProbeSteeringVectors.load(vectors_path)

    # Get step schedule
    apply_at_steps = get_step_schedule(steps, apply_steering_at)

    # Create steering callback
    callback = None
    if property_steering and sae is not None and steering_vectors is not None:
        callback = create_steering_callback(
            sae=sae,
            steering_vectors=steering_vectors,
            property_alphas=property_steering,
            apply_at_steps=apply_at_steps,
        )
        props_str = ", ".join(f"{k}={v:+.2f}" for k, v in property_steering.items())
        logger.info(f"Generating with steering: {props_str}")
    else:
        logger.info("Generating baseline (no steering)")

    # Build conditioning (must be a list of dicts, one per batch item)
    sample_rate = model_config.get("sample_rate", 44100)
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": sample_size / sample_rate,
    }]

    logger.info(f"Prompt: '{prompt}', steps={steps}, cfg={cfg_scale}, seed={seed}")

    # Generate with callback passed through sampler_kwargs
    audio = generate_diffusion_cond(
        model=model,
        conditioning=conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        sample_size=sample_size,
        device=device,
        callback=callback,
    )

    # Log callback stats if available
    if callback is not None and hasattr(callback, "stats"):
        stats = callback.stats
        logger.info(f"Steering stats: {stats['steered']}/{stats['calls']} steps steered")

    return audio.squeeze()


def generate_comparison(
    prompt: str = "",
    property_name: str = "bass",
    alpha: float = 1.5,
    steps: int = 100,
    cfg_scale: float = 7.0,
    seed: int = 42,
    sample_size: int = 65536,
    device: str = "cuda",
    apply_steering_at: str = "middle",
    # Pre-loaded models
    model: Optional[Any] = None,
    model_config: Optional[dict] = None,
    sae: Optional[AudioSae] = None,
    steering_vectors: Optional[ProbeSteeringVectors] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate comparison triplet: (less, baseline, more).

    Generates three versions of the same audio with different steering:
    - less: Negative alpha steering
    - baseline: No steering
    - more: Positive alpha steering

    Useful for A/B testing and demos.

    Args:
        prompt: Text prompt for generation
        property_name: Which property to steer
        alpha: Steering magnitude (applied as -alpha, 0, +alpha)
        ... (other args same as generate_steered_audio)

    Returns:
        Tuple of (audio_less, audio_baseline, audio_more)
    """
    # Load models once for efficiency
    if model is None:
        model, model_config = load_stable_audio(device)

    if sae is None:
        sae = load_sae("experiments/v2_main/checkpoints/sae_latest.pt", device)

    if steering_vectors is None:
        steering_vectors = ProbeSteeringVectors.load(
            "experiments/v2_main/eval/steering_vectors.npz"
        )

    # Generate less
    logger.info(f"Generating '{property_name}' comparison: less (-{alpha})")
    audio_less = generate_steered_audio(
        prompt=prompt,
        property_steering={property_name: -alpha},
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        sample_size=sample_size,
        device=device,
        apply_steering_at=apply_steering_at,
        model=model,
        model_config=model_config,
        sae=sae,
        steering_vectors=steering_vectors,
    )

    # Generate baseline
    logger.info("Generating baseline (no steering)")
    audio_baseline = generate_steered_audio(
        prompt=prompt,
        property_steering=None,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        sample_size=sample_size,
        device=device,
        model=model,
        model_config=model_config,
    )

    # Generate more
    logger.info(f"Generating '{property_name}' comparison: more (+{alpha})")
    audio_more = generate_steered_audio(
        prompt=prompt,
        property_steering={property_name: alpha},
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        sample_size=sample_size,
        device=device,
        apply_steering_at=apply_steering_at,
        model=model,
        model_config=model_config,
        sae=sae,
        steering_vectors=steering_vectors,
    )

    return audio_less, audio_baseline, audio_more
