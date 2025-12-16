"""Steered Drum Generation Demo — Gradio 6

Control what the AI generates using interpretable audio features.

This demo uses Sparse Autoencoders (SAEs) to steer the generation of
Stable Audio Open. Adjust sliders to influence acoustic properties,
then compare steered output to an unsteered baseline (same seed).

Architecture:
    Text → Diffusion → Clean Latents → Normalize → SAE Steer → Denormalize → VAE → Audio

Key insight: SAE steering is applied POST-HOC on clean latents in the
normalized space (same space SAE was trained on).

Usage:
    # Recommended: run via module
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib uv run python -m demo.generation.run

    # Or direct run (this file handles path setup)
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib uv run python demo/generation/app.py
"""

import logging
import random
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from einops import rearrange

# Handle imports for both direct run and module import
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent

# Add paths if running directly
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Generation: text → latents
from drums_SAE.generation import load_models as load_stable_audio_models, load_sae, generate_latents

# Steering: SHARED module (same as VAE decode demo!)
from drums_SAE.steering import steer_with_probe, ProbeSteeringVectors

# VAE decode: SHARED module (same as VAE decode demo!)
from drums_SAE.vae import decode_latents_to_audio, load_latent_stats, load_vae

# Local imports - handle both relative and absolute
try:
    from .analysis import format_comparison, measure_audio_properties
    from .presets import (
        DEFAULT_CFG,
        DEFAULT_SEED,
        DEFAULT_STEPS,
        PROMPT_PRESETS,
        SAE_CHECKPOINT,
        SAMPLE_RATE,
        SLIDER_RANGE,
        SLIDER_STEP,
        STEERING_VECTORS_PATH,
    )
except ImportError:
    # Running directly, not as module
    from demo.generation.analysis import format_comparison, measure_audio_properties
    from demo.generation.presets import (
        DEFAULT_CFG,
        DEFAULT_SEED,
        DEFAULT_STEPS,
        PROMPT_PRESETS,
        SAE_CHECKPOINT,
        SAMPLE_RATE,
        SLIDER_RANGE,
        SLIDER_STEP,
        STEERING_VECTORS_PATH,
    )

logger = logging.getLogger(__name__)

# === Global Model Cache ===
# Load once at startup, reuse for all generations
_model_cache = {
    "model": None,           # Stable Audio diffusion model
    "model_config": None,
    "vae": None,             # VAE decoder (raw autoencoder)
    "sample_rate": None,
    "sae": None,             # Trained SAE
    "vectors": None,         # Probe steering vectors
    "latent_mean": None,     # Normalization mean
    "latent_std": None,      # Normalization std
    "device": None,
    "loaded": False,
}


def load_models(device: str = "mps") -> None:
    """Load all models into cache. Called once at startup.

    Args:
        device: Device to load models to ("cuda", "mps", or "cpu")
    """
    global _model_cache

    if _model_cache["loaded"]:
        logger.info("Models already loaded, skipping")
        return

    logger.info(f"Loading models to device: {device}")
    project_root = Path(__file__).parent.parent.parent

    # Load Stable Audio Open (diffusion model only)
    logger.info("Loading Stable Audio Open model...")
    model, config, sample_rate = load_stable_audio_models(device)
    _model_cache["model"] = model
    _model_cache["model_config"] = config
    _model_cache["sample_rate"] = sample_rate

    # Load VAE using shared module (same as VAE decode demo)
    logger.info("Loading VAE decoder...")
    _model_cache["vae"] = load_vae(device)

    # Load normalization stats (critical for correct steering!)
    logger.info("Loading normalization statistics...")
    latents_path = project_root / "data" / "latents_v2.npz"
    _model_cache["latent_mean"], _model_cache["latent_std"] = load_latent_stats(
        str(latents_path), device
    )

    # Load SAE
    logger.info("Loading SAE...")
    sae_path = project_root / SAE_CHECKPOINT
    _model_cache["sae"] = load_sae(str(sae_path), device)

    # Load steering vectors
    logger.info("Loading steering vectors...")
    vectors_path = project_root / STEERING_VECTORS_PATH
    _model_cache["vectors"] = ProbeSteeringVectors.load(str(vectors_path))

    _model_cache["device"] = device
    _model_cache["loaded"] = True
    logger.info("All models loaded successfully!")


def get_random_prompt() -> str:
    """Return a random prompt preset."""
    return random.choice(PROMPT_PRESETS)


def get_random_seed() -> int:
    """Return a random seed."""
    return random.randint(0, 2**31 - 1)


def steer_and_decode(
    latents: torch.Tensor,
    steering: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply steering and decode both baseline and steered audio.

    This function uses the SAME steering + decode code as the VAE decode demo,
    ensuring consistent behavior across both demos.

    Args:
        latents: Clean latents from diffusion, shape (B, C, T) = (B, 64, T)
        steering: Dict of property_name → alpha values

    Returns:
        Tuple of (baseline_audio, steered_audio) as numpy arrays
    """
    sae = _model_cache["sae"]
    vectors = _model_cache["vectors"]
    vae = _model_cache["vae"]
    mean = _model_cache["latent_mean"]
    std = _model_cache["latent_std"]

    B, C, T = latents.shape

    # Reshape latents: (B, C, T) → (B*T, C) for SAE processing
    z_flat = rearrange(latents, "b c t -> (b t) c")

    # === NORMALIZE to SAE's training distribution ===
    # This is critical! SAE was trained on normalized latents.
    z_norm = (z_flat - mean) / (std + 1e-8)

    # === DECODE BASELINE (no steering) ===
    # Reshape for decode: (B*T, C) → (B, T, C)
    z_baseline = rearrange(z_norm, "(b t) c -> b t c", b=B, t=T)
    baseline_audio = decode_latents_to_audio(z_baseline, vae, mean, std)

    # === STEER AND DECODE ===
    if steering:
        z_steered = z_norm.clone().float()

        # Apply steering for each property using SHARED steering code
        for prop_name, alpha in steering.items():
            if abs(alpha) < 0.01:
                continue

            direction = vectors.get_direction(prop_name)
            logger.debug(f"Steering {prop_name} by {alpha:+.2f}")

            # steer_with_probe handles: encode → steer → RMS norm → decode + residual
            z_steered = steer_with_probe(
                z=z_steered,
                sae=sae,
                direction=direction,
                alpha=alpha,
                preserve_residual=True,
            )

        # Reshape for decode: (B*T, C) → (B, T, C)
        z_steered_reshaped = rearrange(z_steered, "(b t) c -> b t c", b=B, t=T)
        steered_audio = decode_latents_to_audio(z_steered_reshaped, vae, mean, std)
    else:
        # No steering - same as baseline
        steered_audio = baseline_audio

    return baseline_audio, steered_audio


def generate_comparison(
    prompt: str,
    bass: float,
    brightness: float,
    loudness: float,
    punchiness: float,
    steps: int,
    cfg: float,
    seed: int,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    """Generate baseline and steered audio for comparison.

    Pipeline:
    1. Generate clean latents from prompt (return_latents=True)
    2. Normalize latents to SAE's training distribution
    3. Decode baseline (no steering)
    4. Steer latents using SAE + probe directions
    5. Decode steered latents

    Args:
        prompt: Text prompt for generation
        bass: Bass steering (-3 to +3)
        brightness: Brightness steering
        loudness: Loudness steering
        punchiness: Punchiness steering
        steps: Diffusion steps
        cfg: Classifier-free guidance scale
        seed: Random seed
        progress: Gradio progress callback

    Returns:
        Tuple of (baseline_audio, steered_audio, comparison_md, status_md)
    """
    if not _model_cache["loaded"]:
        raise gr.Error("Models not loaded. Please wait for startup to complete.")

    # Build steering dict (filter out zeros)
    steering = {
        "bass": bass,
        "spectral_centroid": brightness,
        "rms": loudness,
        "crest_factor": punchiness,
    }
    steering = {k: v for k, v in steering.items() if abs(v) > 0.01}

    device = _model_cache["device"]
    sample_rate = _model_cache["sample_rate"]

    # === STEP 1: GENERATE CLEAN LATENTS ===
    progress(0.1, desc="Generating latents from prompt...")

    latents = generate_latents(
        model=_model_cache["model"],
        prompt=prompt,
        seconds=2.0,  # ~88200 samples at 44.1kHz
        steps=int(steps),
        cfg_scale=cfg,
        seed=int(seed),
        device=device,
    )
    logger.info(f"Generated latents shape: {latents.shape}")

    # === STEP 2-5: STEER AND DECODE (using shared code!) ===
    steering_desc = (
        ", ".join(f"{k}={v:+.1f}" for k, v in steering.items()) if steering else "none"
    )

    progress(0.4, desc="Decoding baseline...")
    if steering:
        progress(0.6, desc=f"Steering latents ({steering_desc})...")

    baseline_np, steered_np = steer_and_decode(latents, steering)

    # === ANALYZE RESULTS ===
    progress(0.9, desc="Analyzing and comparing results...")

    logger.info(
        f"FINAL - baseline: {baseline_np.shape}, std: {baseline_np.std():.4f}"
    )
    logger.info(
        f"FINAL - steered: {steered_np.shape}, std: {steered_np.std():.4f}"
    )

    # Measure properties
    baseline_props = measure_audio_properties(baseline_np, sample_rate)
    steered_props = measure_audio_properties(steered_np, sample_rate)

    # Format comparison
    comparison_md = format_comparison(baseline_props, steered_props, steering)

    if not steering:
        comparison_md = "*No steering applied — both outputs are identical.*"

    progress(1.0, desc="Done!")

    # Status message
    status_md = f"**Generation complete.** Seed: {seed}, Steering: {steering_desc}"

    return (
        (sample_rate, baseline_np),
        (sample_rate, steered_np),
        comparison_md,
        status_md,
    )


def create_demo() -> gr.Blocks:
    """Create and return the Gradio Blocks demo."""

    with gr.Blocks(title="Steered Drum Generation") as demo:
        # === Header ===
        gr.Markdown(
            """
            # Steered Drum Generation

            **Control what the AI generates** using interpretable audio features.

            This demo uses Sparse Autoencoders (SAEs) to steer audio generation.
            Adjust the sliders to influence the generated sound's acoustic properties —
            then compare the steered output to an unsteered baseline.

            *Same prompt + same seed = only difference is steering.*
            """
        )

        # === Prompt Section ===
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the drum sound you want...",
                value="punchy kick drum",
                scale=4,
            )
            with gr.Column(scale=1):
                random_prompt_btn = gr.Button("Random Prompt")
                random_seed_btn = gr.Button("Random Seed")

        # === Property Sliders ===
        gr.Markdown("### Property Steering")
        gr.Markdown(
            "*Adjust sliders to steer generation. 0 = no change, positive = more, negative = less.*"
        )

        with gr.Row():
            bass = gr.Slider(
                minimum=SLIDER_RANGE[0],
                maximum=SLIDER_RANGE[1],
                value=0,
                step=SLIDER_STEP,
                label="Bass (less <-> more)",
            )
            brightness = gr.Slider(
                minimum=SLIDER_RANGE[0],
                maximum=SLIDER_RANGE[1],
                value=0,
                step=SLIDER_STEP,
                label="Brightness (darker <-> brighter)",
            )

        with gr.Row():
            loudness = gr.Slider(
                minimum=SLIDER_RANGE[0],
                maximum=SLIDER_RANGE[1],
                value=0,
                step=SLIDER_STEP,
                label="Loudness (quieter <-> louder)",
            )
            punchiness = gr.Slider(
                minimum=SLIDER_RANGE[0],
                maximum=SLIDER_RANGE[1],
                value=0,
                step=SLIDER_STEP,
                label="Punchiness (sustained <-> transient)",
            )

        # === Advanced Settings ===
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                steps = gr.Slider(
                    minimum=10,
                    maximum=150,
                    value=DEFAULT_STEPS,
                    step=10,
                    label="Diffusion Steps",
                )
                cfg = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    value=DEFAULT_CFG,
                    step=0.5,
                    label="CFG Scale",
                )
            with gr.Row():
                seed = gr.Number(
                    value=DEFAULT_SEED,
                    label="Seed",
                    precision=0,
                )

        # === Generate Button ===
        generate_btn = gr.Button(
            "Generate Comparison",
            variant="primary",
            size="lg",
        )

        # === Status Indicator ===
        status_text = gr.Markdown(
            "*Ready to generate. Click the button above to start.*", elem_id="status"
        )

        # === Results Section ===
        gr.Markdown("---")
        gr.Markdown("### Results")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Baseline** (no steering)")
                baseline_audio = gr.Audio(
                    label="Baseline",
                    type="numpy",
                    autoplay=False,
                )
            with gr.Column():
                gr.Markdown("**Steered** (with your settings)")
                steered_audio = gr.Audio(
                    label="Steered",
                    type="numpy",
                    autoplay=True,  # Auto-play the steered version
                )

        comparison_display = gr.Markdown(
            "*Click 'Generate Comparison' to create audio and see measured differences.*"
        )

        # === Footer ===
        gr.Markdown(
            """
            ---
            *Built with [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0)
            and SAE-based steering. Based on
            [Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](https://arxiv.org/abs/2510.23802).*
            """
        )

        # === Event Handlers ===

        random_prompt_btn.click(
            fn=get_random_prompt,
            outputs=prompt,
        )

        random_seed_btn.click(
            fn=get_random_seed,
            outputs=seed,
        )

        generate_btn.click(
            fn=generate_comparison,
            inputs=[
                prompt,
                bass,
                brightness,
                loudness,
                punchiness,
                steps,
                cfg,
                seed,
            ],
            outputs=[
                baseline_audio,
                steered_audio,
                comparison_display,
                status_text,
            ],
        )

    return demo


# === Main ===

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument(
        "--port", type=int, default=None, help="Port (auto if not specified)"
    )
    args = parser.parse_args()

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load models before launching
    load_models(device=device)

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_port=args.port,
        share=args.share,
    )
