"""Steered Drum Generation Demo — Gradio 6

Control what the AI generates using interpretable audio features.

This demo uses Sparse Autoencoders (SAEs) to steer the diffusion process
of Stable Audio Open. Adjust sliders to influence acoustic properties,
then compare steered output to an unsteered baseline (same seed).
"""

import logging
import random
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch

from drums_SAE.diffusion import generate_steered_audio, load_sae, load_stable_audio
from drums_SAE.steering.probe_steer import ProbeSteeringVectors

from .analysis import create_evolution_plot, format_comparison, measure_audio_properties
from .presets import (
    DEFAULT_CFG,
    DEFAULT_SCHEDULE,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    PROMPT_PRESETS,
    PROPERTIES,
    PROPERTY_NAMES,
    SAE_CHECKPOINT,
    SAMPLE_RATE,
    SLIDER_RANGE,
    SLIDER_STEP,
    STEERING_VECTORS_PATH,
)
from .tracking import create_tracking_callback, get_step_schedule

logger = logging.getLogger(__name__)

# === Global Model Cache ===
# Load once at startup, reuse for all generations
_model_cache = {
    "model": None,
    "model_config": None,
    "sae": None,
    "vectors": None,
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

    # Load Stable Audio Open
    logger.info("Loading Stable Audio Open model...")
    model, config = load_stable_audio(device)
    _model_cache["model"] = model
    _model_cache["model_config"] = config

    # Load SAE
    logger.info("Loading SAE...")
    project_root = Path(__file__).parent.parent.parent
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


def generate_comparison(
    prompt: str,
    bass: float,
    brightness: float,
    loudness: float,
    punchiness: float,
    steps: int,
    cfg: float,
    seed: int,
    schedule: str,
    track_evolution: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    """Generate baseline and steered audio for comparison.

    Args:
        prompt: Text prompt for generation
        bass: Bass steering (-3 to +3)
        brightness: Brightness steering
        loudness: Loudness steering
        punchiness: Punchiness steering
        steps: Diffusion steps
        cfg: Classifier-free guidance scale
        seed: Random seed
        schedule: Steering schedule ("all", "early", "middle", "late")
        track_evolution: Whether to track property evolution
        progress: Gradio progress callback

    Returns:
        Tuple of (baseline_audio, steered_audio, comparison_md, evolution_plot)
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
    sample_rate = _model_cache["model_config"].get("sample_rate", SAMPLE_RATE)

    # Get step schedule
    apply_at_steps = get_step_schedule(steps, schedule)

    # Initialize data for evolution plot
    baseline_steps = []
    steered_steps = []
    evolution_plot = None

    # === GENERATE BASELINE ===
    progress(0.05, desc="[1/2] Generating BASELINE (no steering)...")

    if track_evolution:
        # Use tracking callback for baseline (no steering)
        baseline_callback, baseline_steps = create_tracking_callback(
            sae=_model_cache["sae"],
            steering_vectors=_model_cache["vectors"],
            property_alphas=None,  # No steering
            apply_at_steps=None,
        )
        baseline_audio = generate_steered_audio(
            prompt=prompt,
            property_steering={},  # No steering
            steps=steps,
            cfg_scale=cfg,
            seed=seed,
            device=device,
            model=_model_cache["model"],
            model_config=_model_cache["model_config"],
            sae=_model_cache["sae"],
            steering_vectors=_model_cache["vectors"],
            callback=baseline_callback,
        )
    else:
        baseline_audio = generate_steered_audio(
            prompt=prompt,
            property_steering={},  # No steering
            steps=steps,
            cfg_scale=cfg,
            seed=seed,
            device=device,
            model=_model_cache["model"],
            model_config=_model_cache["model_config"],
        )

    # === GENERATE STEERED ===
    steering_desc = (
        ", ".join(f"{k}={v:+.1f}" for k, v in steering.items()) if steering else "none"
    )
    progress(0.5, desc=f"[2/2] Generating STEERED ({steering_desc})...")

    if steering:
        if track_evolution:
            # Use tracking callback with steering
            steered_callback, steered_steps = create_tracking_callback(
                sae=_model_cache["sae"],
                steering_vectors=_model_cache["vectors"],
                property_alphas=steering,
                apply_at_steps=apply_at_steps,
            )
            steered_audio = generate_steered_audio(
                prompt=prompt,
                property_steering={},  # Steering happens in callback
                steps=steps,
                cfg_scale=cfg,
                seed=seed,
                device=device,
                model=_model_cache["model"],
                model_config=_model_cache["model_config"],
                sae=_model_cache["sae"],
                steering_vectors=_model_cache["vectors"],
                callback=steered_callback,
            )
        else:
            steered_audio = generate_steered_audio(
                prompt=prompt,
                property_steering=steering,
                steps=steps,
                cfg_scale=cfg,
                seed=seed,
                apply_steering_at=schedule,
                device=device,
                model=_model_cache["model"],
                model_config=_model_cache["model_config"],
                sae=_model_cache["sae"],
                steering_vectors=_model_cache["vectors"],
            )
    else:
        # No steering requested, copy baseline
        steered_audio = baseline_audio.clone()
        steered_steps = baseline_steps.copy() if track_evolution else []

    # === ANALYZE RESULTS ===
    progress(0.9, desc="Analyzing and comparing results...")

    # Convert to numpy and ensure mono 1D array
    baseline_np = baseline_audio.cpu().numpy()
    steered_np = steered_audio.cpu().numpy()

    # Handle various output shapes from the model
    # Could be (samples,), (1, samples), (2, samples), or (batch, channels, samples)
    def to_mono_1d(arr):
        """Convert any audio array to mono 1D."""
        arr = arr.squeeze()  # Remove singleton dims
        if arr.ndim == 1:
            return arr
        elif arr.ndim == 2:
            # (channels, samples) -> take first channel or average
            if arr.shape[0] <= 2:  # channels first
                return arr[0]  # Take first channel
            else:  # samples first (unlikely but handle it)
                return arr[:, 0]
        else:
            # Flatten as last resort
            return arr.flatten()

    # Debug: log raw tensor stats before any processing
    logger.info(
        f"RAW tensor - baseline shape: {baseline_np.shape}, min: {baseline_np.min():.4f}, max: {baseline_np.max():.4f}, std: {baseline_np.std():.4f}"
    )
    logger.info(
        f"RAW tensor - steered shape: {steered_np.shape}, min: {steered_np.min():.4f}, max: {steered_np.max():.4f}, std: {steered_np.std():.4f}"
    )

    baseline_np = to_mono_1d(baseline_np)
    steered_np = to_mono_1d(steered_np)

    logger.info(
        f"AFTER to_mono - baseline shape: {baseline_np.shape}, min: {baseline_np.min():.4f}, max: {baseline_np.max():.4f}"
    )
    logger.info(
        f"AFTER to_mono - steered shape: {steered_np.shape}, min: {steered_np.min():.4f}, max: {steered_np.max():.4f}"
    )

    # Normalize to [-1, 1] range to prevent clipping issues
    def normalize_audio(arr):
        max_val = np.abs(arr).max()
        if max_val > 0:
            return arr / max_val * 0.95  # Leave some headroom
        return arr

    baseline_np = normalize_audio(baseline_np.astype(np.float32))
    steered_np = normalize_audio(steered_np.astype(np.float32))

    logger.info(
        f"FINAL - baseline: {baseline_np.shape}, min: {baseline_np.min():.4f}, max: {baseline_np.max():.4f}, std: {baseline_np.std():.4f}"
    )
    logger.info(
        f"FINAL - steered: {steered_np.shape}, min: {steered_np.min():.4f}, max: {steered_np.max():.4f}, std: {steered_np.std():.4f}"
    )
    logger.info(f"SAMPLE RATE: {sample_rate}, type: {type(sample_rate)}")

    # Measure properties
    baseline_props = measure_audio_properties(baseline_np, sample_rate)
    steered_props = measure_audio_properties(steered_np, sample_rate)

    # Format comparison
    comparison_md = format_comparison(baseline_props, steered_props, steering)

    if not steering:
        comparison_md = "*No steering applied — both outputs are identical.*"

    # Create evolution plot if tracking
    if track_evolution and baseline_steps and steered_steps:
        evolution_plot = create_evolution_plot(
            baseline_steps, steered_steps, steering, schedule
        )

    progress(1.0, desc="Done!")

    # Status message
    steering_str = (
        ", ".join(f"{k}={v:+.1f}" for k, v in steering.items()) if steering else "none"
    )
    status_md = f"**Generation complete.** Seed: {seed}, Steering: {steering_str}"

    return (
        (sample_rate, baseline_np),
        (sample_rate, steered_np),
        comparison_md,
        evolution_plot,
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

            This demo uses Sparse Autoencoders (SAEs) to steer the diffusion process
            of Stable Audio Open. Adjust the sliders to influence the generated sound's
            acoustic properties — then compare the steered output to an unsteered baseline.

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
                schedule = gr.Dropdown(
                    choices=["all", "early", "middle", "late"],
                    value=DEFAULT_SCHEDULE,
                    label="Steering Schedule",
                    info="When to apply steering during diffusion",
                )

        # === Evolution Tracking ===
        with gr.Accordion("Generation Visualization", open=False):
            track_evolution = gr.Checkbox(
                label="Track property evolution",
                value=False,
                info="Shows how properties crystallize during diffusion (adds ~20% generation time)",
            )
            evolution_plot = gr.Plot(
                label="Property Evolution",
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
                schedule,
                track_evolution,
            ],
            outputs=[
                baseline_audio,
                steered_audio,
                comparison_display,
                evolution_plot,
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
