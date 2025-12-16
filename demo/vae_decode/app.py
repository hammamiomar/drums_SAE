"""
Drums SAE Steering Lab — Probe-Based Audio Control
====================================================

Research demo for evaluating and steering drum sounds using
probe-based steering vectors. Supports multiple models with
automatic steering vector loading.

Usage:
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib uv run python demo/app.py
"""

import sys
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import pandas as pd
import torch

# Add paths for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from drums_SAE.sae.model import AudioSae
from drums_SAE.steering import (
    ProbeSteeringVectors,
    create_steered_triplet,
)
# VAE utilities now in shared package location
from drums_SAE.vae import DEVICE, decode_latents_to_audio, load_latent_stats, load_vae
from demo.shared.viz import create_triplet_spectrogram, create_triplet_waveform


# =============================================================================
# Configuration
# =============================================================================

SAMPLE_RATE = 44100

# Strength slider (positive only - triplets apply ±)
STRENGTH_MIN = 0.5
STRENGTH_MAX = 6.0
STRENGTH_DEFAULT = 1.5

# Sample settings
MAX_SAMPLES = 3
DEFAULT_SAMPLES = 3

# Human-readable property names
PROPERTY_DISPLAY = {
    "spectral_centroid": "Brightness",
    "rms": "Loudness",
    "crest_factor": "Punchiness",
    "bass": "Body/Warmth",
    "brightness": "Brightness",
    "loudness": "Loudness",
    "boominess": "Boominess",
    "hardness": "Hardness",
    "depth": "Depth",
}

# Preferred order for properties
PROPERTY_ORDER = [
    "spectral_centroid", "rms", "crest_factor", "bass",  # V2
    "brightness", "loudness", "boominess", "hardness", "depth",  # V1
]


def sort_properties(props: list[str]) -> list[str]:
    """Sort properties in preferred display order."""
    def key(p):
        try:
            return PROPERTY_ORDER.index(p)
        except ValueError:
            return len(PROPERTY_ORDER)
    return sorted(props, key=key)


def get_property_label(prop: str) -> str:
    """Get label showing both display name and internal name."""
    display = PROPERTY_DISPLAY.get(prop, prop.replace("_", " ").title())
    return f"{display} ({prop})"


# =============================================================================
# Audio Measurement and Comparison
# =============================================================================

def measure_audio(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Measure acoustic properties from audio for verification.

    Computes the same features our probes were trained on, allowing
    verification that steering changes properties as expected.
    """
    if audio is None:
        return {}

    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Spectral centroid (brightness)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))

    # RMS energy
    rms = float(np.mean(librosa.feature.rms(y=audio)))

    # Bass energy (< 200 Hz)
    stft = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = freqs < 200
    bass = float(np.mean(stft[bass_mask, :])) if bass_mask.any() else 0.0

    # Crest factor
    peak = np.max(np.abs(audio))
    crest = float(peak / (rms + 1e-8))

    return {
        "spectral_centroid": centroid,
        "rms": rms,
        "bass": bass,
        "crest_factor": crest,
    }


def format_triplet_comparison(
    less: dict[str, float],
    orig: dict[str, float],
    more: dict[str, float],
    property_steered: str,
    strength: float,
) -> str:
    """Format a markdown table comparing LESS / ORIGINAL / MORE triplet.

    Args:
        less: Measured properties of LESS audio (-strength)
        orig: Measured properties of ORIGINAL audio
        more: Measured properties of MORE audio (+strength)
        property_steered: Which property was steered
        strength: Steering strength applied

    Returns:
        Markdown table string
    """
    lines = [
        f"### Verification: Steering `{property_steered}` at ±{strength:.1f}",
        "",
        "| Property | LESS | ORIGINAL | MORE | Less→More |",
        "|----------|------|----------|------|-----------|",
    ]

    props_to_show = ["spectral_centroid", "rms", "bass", "crest_factor"]
    names = {
        "spectral_centroid": "Brightness",
        "rms": "Loudness",
        "bass": "Bass",
        "crest_factor": "Punchiness",
    }

    for prop in props_to_show:
        l = less.get(prop, 0)
        o = orig.get(prop, 0)
        m = more.get(prop, 0)
        name = names.get(prop, prop)

        # Is this the steered property?
        is_steered = prop == property_steered or (
            property_steered == "brightness" and prop == "spectral_centroid"
        ) or (
            property_steered == "loudness" and prop == "rms"
        )
        marker = " ⬅" if is_steered else ""

        # Calculate less→more change
        if abs(o) > 1e-8:
            pct = ((m - l) / abs(o)) * 100
            direction = "+" if pct > 0 else ""
            change_str = f"{direction}{pct:.0f}%"
        else:
            change_str = "N/A"

        # Format values
        if prop == "spectral_centroid":
            l_str = f"{l:.0f} Hz"
            o_str = f"{o:.0f} Hz"
            m_str = f"{m:.0f} Hz"
        elif prop in ("rms", "bass"):
            l_str = f"{l:.4f}"
            o_str = f"{o:.4f}"
            m_str = f"{m:.4f}"
        else:
            l_str = f"{l:.2f}"
            o_str = f"{o:.2f}"
            m_str = f"{m:.2f}"

        lines.append(f"| {name}{marker} | {l_str} | {o_str} | {m_str} | {change_str} |")

    return "\n".join(lines)


# =============================================================================
# Model Discovery and Loading
# =============================================================================

def discover_models() -> list[str]:
    """Find all experiments with steering vectors."""
    models = []
    experiments_dir = PROJECT_ROOT / "experiments"

    if not experiments_dir.exists():
        print("[WARN] No experiments directory found")
        return models

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        vectors_path = exp_dir / "eval" / "steering_vectors.npz"
        if vectors_path.exists():
            models.append(exp_dir.name)

    # Sort with v2_main first
    return sorted(models, key=lambda x: (not x.startswith("v2_main"), x))


def detect_version(model_name: str) -> str:
    """Detect v1/v2 from model name."""
    if model_name.startswith("v1"):
        return "v1"
    return "v2"


def get_data_paths(version: str) -> tuple[str, str | None]:
    """Get latent and features paths for version."""
    if version == "v1":
        return "data/drums_encoded.npz", None
    return "data/latents_v2.npz", "data/features_v2.csv"


def load_sae_checkpoint(checkpoint_path: Path) -> AudioSae:
    """Load SAE from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]

    sae = AudioSae(
        d_input=cfg["d_input"],
        expansion_factor=cfg["expansion_factor"],
        topk=cfg["topk"],
        topk_aux=cfg.get("topk_aux", 128),
        dead_threshold=cfg.get("dead_threshold", 10000),
    ).to(DEVICE)
    sae.load_state_dict(checkpoint["model_state_dict"])
    sae.training = False

    print(f"[SAE] Loaded: {cfg['d_input']} → {cfg['d_input'] * cfg['expansion_factor']} features")
    return sae


def load_latents(
    latents_path: str,
    features_path: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], int]:
    """Load and normalize latents, return sample indices."""
    latent_data = np.load(PROJECT_ROOT / latents_path)
    all_latents = torch.from_numpy(latent_data["latents"]).float().to(DEVICE)
    latent_mean, latent_std = load_latent_stats(str(PROJECT_ROOT / latents_path), DEVICE)

    # Normalize
    all_latents_norm = (all_latents - latent_mean) / (latent_std + 1e-8)

    # Determine timesteps per sample
    n_timesteps = 16 if "v2" in latents_path else 32
    n_total_samples = len(all_latents) // n_timesteps

    # Filter to non-silence samples if features available
    if features_path is not None:
        features_df = pd.read_csv(PROJECT_ROOT / features_path)
        non_silence_mask = ~features_df["is_silence"].values

        sample_has_content = []
        for i in range(n_total_samples):
            start = i * n_timesteps
            end = start + n_timesteps
            if non_silence_mask[start:end].any():
                sample_has_content.append(i)
        sample_indices = sample_has_content
    else:
        sample_indices = list(range(n_total_samples))

    print(f"[DATA] {len(sample_indices)} samples with content")
    return all_latents_norm, latent_mean, latent_std, sample_indices, n_timesteps


def load_model(model_name: str, vae) -> dict | None:
    """Load all components for a model."""
    exp_dir = PROJECT_ROOT / "experiments" / model_name

    # Find checkpoint
    checkpoint_path = exp_dir / "checkpoints" / "sae_latest.pt"
    if not checkpoint_path.exists():
        ckpts = list((exp_dir / "checkpoints").glob("sae_*.pt"))
        if ckpts:
            checkpoint_path = sorted(ckpts)[-1]
        else:
            print(f"[ERROR] No checkpoint found for {model_name}")
            return None

    # Load steering vectors
    vectors_path = exp_dir / "eval" / "steering_vectors.npz"
    if not vectors_path.exists():
        print(f"[ERROR] No steering vectors for {model_name}")
        return None

    print(f"\n[LOAD] Model: {model_name}")

    sae = load_sae_checkpoint(checkpoint_path)
    vectors = ProbeSteeringVectors.load(str(vectors_path))
    print(f"[STEER] Properties: {vectors.properties}")

    version = detect_version(model_name)
    latents_path, features_path = get_data_paths(version)
    latents_norm, latent_mean, latent_std, sample_indices, n_timesteps = load_latents(
        latents_path, features_path
    )

    return {
        "name": model_name,
        "sae": sae,
        "steering": vectors,
        "latents_norm": latents_norm,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "sample_indices": sample_indices,
        "n_timesteps": n_timesteps,
        "version": version,
        "vae": vae,
    }


# =============================================================================
# Steering Functions
# =============================================================================

def get_sample_latents(models: dict, sample_idx: int) -> torch.Tensor:
    """Get normalized latents for a single sample."""
    n_timesteps = models["n_timesteps"]
    start = sample_idx * n_timesteps
    end = start + n_timesteps
    return models["latents_norm"][start:end]


def decode_to_audio(z_norm: torch.Tensor, models: dict) -> np.ndarray | None:
    """Decode normalized latents to audio."""
    return decode_latents_to_audio(
        z_norm,
        models["vae"],
        models["latent_mean"],
        models["latent_std"],
    )


def generate_triplet(
    models: dict,
    sample_idx: int,
    property_name: str,
    strength: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Generate (less, original, more) audio triplet."""
    z = get_sample_latents(models, sample_idx)

    z_less, z_orig, z_more = create_steered_triplet(
        z, models["sae"], models["steering"], property_name, strength
    )

    audio_less = decode_to_audio(z_less, models)
    audio_orig = decode_to_audio(z_orig, models)
    audio_more = decode_to_audio(z_more, models)

    return audio_less, audio_orig, audio_more


# =============================================================================
# Gradio Interface
# =============================================================================

def build_interface() -> gr.Blocks:
    """Build the Gradio interface."""

    available_models = discover_models()
    if not available_models:
        print("[ERROR] No models with steering vectors found!")
        available_models = ["(none)"]

    default_model = available_models[0]

    print("\n[INIT] Loading VAE...")
    vae = load_vae(DEVICE)

    print(f"\n[INIT] Loading default model: {default_model}")
    initial_models = load_model(default_model, vae) if default_model != "(none)" else None

    # Mutable container for current model
    model_container = {"current": initial_models, "vae": vae}

    with gr.Blocks(title="DRUMS SAE STEERING LAB") as demo:

        # =====================================================================
        # Header
        # =====================================================================
        gr.Markdown("# 🔬 DRUMS SAE STEERING LAB")
        gr.Markdown(
            "Probe-based steering for interpretable drum sound control. "
            "Triplets show **LESS** (−strength) / **ORIGINAL** / **MORE** (+strength)."
        )

        # =====================================================================
        # Control Panel
        # =====================================================================
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=default_model,
                    label="MODEL",
                    info="Experiment with steering vectors",
                )

            with gr.Column(scale=1):
                initial_props = sort_properties(initial_models["steering"].properties) if initial_models else []
                initial_choices = [(get_property_label(p), p) for p in initial_props]
                property_dropdown = gr.Dropdown(
                    choices=initial_choices,
                    value=initial_props[0] if initial_props else None,
                    label="PROPERTY",
                    info="Acoustic property to steer",
                )

            with gr.Column(scale=2):
                strength_slider = gr.Slider(
                    minimum=STRENGTH_MIN,
                    maximum=STRENGTH_MAX,
                    value=STRENGTH_DEFAULT,
                    step=0.1,
                    label="STEERING STRENGTH",
                    info="Applied as ± to create LESS/MORE triplets",
                )

        with gr.Row():
            n_samples_slider = gr.Slider(
                minimum=1,
                maximum=MAX_SAMPLES,
                value=DEFAULT_SAMPLES,
                step=1,
                label="SAMPLES",
            )
            seed_input = gr.Number(
                value=42,
                label="SEED",
                precision=0,
            )
            generate_btn = gr.Button("▶ GENERATE", variant="primary", scale=2)

        status_text = gr.Markdown("*Select model and click GENERATE to begin*")

        # =====================================================================
        # Results Area - 3 sample rows
        # =====================================================================
        gr.Markdown("---")

        sample_outputs = []

        for i in range(3):
            with gr.Group():
                gr.Markdown(f"### SAMPLE {i + 1}")
                with gr.Row():
                    less_audio = gr.Audio(label="LESS", type="numpy", interactive=False)
                    orig_audio = gr.Audio(label="ORIGINAL", type="numpy", interactive=False)
                    more_audio = gr.Audio(label="MORE", type="numpy", interactive=False)
                with gr.Row():
                    waveform_plot = gr.Plot(label="Waveform Comparison")
                    spectrogram_plot = gr.Plot(label="Spectrogram Comparison")
                # Comparison table showing what changed
                comparison_md = gr.Markdown("*Generate to see measured changes*")

            sample_outputs.append({
                "less": less_audio,
                "orig": orig_audio,
                "more": more_audio,
                "waveform": waveform_plot,
                "spectrogram": spectrogram_plot,
                "comparison": comparison_md,
            })

        # =====================================================================
        # Event Handlers
        # =====================================================================

        def on_model_change(model_name):
            """Load new model and update property dropdown."""
            if model_name == "(none)":
                model_container["current"] = None
                return gr.update(choices=[], value=None), "*No model selected*"

            print(f"\n[UI] Switching to model: {model_name}")
            new_models = load_model(model_name, model_container["vae"])

            if new_models is None:
                return gr.update(choices=[], value=None), f"*Failed to load {model_name}*"

            model_container["current"] = new_models

            props = sort_properties(new_models["steering"].properties)
            choices = [(get_property_label(p), p) for p in props]

            return (
                gr.update(choices=choices, value=props[0] if props else None),
                f"*Loaded {model_name} — {len(props)} properties*",
            )

        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[property_dropdown, status_text],
        )

        def on_generate(model_name, property_name, strength, n_samples, seed):
            """Generate batch of triplets."""
            models_dict = model_container["current"]

            print(f"\n[GEN] property={property_name}, strength={strength}, n_samples_raw={n_samples} (type={type(n_samples).__name__}), seed={seed}")

            if models_dict is None:
                empty_outputs = [None] * 18 + ["*No model loaded*"]
                return empty_outputs

            if property_name is None or property_name not in models_dict["steering"].properties:
                print(f"  [ERROR] Invalid property: {property_name}")
                print(f"  Available: {models_dict['steering'].properties}")
                empty_outputs = [None] * 18 + [f"*Invalid property: {property_name}*"]
                return empty_outputs

            n_samples = int(min(n_samples, 3))
            seed = int(seed)
            strength = float(strength)

            # Select random samples
            np.random.seed(seed)
            sample_indices = models_dict["sample_indices"]
            selected = np.random.choice(
                sample_indices,
                size=min(n_samples, len(sample_indices)),
                replace=False,
            )

            # Generate triplets
            outputs = []
            for i, sample_idx in enumerate(selected):
                print(f"  Sample {i + 1}/{n_samples} (idx={sample_idx})")

                audio_less, audio_orig, audio_more = generate_triplet(
                    models_dict, sample_idx, property_name, strength
                )

                # Create visualizations
                waveform_fig = create_triplet_waveform(
                    audio_less, audio_orig, audio_more, SAMPLE_RATE, strength
                )
                spec_fig = create_triplet_spectrogram(
                    audio_less, audio_orig, audio_more, SAMPLE_RATE, strength
                )

                # Measure audio properties and create comparison table
                less_props = measure_audio(audio_less, SAMPLE_RATE)
                orig_props = measure_audio(audio_orig, SAMPLE_RATE)
                more_props = measure_audio(audio_more, SAMPLE_RATE)

                comparison_table = format_triplet_comparison(
                    less_props, orig_props, more_props,
                    property_name, strength
                )

                # Print to console for verification
                print(f"    LESS centroid={less_props.get('spectral_centroid', 0):.0f} Hz")
                print(f"    ORIG centroid={orig_props.get('spectral_centroid', 0):.0f} Hz")
                print(f"    MORE centroid={more_props.get('spectral_centroid', 0):.0f} Hz")

                outputs.extend([
                    (SAMPLE_RATE, audio_less) if audio_less is not None else None,
                    (SAMPLE_RATE, audio_orig) if audio_orig is not None else None,
                    (SAMPLE_RATE, audio_more) if audio_more is not None else None,
                    waveform_fig,
                    spec_fig,
                    comparison_table,
                ])

            # Pad remaining rows
            for _ in range(3 - len(selected)):
                outputs.extend([None, None, None, None, None, "*No sample*"])

            prop_label = get_property_label(property_name)
            status = f"*Generated {len(selected)} triplets for **{prop_label}** at ±{strength:.1f}× strength*"
            outputs.append(status)

            return outputs

        # Build outputs list: 3 rows × 6 outputs + status
        generate_outputs = []
        for row in sample_outputs:
            generate_outputs.extend([
                row["less"], row["orig"], row["more"],
                row["waveform"], row["spectrogram"], row["comparison"]
            ])
        generate_outputs.append(status_text)

        generate_btn.click(
            fn=on_generate,
            inputs=[model_dropdown, property_dropdown, strength_slider, n_samples_slider, seed_input],
            outputs=generate_outputs,
        )

    return demo


# =============================================================================
# Main Entry
# =============================================================================

def main():
    """Launch the demo."""
    print("\n" + "=" * 60)
    print("🔬 DRUMS SAE STEERING LAB")
    print("=" * 60)

    demo = build_interface()

    print("\n[LAUNCH] Starting Gradio server...")
    demo.launch(share=False)


if __name__ == "__main__":
    main()
