"""
Drums SAE V2 — Steering Quality Tester

Bulk testing demo to check whether steering features actually
changes the audio in expected ways. Uses the Gytis residual trick
for quality preservation.

Usage:
    python -m demo.run --v2
"""

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch

# Add src to path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

from demo.shared.theme import BRUTALIST_CSS
from demo.shared.vae import DEVICE, decode_latents_to_audio, load_latent_stats, load_vae
from demo.shared.viz import (
    create_audio_triplet_display,
    create_empty_plot,
    create_spectrogram_comparison,
    create_waveform_overlay,
)
from demo.v2.config import CONFIG, PROJECT_ROOT
from drums_SAE.sae.model import AudioSae
from drums_SAE.steering.steer import steer_with_residual


# =============================================================================
# Model Loading
# =============================================================================


def load_v2_models() -> dict:
    """Load SAE, VAE, latents, and features for v2 demo."""
    print(f"\n[LOAD] Device: {DEVICE}")

    checkpoint_path = PROJECT_ROOT / CONFIG.checkpoint_path
    latent_data_path = PROJECT_ROOT / CONFIG.latent_data_path
    features_path = PROJECT_ROOT / CONFIG.features_path

    # Load SAE
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]

    sae = AudioSae(
        d_input=cfg["d_input"],
        expansion_factor=cfg["expansion_factor"],
        topk=cfg["topk"],
        topk_aux=cfg["topk_aux"],
        dead_threshold=cfg["dead_threshold"],
    ).to(DEVICE)
    sae.load_state_dict(checkpoint["model_state_dict"])
    sae.training = False  # Inference mode
    print(f"[LOAD] SAE: {cfg['d_input']} -> {cfg['d_input'] * cfg['expansion_factor']} features")

    # Load latents and stats
    latent_data = np.load(latent_data_path)
    all_latents = torch.from_numpy(latent_data["latents"]).float().to(DEVICE)
    latent_mean, latent_std = load_latent_stats(str(latent_data_path), DEVICE)

    # Normalize latents
    all_latents_norm = (all_latents - latent_mean) / (latent_std + 1e-8)

    # Load features for filtering and correlation analysis
    features_df = pd.read_csv(features_path)

    # Get non-silence sample indices
    non_silence_mask = ~features_df["is_silence"].values
    non_silence_indices = np.where(non_silence_mask)[0]

    # Group by sample (16 timesteps each)
    n_timesteps = CONFIG.n_timesteps
    n_total_samples = len(all_latents) // n_timesteps

    # Find samples with at least some non-silence content
    sample_has_content = []
    for i in range(n_total_samples):
        start = i * n_timesteps
        end = start + n_timesteps
        if non_silence_mask[start:end].any():
            sample_has_content.append(i)

    print(f"[LOAD] Samples with content: {len(sample_has_content)} / {n_total_samples}")

    # Load VAE
    vae = load_vae(DEVICE)

    # Compute feature-property correlations
    correlations = compute_feature_correlations(sae, all_latents_norm, features_df)

    return {
        "sae": sae,
        "vae": vae,
        "all_latents_norm": all_latents_norm,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "features_df": features_df,
        "sample_indices": sample_has_content,
        "n_timesteps": n_timesteps,
        "correlations": correlations,
    }


def compute_feature_correlations(
    sae: AudioSae,
    latents_norm: torch.Tensor,
    features_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Compute correlations between SAE features and audio properties.

    Returns dict mapping property name -> correlation array (n_features,)
    """
    print("[LOAD] Computing feature correlations...")

    # Encode all latents
    sae.training = False
    with torch.no_grad():
        enc = sae.encode(latents_norm, return_aux=False)
        activations = enc["f"].cpu().numpy()  # (n_samples, n_features)

    correlations = {}
    properties = ["spectral_centroid", "sub_bass", "bass", "crest_factor", "rms"]

    for prop in properties:
        if prop not in features_df.columns:
            continue

        values = features_df[prop].values
        valid_mask = ~np.isnan(values)

        if valid_mask.sum() < 100:
            continue

        # Compute correlation for each feature
        corrs = np.zeros(activations.shape[1])
        for f in range(activations.shape[1]):
            act_f = activations[valid_mask, f]
            val_f = values[valid_mask]

            if act_f.std() > 1e-8 and val_f.std() > 1e-8:
                corrs[f] = np.corrcoef(act_f, val_f)[0, 1]

        correlations[prop] = corrs
        max_idx = np.argmax(np.abs(corrs))
        print(f"  {prop}: max |r| = {np.abs(corrs).max():.3f} (feature {max_idx})")

    return correlations


def get_sample_latents(models: dict, sample_idx: int) -> torch.Tensor:
    """Get normalized latents for a single sample (all timesteps)."""
    n_timesteps = models["n_timesteps"]
    start = sample_idx * n_timesteps
    end = start + n_timesteps
    return models["all_latents_norm"][start:end]


def decode_to_audio(z_norm: torch.Tensor, models: dict) -> np.ndarray | None:
    """Decode normalized latents to audio."""
    return decode_latents_to_audio(
        z_norm,
        models["vae"],
        models["latent_mean"],
        models["latent_std"],
    )


# =============================================================================
# Steering Functions
# =============================================================================


def generate_steering_triplet(
    models: dict,
    sample_idx: int,
    feature_idx: int,
    steering_value: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Generate original, +steered, and -steered audio for a sample.

    Uses steer_with_residual() to preserve audio quality.
    """
    z_orig = get_sample_latents(models, sample_idx)

    # Original
    audio_orig = decode_to_audio(z_orig, models)

    # +Steered
    z_plus = steer_with_residual(z_orig, models["sae"], feature_idx, steering_value)
    audio_plus = decode_to_audio(z_plus, models)

    # -Steered
    z_minus = steer_with_residual(z_orig, models["sae"], feature_idx, -steering_value)
    audio_minus = decode_to_audio(z_minus, models)

    return audio_orig, audio_plus, audio_minus


def get_best_feature_for_property(models: dict, property_name: str) -> int | None:
    """Get the feature index with highest correlation to a property."""
    if property_name not in models["correlations"]:
        return None
    corrs = models["correlations"][property_name]
    return int(np.argmax(np.abs(corrs)))


def get_feature_stats(models: dict, feature_idx: int) -> dict:
    """Get statistics about a feature's correlations with properties."""
    stats = {"feature_idx": feature_idx}
    for prop, corrs in models["correlations"].items():
        stats[f"corr_{prop}"] = float(corrs[feature_idx])
    return stats


# =============================================================================
# Gradio Interface
# =============================================================================


def build_interface(models: dict) -> gr.Blocks:
    """Build the V2 bulk testing interface."""

    sample_indices = models["sample_indices"]
    available_props = list(models["correlations"].keys())
    n_features = models["sae"].d_hidden

    with gr.Blocks(title="DRUMS SAE V2 TESTER") as demo:

        # Header
        gr.Markdown("# DRUMS SAE V2 — STEERING QUALITY TESTER")
        gr.Markdown(
            "*Bulk test steering with the Gytis residual trick. "
            "Pick a feature, generate triplets, and listen for differences.*"
        )

        # =================================================================
        # Control Panel
        # =================================================================
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### FEATURE SELECTION")

                # Option 1: By property
                property_dropdown = gr.Dropdown(
                    choices=["(manual)"] + available_props,
                    value="sub_bass" if "sub_bass" in available_props else available_props[0],
                    label="SELECT BY PROPERTY",
                    info="Pick best feature for this property",
                )

                # Option 2: Manual index
                feature_input = gr.Number(
                    value=1852,
                    label="OR ENTER FEATURE INDEX",
                    precision=0,
                    minimum=0,
                    maximum=n_features - 1,
                )

                # Steering value
                steering_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.5,
                    step=0.1,
                    label="STEERING MAGNITUDE",
                    info="Applied as +value and -value",
                )

            with gr.Column(scale=1):
                gr.Markdown("### BATCH SETTINGS")

                n_samples_slider = gr.Slider(
                    minimum=1,
                    maximum=5,  # Fixed to match UI rows
                    value=5,
                    step=1,
                    label="NUMBER OF SAMPLES",
                )

                random_seed = gr.Number(
                    value=42,
                    label="RANDOM SEED",
                    precision=0,
                    info="For reproducible sample selection",
                )

                generate_btn = gr.Button("GENERATE BATCH", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### FEATURE STATS")
                stats_display = gr.Markdown("*Select feature to see stats*")

        # =================================================================
        # Results Area
        # =================================================================
        gr.Markdown("---")
        gr.Markdown("### RESULTS")

        # Create 5 fixed sample rows (simpler than dynamic visibility)
        sample_audios = []
        for i in range(5):
            gr.Markdown(f"**Sample {i+1}**")
            with gr.Row():
                orig_audio = gr.Audio(
                    label="Original",
                    type="numpy",
                    interactive=False,
                )
                plus_audio = gr.Audio(
                    label="+Steered",
                    type="numpy",
                    interactive=False,
                )
                minus_audio = gr.Audio(
                    label="-Steered",
                    type="numpy",
                    interactive=False,
                )
            sample_audios.append((orig_audio, plus_audio, minus_audio))

        status_text = gr.Markdown("*Click GENERATE BATCH to begin*")

        # =================================================================
        # Event Handlers
        # =================================================================

        def on_property_change(prop):
            """Update feature index when property is selected."""
            if prop == "(manual)":
                return gr.update()

            feature_idx = get_best_feature_for_property(models, prop)
            if feature_idx is not None:
                stats = get_feature_stats(models, feature_idx)
                stats_md = f"**Feature {feature_idx}**\n\n"
                for k, v in stats.items():
                    if k.startswith("corr_"):
                        prop_name = k.replace("corr_", "")
                        stats_md += f"- {prop_name}: r = {v:+.3f}\n"
                return feature_idx, stats_md
            return gr.update(), "*No correlation data*"

        property_dropdown.change(
            fn=on_property_change,
            inputs=[property_dropdown],
            outputs=[feature_input, stats_display],
        )

        def on_feature_change(feature_idx):
            """Update stats when feature index changes."""
            feature_idx = int(feature_idx)
            stats = get_feature_stats(models, feature_idx)
            stats_md = f"**Feature {feature_idx}**\n\n"
            for k, v in stats.items():
                if k.startswith("corr_"):
                    prop_name = k.replace("corr_", "")
                    stats_md += f"- {prop_name}: r = {v:+.3f}\n"
            return stats_md

        feature_input.change(
            fn=on_feature_change,
            inputs=[feature_input],
            outputs=[stats_display],
        )

        def on_generate(feature_idx, steering_value, n_samples, seed):
            """Generate batch of steering triplets."""
            feature_idx = int(feature_idx)
            n_samples = int(min(n_samples, 5))  # Cap at 5
            seed = int(seed)

            # Select random samples
            np.random.seed(seed)
            selected = np.random.choice(
                sample_indices,
                size=min(n_samples, len(sample_indices)),
                replace=False,
            )

            # Generate triplets
            results = []
            for i, sample_idx in enumerate(selected):
                print(f"[GEN] Sample {i+1}/{n_samples} (idx={sample_idx})")
                audio_orig, audio_plus, audio_minus = generate_steering_triplet(
                    models, sample_idx, feature_idx, steering_value
                )
                results.append((audio_orig, audio_plus, audio_minus))

            # Build outputs for all 5 audio rows
            outputs = []
            for i in range(5):
                if i < len(results):
                    audio_orig, audio_plus, audio_minus = results[i]
                    # Format as (sample_rate, audio_array) tuple for Gradio
                    outputs.append((CONFIG.sample_rate, audio_orig) if audio_orig is not None else None)
                    outputs.append((CONFIG.sample_rate, audio_plus) if audio_plus is not None else None)
                    outputs.append((CONFIG.sample_rate, audio_minus) if audio_minus is not None else None)
                else:
                    outputs.extend([None, None, None])

            status = f"*Generated {len(results)} triplets for feature {feature_idx} (seed={seed})*"
            outputs.append(status)

            return outputs

        # Build outputs list: 5 rows × 3 audios + status
        generate_outputs = []
        for orig, plus, minus in sample_audios:
            generate_outputs.extend([orig, plus, minus])
        generate_outputs.append(status_text)

        generate_btn.click(
            fn=on_generate,
            inputs=[feature_input, steering_slider, n_samples_slider, random_seed],
            outputs=generate_outputs,
        )

    return demo


# =============================================================================
# Main Entry
# =============================================================================


def main():
    """Launch the V2 demo."""
    print("\n" + "=" * 60)
    print("DRUMS SAE V2 — STEERING QUALITY TESTER")
    print("=" * 60)

    models = load_v2_models()
    demo = build_interface(models)

    print("\n[LAUNCH] Starting Gradio server...")
    demo.launch(share=False, css=BRUTALIST_CSS)


if __name__ == "__main__":
    main()
