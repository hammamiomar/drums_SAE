"""
Drums SAE — Model Comparison Demo

Bulk testing demo to compare v1 and v2 SAE models side-by-side.
Uses the Gytis residual trick for quality preservation.

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
from demo.v1.config import CONFIG as CONFIG_V1
from demo.v2.config import CONFIG as CONFIG_V2, PROJECT_ROOT
from drums_SAE.sae.model import AudioSae
from drums_SAE.steering.steer import steer_with_residual


# =============================================================================
# Model Loading
# =============================================================================


def load_v1_models() -> dict:
    """Load SAE, VAE, latents, and features for v1 demo.

    V1 uses precomputed correlations from feature_summary.csv.
    """
    print(f"\n[LOAD V1] Device: {DEVICE}")

    checkpoint_path = PROJECT_ROOT / CONFIG_V1.checkpoint_path
    latent_data_path = PROJECT_ROOT / CONFIG_V1.latent_data_path
    features_path = PROJECT_ROOT / CONFIG_V1.feature_summary_path

    # Load SAE
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]

    sae = AudioSae(
        d_input=cfg["d_input"],
        expansion_factor=cfg["expansion_factor"],
        topk=cfg["topk"],
        topk_aux=cfg.get("topk_aux", 128),  # v1 might not have this
        dead_threshold=cfg.get("dead_threshold", 10000),
    ).to(DEVICE)
    sae.load_state_dict(checkpoint["model_state_dict"])
    sae.training = False
    print(f"[LOAD V1] SAE: {cfg['d_input']} -> {cfg['d_input'] * cfg['expansion_factor']} features")

    # Load latents and stats
    latent_data = np.load(latent_data_path)
    all_latents = torch.from_numpy(latent_data["latents"]).float().to(DEVICE)
    latent_mean, latent_std = load_latent_stats(str(latent_data_path), DEVICE)

    # Normalize latents
    all_latents_norm = (all_latents - latent_mean) / (latent_std + 1e-8)

    # Load feature summary with precomputed correlations
    features_df = pd.read_csv(features_path)

    # V1 has all samples (no silence filtering was done during encoding)
    n_timesteps = CONFIG_V1.n_timesteps
    n_total_samples = len(all_latents) // n_timesteps
    sample_indices = list(range(n_total_samples))
    print(f"[LOAD V1] Total samples: {n_total_samples}")

    # Load VAE (shared with v2)
    vae = load_vae(DEVICE)

    # Extract correlations from precomputed columns in feature_summary.csv
    # V1 CSV has columns like: corr_brightness, corr_boominess, etc.
    correlations = {}
    corr_columns = [c for c in features_df.columns if c.startswith("corr_")]
    for col in corr_columns:
        prop_name = col.replace("corr_", "")
        correlations[prop_name] = features_df[col].values
        max_idx = np.argmax(np.abs(correlations[prop_name]))
        max_val = np.abs(correlations[prop_name]).max()
        print(f"  {prop_name}: max |r| = {max_val:.3f} (feature {max_idx})")

    return {
        "sae": sae,
        "vae": vae,
        "all_latents_norm": all_latents_norm,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "features_df": features_df,
        "sample_indices": sample_indices,
        "n_timesteps": n_timesteps,
        "correlations": correlations,
        "sample_rate": CONFIG_V1.sample_rate,
        "version": "v1",
    }


def load_v2_models() -> dict:
    """Load SAE, VAE, latents, and features for v2 demo."""
    print(f"\n[LOAD V2] Device: {DEVICE}")

    checkpoint_path = PROJECT_ROOT / CONFIG_V2.checkpoint_path
    latent_data_path = PROJECT_ROOT / CONFIG_V2.latent_data_path
    features_path = PROJECT_ROOT / CONFIG_V2.features_path

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
    n_timesteps = CONFIG_V2.n_timesteps
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
        "sample_rate": CONFIG_V2.sample_rate,
        "version": "v2",
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


def build_steering_ui(models: dict, tab_label: str) -> None:
    """Build the steering UI for one model version.

    This creates all the UI components within the current Gradio context.
    It's called once per tab (v1 and v2).

    Args:
        models: The loaded model dict (from load_v1_models or load_v2_models)
        tab_label: Label for display (e.g., "V1" or "V2")
    """
    sample_indices = models["sample_indices"]
    available_props = list(models["correlations"].keys())
    n_features = models["sae"].d_hidden
    sample_rate = models["sample_rate"]
    version = models.get("version", "unknown")

    # Info banner
    gr.Markdown(
        f"**{n_features} features** | "
        f"**{len(sample_indices)} samples** | "
        f"**{models['n_timesteps']} timesteps**"
    )

    # =================================================================
    # Control Panel
    # =================================================================
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### FEATURE SELECTION")

            # Option 1: By property
            default_prop = available_props[0] if available_props else "(manual)"
            # Try to pick a sensible default
            for preferred in ["sub_bass", "brightness", "spectral_centroid"]:
                if preferred in available_props:
                    default_prop = preferred
                    break

            property_dropdown = gr.Dropdown(
                choices=["(manual)"] + available_props,
                value=default_prop,
                label="SELECT BY PROPERTY",
                info="Pick best feature for this property",
            )

            # Option 2: Manual index
            feature_input = gr.Number(
                value=0,
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
                maximum=5,
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

    # Create 5 fixed sample rows
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
    # Event Handlers (closure over models)
    # =================================================================

    def on_property_change(prop):
        """Update feature index when property is selected."""
        if prop == "(manual)":
            return gr.update(), gr.update()

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
        if feature_idx < 0 or feature_idx >= n_features:
            return "*Invalid feature index*"
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
        n_samples = int(min(n_samples, 5))
        seed = int(seed)

        # Validate feature index
        if feature_idx < 0 or feature_idx >= n_features:
            return [None] * 15 + [f"*Invalid feature index: {feature_idx}*"]

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
            print(f"[GEN {version.upper()}] Sample {i+1}/{n_samples} (idx={sample_idx})")
            audio_orig, audio_plus, audio_minus = generate_steering_triplet(
                models, sample_idx, feature_idx, steering_value
            )
            results.append((audio_orig, audio_plus, audio_minus))

        # Build outputs for all 5 audio rows
        outputs = []
        for i in range(5):
            if i < len(results):
                audio_orig, audio_plus, audio_minus = results[i]
                outputs.append((sample_rate, audio_orig) if audio_orig is not None else None)
                outputs.append((sample_rate, audio_plus) if audio_plus is not None else None)
                outputs.append((sample_rate, audio_minus) if audio_minus is not None else None)
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


def build_interface(models_v1: dict, models_v2: dict) -> gr.Blocks:
    """Build the tabbed comparison interface."""

    with gr.Blocks(title="DRUMS SAE MODEL COMPARISON") as demo:

        # Header
        gr.Markdown("# DRUMS SAE — MODEL COMPARISON")
        gr.Markdown(
            "*Compare v1 and v2 SAE models side-by-side. "
            "Uses the Gytis residual trick for quality preservation.*"
        )

        with gr.Tabs():
            with gr.Tab(f"V2 ({models_v2['sae'].d_hidden} features)"):
                build_steering_ui(models_v2, "V2")

            with gr.Tab(f"V1 ({models_v1['sae'].d_hidden} features)"):
                build_steering_ui(models_v1, "V1")

    return demo


# =============================================================================
# Main Entry
# =============================================================================


def main():
    """Launch the model comparison demo."""
    print("\n" + "=" * 60)
    print("DRUMS SAE — MODEL COMPARISON DEMO")
    print("=" * 60)

    # Load both models (VAE is shared automatically via load_vae cache)
    print("\n--- Loading V2 Model ---")
    models_v2 = load_v2_models()

    print("\n--- Loading V1 Model ---")
    models_v1 = load_v1_models()

    # Build tabbed interface
    demo = build_interface(models_v1, models_v2)

    print("\n[LAUNCH] Starting Gradio server...")
    demo.launch(share=False, css=BRUTALIST_CSS)


if __name__ == "__main__":
    main()
