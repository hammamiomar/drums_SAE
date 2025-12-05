"""
Drums SAE Steering Demo - Brutalist Techno DAW
==============================================

Main Gradio 6 application with three-panel layout for maximum experimentability.
"""

import sys
import time
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio

# Add src to path for imports
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from .config import (
    CONFIG,
    MODE_SLIDERS,
    PROJECT_ROOT,
    SLIDER_LABELS,
    TEMPORAL_MODES,
)
from .state import (
    SessionState,
    SteeringHistoryItem,
    build_property_params,
    format_alpha_display,
    get_active_properties,
    has_active_steering,
)
from .theme import BRUTALIST_CSS
from .viz import (
    create_alpha_envelope,
    create_empty_plot,
    create_spectrogram_comparison,
    create_waveform_overlay,
)

# =============================================================================
# Device Detection
# =============================================================================

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# =============================================================================
# Model Loading
# =============================================================================

def load_models() -> dict:
    """Load SAE, control vectors, temporal vectors, dataset, and VAE."""
    from drums_SAE.sae.model import AudioSae
    from drums_SAE.training.data import LatentDataset

    print(f"\n[LOAD] Device: {DEVICE}")

    checkpoint_path = PROJECT_ROOT / CONFIG.checkpoint_path
    feature_summary_path = PROJECT_ROOT / CONFIG.feature_summary_path
    latent_data_path = PROJECT_ROOT / CONFIG.latent_data_path

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
    sae.eval()
    print(f"[LOAD] SAE: {cfg['d_input']} -> {cfg['d_input'] * cfg['expansion_factor']} dims")

    # Load feature summary
    feature_df = pd.read_csv(feature_summary_path)
    print(f"[LOAD] Features: {len(feature_df)}")

    # Build control vectors
    W_dec = sae.decoder.weight.data.cpu()
    control_vectors = {}
    attack_ratios = feature_df["attack_ratio"].values

    for label in CONFIG.steering_labels:
        corr_col = f"corr_{label}"
        if corr_col not in feature_df.columns:
            continue
        corr = feature_df[corr_col].values
        if np.abs(corr).max() < 0.05:
            continue

        top_k = 20
        top_idx = np.argsort(np.abs(corr))[-top_k:]
        weights = torch.tensor(corr[top_idx], dtype=torch.float32)
        vec = W_dec[:, top_idx] @ weights
        vec = vec / vec.norm()

        control_vectors[label] = {
            "direction": vec.to(DEVICE),
            "correlations": corr,
            "top_features": top_idx,
            "max_corr": float(np.abs(corr).max()),
        }

    print(f"[LOAD] Control vectors: {list(control_vectors.keys())}")

    # Build temporal control vectors
    temporal_vectors = {}
    attack_threshold = np.median(attack_ratios[attack_ratios > 0])

    for label in control_vectors.keys():
        corr = control_vectors[label]["correlations"]
        attack_mask = (attack_ratios > attack_threshold) & (np.abs(corr) > 0.05)
        sustain_mask = (attack_ratios <= attack_threshold) & (attack_ratios > 0) & (np.abs(corr) > 0.05)

        def build_vec(mask):
            if mask.sum() >= 5:
                idx = np.where(mask)[0]
                top = idx[np.argsort(np.abs(corr[idx]))[-10:]]
                w = torch.tensor(corr[top], dtype=torch.float32)
                v = W_dec[:, top] @ w
                return (v / (v.norm() + 1e-8)).to(DEVICE)
            return control_vectors[label]["direction"]

        temporal_vectors[label] = {
            "attack": build_vec(attack_mask),
            "sustain": build_vec(sustain_mask),
        }

    # Load dataset
    dataset = LatentDataset(str(latent_data_path), normalize=True)
    n_samples = len(dataset) // CONFIG.n_timesteps
    print(f"[LOAD] Dataset: {n_samples:,} samples")

    # Load normalization stats
    latent_data = np.load(latent_data_path)
    latent_mean = torch.tensor(latent_data["mean"], dtype=torch.float32).to(DEVICE)
    latent_std = torch.tensor(latent_data["std"], dtype=torch.float32).to(DEVICE)

    # Load VAE
    vae = None
    try:
        from stable_audio_tools import get_pretrained_model
        print("[LOAD] Loading VAE...")
        model, _ = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        vae = model.pretransform.model.to(DEVICE)
        for p in vae.parameters():
            p.requires_grad = False
        print("[LOAD] VAE ready")
    except Exception as e:
        print(f"[WARN] VAE unavailable: {e}")

    return {
        "sae": sae,
        "control_vectors": control_vectors,
        "temporal_vectors": temporal_vectors,
        "dataset": dataset,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "vae": vae,
        "n_samples": n_samples,
        "attack_ratios": attack_ratios,
        "feature_df": feature_df,
    }


# =============================================================================
# Steering Functions
# =============================================================================

def get_audio_latents(models: dict, audio_idx: int) -> torch.Tensor:
    """Get all 32 latent vectors for one audio sample."""
    dataset = models["dataset"]
    start = audio_idx * CONFIG.n_timesteps
    end = start + CONFIG.n_timesteps
    return torch.stack([dataset[i] for i in range(start, end)]).to(DEVICE)


def steer_uniform(z, direction, alpha):
    return z + alpha * direction.unsqueeze(0)


def steer_segment(z, direction, alpha_attack, alpha_body, alpha_tail):
    z_out = z.clone()
    z_out[:CONFIG.attack_end] += alpha_attack * direction
    z_out[CONFIG.attack_end:CONFIG.body_end] += alpha_body * direction
    z_out[CONFIG.body_end:] += alpha_tail * direction
    return z_out


def steer_envelope(z, direction, alpha_start, alpha_end):
    alphas = torch.linspace(alpha_start, alpha_end, CONFIG.n_timesteps, device=DEVICE)
    return z + alphas.unsqueeze(1) * direction.unsqueeze(0)


def steer_temporal(z, attack_dir, sustain_dir, alpha_attack, alpha_sustain):
    z_out = z.clone()
    z_out[:CONFIG.attack_end] += alpha_attack * attack_dir
    z_out[CONFIG.attack_end:] += alpha_sustain * sustain_dir
    return z_out


def apply_steering(models, z, label, mode, params):
    """Apply steering for a single property."""
    cv = models["control_vectors"]
    tv = models["temporal_vectors"]

    if label not in cv:
        return z

    direction = cv[label]["direction"]

    if mode == "uniform":
        return steer_uniform(z, direction, params.get("alpha", 0.0))
    elif mode == "segment":
        return steer_segment(
            z, direction,
            params.get("alpha_attack", 0.0),
            params.get("alpha_body", 0.0),
            params.get("alpha_tail", 0.0),
        )
    elif mode == "envelope":
        return steer_envelope(
            z, direction,
            params.get("alpha_start", 0.0),
            params.get("alpha_end", 0.0),
        )
    elif mode == "temporal_features":
        return steer_temporal(
            z,
            tv[label]["attack"],
            tv[label]["sustain"],
            params.get("alpha_attack", 0.0),
            params.get("alpha_sustain", 0.0),
        )
    return z


def multi_property_steer(models, z, mode, property_params):
    """Apply steering for multiple properties sequentially."""
    z_steered = z.clone()
    for label, params in property_params.items():
        if all(abs(v) < 1e-6 for v in params.values()):
            continue
        z_steered = apply_steering(models, z_steered, label, mode, params)
    return z_steered


# =============================================================================
# Audio Decoding
# =============================================================================

def decode_to_audio(z_norm: torch.Tensor, models: dict) -> np.ndarray | None:
    """Decode normalized latents to audio. Returns 1D mono array."""
    vae = models["vae"]
    if vae is None:
        return None

    from einops import rearrange

    if z_norm.dim() == 2:
        z_norm = z_norm.unsqueeze(0)

    z = z_norm * models["latent_std"] + models["latent_mean"]
    z = rearrange(z, "b t c -> b c t")

    with torch.no_grad():
        audio = vae.decode(z)

    # VAE outputs (batch, channels, samples) - extract mono
    audio = audio.cpu().numpy()
    audio = np.squeeze(audio)  # Remove batch dim if present

    # If stereo (2, samples) or (samples, 2), take first channel
    if audio.ndim == 2:
        if audio.shape[0] <= 2:  # (channels, samples)
            audio = audio[0]
        else:  # (samples, channels)
            audio = audio[:, 0]

    # Ensure float32 in [-1, 1] range for Gradio
    audio = audio.astype(np.float32)
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()

    return audio


# =============================================================================
# Gradio Interface Builder
# =============================================================================

def build_interface(models: dict) -> gr.Blocks:
    """Build the Gradio 6 interface - Tangled Cables DAW layout."""

    n_samples = models["n_samples"]
    available_labels = list(models["control_vectors"].keys())

    # Create sample choices
    sample_choices = [(f"{i:04d}", i) for i in range(min(CONFIG.max_samples_in_dropdown, n_samples))]

    with gr.Blocks(title="DRUMS SAE STEERING") as demo:

        # Session state
        state = gr.State(SessionState())

        # =====================================================================
        # HEADER with logo
        # =====================================================================
        with gr.Row():
            gr.Image(
                value=str(PROJECT_ROOT / "demo" / "logos" / "logo.png"),
                show_label=False,
                height=60,
                width=60,
                scale=0,
                container=False,
            )
            gr.Markdown("# DRUMS SAE STEERING")

        # =====================================================================
        # TOP ROW: Sample + A/B Transport
        # =====================================================================
        with gr.Row():
            sample_dropdown = gr.Dropdown(
                choices=sample_choices,
                value=0,
                label="SAMPLE",
                scale=2,
            )
            random_btn = gr.Button("RND", scale=1)
            ab_a_btn = gr.Button("A ORIGINAL", variant="primary", scale=2)  # Active by default
            ab_b_btn = gr.Button("B STEERED", variant="secondary", scale=2)

        # =====================================================================
        # MAIN "TV SCREEN" - Audio + Waveform
        # =====================================================================
        with gr.Group():
            main_audio = gr.Audio(
                label=None,
                type="numpy",
                interactive=False,
            )
            waveform_plot = gr.Plot(label=None)

        # =====================================================================
        # CONTROL PANEL - Property, Mode, Alpha side by side
        # =====================================================================
        with gr.Row():
            # Property selector
            with gr.Column(scale=1):
                gr.Markdown("### PROPERTY")
                property_radio = gr.Radio(
                    choices=available_labels,
                    value=available_labels[0],
                    show_label=False,
                    interactive=True,
                )

            # Mode selector
            with gr.Column(scale=1):
                gr.Markdown("### MODE")
                mode_radio = gr.Radio(
                    choices=["uniform", "segment", "envelope", "temporal_features"],
                    value="uniform",
                    show_label=False,
                    interactive=True,
                )
                mode_desc = gr.Markdown(f"*{TEMPORAL_MODES['uniform']}*")

            # Alpha sliders
            with gr.Column(scale=2):
                gr.Markdown("### ALPHA")

                # Uniform mode slider (visible by default)
                uniform_slider = gr.Slider(
                    minimum=CONFIG.alpha_min,
                    maximum=CONFIG.alpha_max,
                    value=CONFIG.alpha_default,
                    step=CONFIG.alpha_step,
                    label="STRENGTH",
                    info="+ adds property, - removes it",
                    visible=True,
                )

                # Segment mode sliders (hidden by default)
                segment_attack = gr.Slider(
                    minimum=CONFIG.alpha_min, maximum=CONFIG.alpha_max,
                    value=0, step=CONFIG.alpha_step, label="ATTACK [0-375ms]",
                    info="Transient/hit - affects punch",
                    visible=False,
                )
                segment_body = gr.Slider(
                    minimum=CONFIG.alpha_min, maximum=CONFIG.alpha_max,
                    value=0, step=CONFIG.alpha_step, label="BODY [375-1125ms]",
                    info="Main resonance - affects tone",
                    visible=False,
                )
                segment_tail = gr.Slider(
                    minimum=CONFIG.alpha_min, maximum=CONFIG.alpha_max,
                    value=0, step=CONFIG.alpha_step, label="TAIL [1125-1500ms]",
                    info="Decay/release - affects sustain",
                    visible=False,
                )

                # Envelope mode sliders (hidden by default)
                envelope_start = gr.Slider(
                    minimum=CONFIG.alpha_min, maximum=CONFIG.alpha_max,
                    value=0, step=CONFIG.alpha_step, label="START",
                    info="Alpha at beginning",
                    visible=False,
                )
                envelope_end = gr.Slider(
                    minimum=CONFIG.alpha_min, maximum=CONFIG.alpha_max,
                    value=0, step=CONFIG.alpha_step, label="END",
                    info="Alpha at end (linear interpolation)",
                    visible=False,
                )

                # Temporal features mode sliders (hidden by default)
                temporal_attack = gr.Slider(
                    minimum=CONFIG.alpha_min, maximum=CONFIG.alpha_max,
                    value=0, step=CONFIG.alpha_step, label="ATTACK FEATURES",
                    info="SAE features that fire during transient",
                    visible=False,
                )
                temporal_sustain = gr.Slider(
                    minimum=CONFIG.alpha_min, maximum=CONFIG.alpha_max,
                    value=0, step=CONFIG.alpha_step, label="SUSTAIN FEATURES",
                    info="SAE features that fire during body/tail",
                    visible=False,
                )

        # =====================================================================
        # BOTTOM ROW: Actions + Status
        # =====================================================================
        with gr.Row():
            apply_btn = gr.Button("STEER", variant="primary", scale=3)
            reset_btn = gr.Button("RESET", scale=1)
            save_btn = gr.Button("SAVE", scale=1)
        status_text = gr.Markdown("*Select sample to begin*")

        # =====================================================================
        # EXTRA TABS (collapsed by default)
        # =====================================================================
        with gr.Accordion("MORE VISUALIZATIONS", open=False):
            with gr.Tabs():
                with gr.Tab("SPECTROGRAM"):
                    spectrogram_plot = gr.Plot(label=None)
                with gr.Tab("ALPHA ENVELOPE"):
                    envelope_plot = gr.Plot(label=None)
                with gr.Tab("FEATURE INFO"):
                    with gr.Row():
                        data_readout = gr.Markdown("**PROP:** ---\n**MODE:** ---\n**ALPHA:** ---")
                        feature_info = gr.Markdown("*Select property*")
                        history_display = gr.Markdown("")

        # =====================================================================
        # EVENT HANDLERS
        # =====================================================================

        def on_mode_change(mode):
            """Update visibility and reset sliders based on mode."""
            is_uniform = mode == "uniform"
            is_segment = mode == "segment"
            is_envelope = mode == "envelope"
            is_temporal = mode == "temporal_features"

            return (
                # Description
                f"*{TEMPORAL_MODES[mode]}*",
                # Uniform slider - visible + reset
                gr.update(visible=is_uniform, value=0),
                # Segment sliders - visible + reset
                gr.update(visible=is_segment, value=0),
                gr.update(visible=is_segment, value=0),
                gr.update(visible=is_segment, value=0),
                # Envelope sliders - visible + reset
                gr.update(visible=is_envelope, value=0),
                gr.update(visible=is_envelope, value=0),
                # Temporal sliders - visible + reset
                gr.update(visible=is_temporal, value=0),
                gr.update(visible=is_temporal, value=0),
            )

        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio],
            outputs=[
                mode_desc,
                uniform_slider,
                segment_attack, segment_body, segment_tail,
                envelope_start, envelope_end,
                temporal_attack, temporal_sustain,
            ],
        )

        def on_random_sample():
            """Select random sample."""
            idx = np.random.randint(0, min(CONFIG.max_samples_in_dropdown, n_samples))
            return gr.update(value=idx)

        random_btn.click(fn=on_random_sample, outputs=[sample_dropdown])

        def on_property_change(prop, session_state):
            """Update feature info when property changes."""
            if prop not in models["control_vectors"]:
                return "---", session_state

            cv = models["control_vectors"][prop]
            top_idx = cv["top_features"]
            corrs = cv["correlations"][top_idx]
            max_corr = cv["max_corr"]

            # Format feature info
            lines = [f"**{prop.upper()}** (max |r| = {max_corr:.3f})\n"]
            for idx, c in zip(top_idx[-5:], corrs[-5:]):
                lines.append(f"`F{idx:04d}`: {c:+.3f}")

            session_state.current_property = prop
            return "\n".join(lines), session_state

        property_radio.change(
            fn=on_property_change,
            inputs=[property_radio, state],
            outputs=[feature_info, state],
        )

        def on_sample_change(sample_idx, session_state):
            """Load sample and decode original audio."""
            z_original = get_audio_latents(models, sample_idx)
            audio_original = decode_to_audio(z_original, models)

            session_state.current_sample_idx = sample_idx
            session_state.z_original = z_original
            session_state.audio_original = audio_original
            session_state.z_steered = None
            session_state.audio_steered = None
            session_state.current_playback = "A"

            # Create initial visualizations
            waveform_fig = create_waveform_overlay(audio_original, None)
            spec_fig = create_spectrogram_comparison(audio_original, None)
            env_fig = create_empty_plot("APPLY STEERING TO SEE ENVELOPE")

            audio_out = (CONFIG.sample_rate, audio_original) if audio_original is not None else None

            return (
                audio_out,
                waveform_fig,
                spec_fig,
                env_fig,
                f"*Loaded sample {sample_idx}*",
                session_state,
            )

        sample_dropdown.change(
            fn=on_sample_change,
            inputs=[sample_dropdown, state],
            outputs=[main_audio, waveform_plot, spectrogram_plot, envelope_plot, status_text, state],
        )

        def on_apply_steering(
            sample_idx, mode, prop,
            uniform_alpha,
            seg_attack, seg_body, seg_tail,
            env_start, env_end,
            temp_attack, temp_sustain,
            session_state,
        ):
            """Apply steering and update all outputs."""
            # Build params for current property based on mode
            if mode == "uniform":
                params = {"alpha": uniform_alpha}
            elif mode == "segment":
                params = {"alpha_attack": seg_attack, "alpha_body": seg_body, "alpha_tail": seg_tail}
            elif mode == "envelope":
                params = {"alpha_start": env_start, "alpha_end": env_end}
            elif mode == "temporal_features":
                params = {"alpha_attack": temp_attack, "alpha_sustain": temp_sustain}
            else:
                params = {}

            # Store in state
            for k, v in params.items():
                session_state.set_param(prop, k, v)
            session_state.current_mode = mode
            session_state.current_property = prop

            # Get or compute original latents
            if session_state.z_original is None:
                z_original = get_audio_latents(models, sample_idx)
                session_state.z_original = z_original
                session_state.audio_original = decode_to_audio(z_original, models)
            else:
                z_original = session_state.z_original

            # Build property_params dict for steering
            property_params = {prop: params}

            # Apply steering
            z_steered = multi_property_steer(models, z_original, mode, property_params)
            audio_steered = decode_to_audio(z_steered, models)

            session_state.z_steered = z_steered
            session_state.audio_steered = audio_steered

            # Add to history
            history_item = SteeringHistoryItem(
                sample_idx=sample_idx,
                mode=mode,
                property_name=prop,
                params=params.copy(),
                timestamp=time.time(),
            )
            session_state.add_to_history(history_item)

            # Create visualizations
            waveform_fig = create_waveform_overlay(session_state.audio_original, audio_steered)
            spec_fig = create_spectrogram_comparison(session_state.audio_original, audio_steered)
            env_fig = create_alpha_envelope(mode, params, prop)

            # Format data readout
            alpha_str = format_alpha_display(params, mode)
            max_corr = models["control_vectors"].get(prop, {}).get("max_corr", 0)
            data_md = (
                f"**PROPERTY:** {prop.upper()}\n\n"
                f"**MODE:** {mode.upper()}\n\n"
                f"**ALPHA:** `{alpha_str}`\n\n"
                f"**MAX |r|:** {max_corr:.3f}"
            )

            # Format history
            history_lines = []
            for i, h in enumerate(session_state.steering_history[:5]):
                history_lines.append(f"{i+1}. {h.property_name}: {h.summary()}")
            history_md = "\n".join(history_lines) if history_lines else "*No history*"

            # Audio output (show steered)
            audio_out = (CONFIG.sample_rate, audio_steered) if audio_steered is not None else None
            session_state.current_playback = "B"

            active_props = get_active_properties(property_params)
            status = f"*Steered: {', '.join(active_props)} | Mode: {mode}*"

            return (
                audio_out,
                waveform_fig,
                spec_fig,
                env_fig,
                status,
                data_md,
                history_md,
                gr.update(variant="secondary"),  # A inactive
                gr.update(variant="primary"),    # B active (now playing steered)
                session_state,
            )

        apply_btn.click(
            fn=on_apply_steering,
            inputs=[
                sample_dropdown, mode_radio, property_radio,
                uniform_slider,
                segment_attack, segment_body, segment_tail,
                envelope_start, envelope_end,
                temporal_attack, temporal_sustain,
                state,
            ],
            outputs=[
                main_audio,
                waveform_plot,
                spectrogram_plot,
                envelope_plot,
                status_text,
                data_readout,
                history_display,
                ab_a_btn,
                ab_b_btn,
                state,
            ],
        )

        def on_reset(session_state):
            """Reset all sliders to default."""
            session_state.clear_steering()
            return (
                CONFIG.alpha_default,  # uniform
                CONFIG.alpha_default, CONFIG.alpha_default, CONFIG.alpha_default,  # segment
                CONFIG.alpha_default, CONFIG.alpha_default,  # envelope
                CONFIG.alpha_default, CONFIG.alpha_default,  # temporal
                "*Sliders reset*",
                session_state,
            )

        reset_btn.click(
            fn=on_reset,
            inputs=[state],
            outputs=[
                uniform_slider,
                segment_attack, segment_body, segment_tail,
                envelope_start, envelope_end,
                temporal_attack, temporal_sustain,
                status_text,
                state,
            ],
        )

        def on_ab_toggle_a(session_state):
            """Switch to original audio - highlight A button."""
            session_state.current_playback = "A"
            audio_out = (CONFIG.sample_rate, session_state.audio_original) if session_state.audio_original is not None else None
            return (
                audio_out,
                gr.update(variant="primary"),   # A active
                gr.update(variant="secondary"), # B inactive
                session_state,
            )

        def on_ab_toggle_b(session_state):
            """Switch to steered audio - highlight B button."""
            session_state.current_playback = "B"
            audio_out = (CONFIG.sample_rate, session_state.audio_steered) if session_state.audio_steered is not None else None
            return (
                audio_out,
                gr.update(variant="secondary"), # A inactive
                gr.update(variant="primary"),   # B active
                session_state,
            )

        ab_a_btn.click(fn=on_ab_toggle_a, inputs=[state], outputs=[main_audio, ab_a_btn, ab_b_btn, state])
        ab_b_btn.click(fn=on_ab_toggle_b, inputs=[state], outputs=[main_audio, ab_a_btn, ab_b_btn, state])

        def on_save(
            sample_idx, mode, prop,
            uniform_alpha,
            seg_attack, seg_body, seg_tail,
            env_start, env_end,
            temp_attack, temp_sustain,
            session_state,
        ):
            """Save comprehensive experiment data: audio, metadata, plots."""
            import json
            from datetime import datetime

            # Create timestamped experiment folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"exp_{timestamp}_sample{sample_idx:04d}_{prop}_{mode}"
            exp_dir = PROJECT_ROOT / CONFIG.output_dir / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Build alpha params based on mode
            if mode == "uniform":
                alpha_params = {"alpha": uniform_alpha}
            elif mode == "segment":
                alpha_params = {"alpha_attack": seg_attack, "alpha_body": seg_body, "alpha_tail": seg_tail}
            elif mode == "envelope":
                alpha_params = {"alpha_start": env_start, "alpha_end": env_end}
            elif mode == "temporal_features":
                alpha_params = {"alpha_attack": temp_attack, "alpha_sustain": temp_sustain}
            else:
                alpha_params = {}

            # Get feature info
            cv = models["control_vectors"].get(prop, {})
            top_features = cv.get("top_features", []).tolist() if hasattr(cv.get("top_features", []), "tolist") else []
            correlations = cv.get("correlations", [])
            top_correlations = {int(idx): float(correlations[idx]) for idx in top_features[-10:]} if len(correlations) > 0 else {}

            # Build metadata
            metadata = {
                "timestamp": timestamp,
                "sample_idx": int(sample_idx),
                "property": prop,
                "mode": mode,
                "alphas": alpha_params,
                "max_correlation": float(cv.get("max_corr", 0)),
                "top_features": top_features[-10:],
                "top_correlations": top_correlations,
                "config": {
                    "n_timesteps": CONFIG.n_timesteps,
                    "attack_end": CONFIG.attack_end,
                    "body_end": CONFIG.body_end,
                    "sample_rate": CONFIG.sample_rate,
                },
                "notes": "",  # User can edit this later
            }

            # Save metadata JSON
            with open(exp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Save original audio
            if session_state.audio_original is not None:
                torchaudio.save(
                    str(exp_dir / "original.wav"),
                    torch.tensor(session_state.audio_original).unsqueeze(0),
                    CONFIG.sample_rate,
                )

            # Save steered audio
            if session_state.audio_steered is not None:
                torchaudio.save(
                    str(exp_dir / "steered.wav"),
                    torch.tensor(session_state.audio_steered).unsqueeze(0),
                    CONFIG.sample_rate,
                )

            # Save visualizations
            try:
                # Waveform
                waveform_fig = create_waveform_overlay(
                    session_state.audio_original,
                    session_state.audio_steered,
                )
                waveform_fig.savefig(exp_dir / "waveform.png", dpi=150, bbox_inches="tight")
                plt.close(waveform_fig)

                # Spectrogram
                spec_fig = create_spectrogram_comparison(
                    session_state.audio_original,
                    session_state.audio_steered,
                )
                spec_fig.savefig(exp_dir / "spectrogram.png", dpi=150, bbox_inches="tight")
                plt.close(spec_fig)

                # Alpha envelope
                env_fig = create_alpha_envelope(mode, alpha_params, prop)
                env_fig.savefig(exp_dir / "alpha_envelope.png", dpi=150, bbox_inches="tight")
                plt.close(env_fig)
            except Exception as e:
                print(f"[WARN] Failed to save plots: {e}")

            return f"*Saved experiment to {exp_dir.name}/*"

        save_btn.click(
            fn=on_save,
            inputs=[
                sample_dropdown, mode_radio, property_radio,
                uniform_slider,
                segment_attack, segment_body, segment_tail,
                envelope_start, envelope_end,
                temporal_attack, temporal_sustain,
                state,
            ],
            outputs=[status_text],
        )

    return demo


# =============================================================================
# Main Entry
# =============================================================================

def main():
    """Launch the demo."""
    print("\n" + "=" * 60)
    print("DRUMS SAE STEERING DEMO")
    print("=" * 60)

    models = load_models()
    demo = build_interface(models)

    print("\n[LAUNCH] Starting Gradio server...")
    demo.launch(
        share=False,
        css=BRUTALIST_CSS,
        favicon_path=str(PROJECT_ROOT / "demo" / "logos" / "favicon.ico"),
    )


if __name__ == "__main__":
    main()
