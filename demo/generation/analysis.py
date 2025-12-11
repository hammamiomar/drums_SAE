"""Audio analysis utilities for measuring steering effects."""

import numpy as np
import librosa
import plotly.graph_objects as go
from typing import Optional

from .presets import PROPERTY_NAMES, PROPERTY_DISPLAY_NAMES, PROPERTY_COLORS


def measure_audio_properties(
    audio: np.ndarray,
    sr: int,
) -> dict[str, float]:
    """Extract acoustic properties from audio for comparison.

    Computes the same features our SAE probes were trained on, allowing
    us to verify that steering changes properties as expected.

    Args:
        audio: Audio array, shape (samples,) or (channels, samples)
        sr: Sample rate

    Returns:
        Dict of measured properties:
            - spectral_centroid: Brightness in Hz
            - rms: Root mean square energy
            - bass: Low-frequency energy (< 200 Hz)
            - crest_factor: Peak / RMS ratio (transient-ness)
    """
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    # Ensure float32
    audio = audio.astype(np.float32)

    # Normalize to prevent numerical issues
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio_norm = audio / max_val
    else:
        audio_norm = audio

    # Spectral centroid (brightness)
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=audio_norm, sr=sr)
    )

    # RMS energy
    rms = np.mean(librosa.feature.rms(y=audio_norm))

    # Bass energy (< 200 Hz)
    stft = np.abs(librosa.stft(audio_norm))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = freqs < 200
    bass = np.mean(stft[bass_mask, :]) if bass_mask.any() else 0.0

    # Crest factor (peak / RMS)
    peak = np.max(np.abs(audio_norm))
    crest_factor = float(peak / (rms + 1e-8))

    return {
        "spectral_centroid": float(spectral_centroid),
        "rms": float(rms),
        "bass": float(bass),
        "crest_factor": float(crest_factor),
    }


def format_comparison(
    baseline: dict[str, float],
    steered: dict[str, float],
    steering_applied: dict[str, float],
) -> str:
    """Format a markdown comparison table showing what changed.

    Args:
        baseline: Measured properties of baseline audio
        steered: Measured properties of steered audio
        steering_applied: The steering dict that was applied

    Returns:
        Markdown table string
    """
    lines = [
        "### What Changed",
        "",
        "| Property | Baseline | Steered | Change | Requested |",
        "|----------|----------|---------|--------|-----------|",
    ]

    for prop in PROPERTY_NAMES:
        b = baseline.get(prop, 0)
        s = steered.get(prop, 0)

        # Calculate percentage change
        if abs(b) > 1e-8:
            pct = ((s - b) / abs(b)) * 100
            direction = "+" if pct > 0 else ""
            change_str = f"{direction}{pct:.0f}%"
        else:
            change_str = "N/A"

        # What was requested
        requested = steering_applied.get(prop, 0)
        if abs(requested) > 0.01:
            req_str = f"{requested:+.1f}"
        else:
            req_str = "—"

        # Format values appropriately
        if prop == "spectral_centroid":
            b_str = f"{b:.0f} Hz"
            s_str = f"{s:.0f} Hz"
        elif prop in ("rms", "bass"):
            b_str = f"{b:.4f}"
            s_str = f"{s:.4f}"
        else:
            b_str = f"{b:.2f}"
            s_str = f"{s:.2f}"

        name = PROPERTY_DISPLAY_NAMES.get(prop, prop)
        lines.append(f"| {name} | {b_str} | {s_str} | {change_str} | {req_str} |")

    return "\n".join(lines)


def create_evolution_plot(
    baseline_steps: list[dict],
    steered_steps: list[dict],
    steering_applied: dict[str, float],
    schedule: str = "middle",
) -> Optional[go.Figure]:
    """Create a Plotly figure showing property evolution during generation.

    This visualizes how acoustic properties crystallize during diffusion,
    and where the steered values diverge from baseline (in the steering window).

    Args:
        baseline_steps: List of {step, bass, spectral_centroid, rms, crest_factor}
        steered_steps: Same format for steered run
        steering_applied: Which properties were steered (for highlighting)
        schedule: Steering schedule used ("all", "early", "middle", "late")

    Returns:
        Plotly Figure object, or None if no data
    """
    if not baseline_steps or not steered_steps:
        return None

    fig = go.Figure()

    for prop in PROPERTY_NAMES:
        color = PROPERTY_COLORS.get(prop, "#888888")
        name = PROPERTY_DISPLAY_NAMES.get(prop, prop)
        was_steered = abs(steering_applied.get(prop, 0)) > 0.01

        # Baseline (dashed, faded)
        fig.add_trace(go.Scatter(
            x=[s["step"] for s in baseline_steps],
            y=[s[prop] for s in baseline_steps],
            mode="lines",
            name=f"{name} (baseline)",
            line=dict(color=color, width=1, dash="dash"),
            opacity=0.5,
            legendgroup=prop,
        ))

        # Steered (solid, thicker if this property was steered)
        fig.add_trace(go.Scatter(
            x=[s["step"] for s in steered_steps],
            y=[s[prop] for s in steered_steps],
            mode="lines",
            name=f"{name} (steered)",
            line=dict(color=color, width=3 if was_steered else 1.5),
            legendgroup=prop,
        ))

    # Add shaded region for steering window
    max_step = steered_steps[-1]["step"]

    if schedule == "early":
        x0, x1 = 0, max_step * 0.3
    elif schedule == "late":
        x0, x1 = max_step * 0.7, max_step
    elif schedule == "middle":
        x0, x1 = max_step * 0.3, max_step * 0.7
    else:  # "all"
        x0, x1 = 0, max_step

    fig.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor="rgba(100, 100, 100, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="Steering Active",
        annotation_position="top left",
        annotation_font_size=10,
    )

    fig.update_layout(
        title="Property Evolution During Diffusion",
        xaxis_title="Diffusion Step",
        yaxis_title="Property Score (normalized)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
        ),
        template="plotly_white",
        margin=dict(r=200),  # Room for legend
        hovermode="x unified",
        height=400,
    )

    return fig
