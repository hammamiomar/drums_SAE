"""Visualization functions for the demo."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from .config import CONFIG

# Sage/Army green palette (inspired by tangled cables logo)
COLORS = {
    "bg": "#0c0e0c",
    "surface": "#141814",
    "border": "#1e241e",
    "text": "#c8d0c8",
    "text_dim": "#5a6b5a",
    "text_muted": "#3a4a3a",
    "accent": "#7a9a6a",       # Muted sage green
    "accent_bright": "#9ab88a",
    "accent_dim": "#5a7a4a",
    "original": "#4a5a4a",
    "original_fill": "#1a221a",
    "wire": "#5a6b52",         # Logo wire color
}


def _setup_ax(ax: plt.Axes) -> None:
    """Apply brutalist styling to axis."""
    ax.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["text_dim"], labelsize=8)
    ax.spines["bottom"].set_color(COLORS["border"])
    ax.spines["left"].set_color(COLORS["border"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ensure_1d(audio: np.ndarray | None) -> np.ndarray | None:
    """Ensure audio is 1D (mono). Squeeze or take first channel if needed."""
    if audio is None:
        return None
    audio = np.asarray(audio)
    # Squeeze any singleton dimensions
    audio = np.squeeze(audio)
    # If still multi-dimensional, take first channel
    if audio.ndim > 1:
        audio = audio[0] if audio.shape[0] < audio.shape[-1] else audio[..., 0]
    return audio


def create_waveform_overlay(
    audio_original: np.ndarray | None,
    audio_steered: np.ndarray | None,
    sample_rate: int = 44100,
) -> plt.Figure:
    """
    Create waveform comparison with original ghosted and steered solid.

    This is a single-plot overlay for direct A/B comparison.
    """
    # Ensure 1D audio
    audio_original = _ensure_1d(audio_original)
    audio_steered = _ensure_1d(audio_steered)

    fig, ax = plt.subplots(figsize=(14, 4), facecolor=COLORS["bg"])
    _setup_ax(ax)

    if audio_original is None and audio_steered is None:
        ax.text(
            0.5, 0.5, "NO AUDIO",
            ha="center", va="center",
            color=COLORS["text_muted"],
            fontsize=12, fontfamily="monospace",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
    else:
        # Use original length for time axis
        audio_ref = audio_original if audio_original is not None else audio_steered
        time = np.linspace(0, len(audio_ref) / sample_rate, len(audio_ref))

        # Original: ghosted fill
        if audio_original is not None:
            ax.fill_between(
                time, audio_original, 0,
                color=COLORS["original_fill"],
                alpha=0.4,
            )
            ax.plot(
                time, audio_original,
                color=COLORS["original"],
                linewidth=0.5,
                alpha=0.6,
                label="ORIGINAL",
            )

        # Steered: solid accent
        if audio_steered is not None:
            ax.plot(
                time, audio_steered,
                color=COLORS["accent"],
                linewidth=0.8,
                alpha=0.9,
                label="STEERED",
            )

        ax.set_xlabel("TIME (s)", color=COLORS["text_dim"], fontsize=9, fontfamily="monospace")
        ax.set_ylabel("AMP", color=COLORS["text_dim"], fontsize=9, fontfamily="monospace")
        ax.set_xlim(0, time[-1])

        # Legend
        ax.legend(
            loc="upper right",
            facecolor=COLORS["surface"],
            edgecolor=COLORS["border"],
            labelcolor=COLORS["text_dim"],
            fontsize=8,
            framealpha=0.9,
        )

    ax.axhline(y=0, color=COLORS["border"], linewidth=0.5)
    plt.tight_layout(pad=0.5)
    return fig


def create_spectrogram_comparison(
    audio_original: np.ndarray | None,
    audio_steered: np.ndarray | None,
    sample_rate: int = 44100,
) -> plt.Figure:
    """Create side-by-side spectrograms with shared colorscale."""
    # Ensure 1D audio
    audio_original = _ensure_1d(audio_original)
    audio_steered = _ensure_1d(audio_steered)

    fig, axes = plt.subplots(1, 2, figsize=(14, 3), facecolor=COLORS["bg"])

    for ax in axes:
        _setup_ax(ax)

    if audio_original is None and audio_steered is None:
        for ax in axes:
            ax.text(
                0.5, 0.5, "NO AUDIO",
                ha="center", va="center",
                color=COLORS["text_muted"],
                fontsize=12, fontfamily="monospace",
                transform=ax.transAxes,
            )
    else:
        from scipy import signal

        # Compute spectrograms
        specs = []
        for audio in [audio_original, audio_steered]:
            if audio is not None:
                f, t, Sxx = signal.spectrogram(
                    audio, fs=sample_rate,
                    nperseg=1024, noverlap=768,
                )
                Sxx_db = 10 * np.log10(Sxx + 1e-10)
                specs.append((f, t, Sxx_db))
            else:
                specs.append(None)

        # Find shared vmin/vmax
        all_db = [s[2] for s in specs if s is not None]
        if all_db:
            vmax = max(s.max() for s in all_db)
            vmin = vmax - 60  # 60dB dynamic range

        titles = ["ORIGINAL", "STEERED"]
        for ax, spec, title in zip(axes, specs, titles):
            if spec is not None:
                f, t, Sxx_db = spec
                ax.pcolormesh(
                    t, f, Sxx_db,
                    shading="gouraud",
                    cmap="magma",
                    vmin=vmin, vmax=vmax,
                )
                ax.set_ylim(0, 8000)
                ax.set_ylabel("FREQ (Hz)", color=COLORS["text_dim"], fontsize=8, fontfamily="monospace")
                ax.set_xlabel("TIME (s)", color=COLORS["text_dim"], fontsize=8, fontfamily="monospace")

            ax.set_title(
                title,
                color=COLORS["text"],
                fontsize=10, fontfamily="monospace",
                pad=4,
            )

    plt.tight_layout(pad=0.5)
    return fig


def create_alpha_envelope(
    mode: str,
    params: dict[str, float],
    label: str = "",
) -> plt.Figure:
    """
    Visualize the alpha envelope for current steering mode.

    Shows how steering strength varies across the 32 timesteps.
    """
    fig, ax = plt.subplots(figsize=(14, 3), facecolor=COLORS["bg"])
    _setup_ax(ax)

    n = CONFIG.n_timesteps
    timesteps = np.arange(n)
    time_ms = timesteps * (1500 / n)  # Convert to ms

    # Compute alpha envelope based on mode
    if mode == "uniform":
        alphas = np.full(n, params.get("alpha", 0.0))
    elif mode == "segment":
        alphas = np.zeros(n)
        alphas[:CONFIG.attack_end] = params.get("alpha_attack", 0.0)
        alphas[CONFIG.attack_end:CONFIG.body_end] = params.get("alpha_body", 0.0)
        alphas[CONFIG.body_end:] = params.get("alpha_tail", 0.0)
    elif mode == "envelope":
        alphas = np.linspace(
            params.get("alpha_start", 0.0),
            params.get("alpha_end", 0.0),
            n,
        )
    elif mode == "temporal_features":
        alphas = np.zeros(n)
        alphas[:CONFIG.attack_end] = params.get("alpha_attack", 0.0)
        alphas[CONFIG.attack_end:] = params.get("alpha_sustain", 0.0)
    else:
        alphas = np.zeros(n)

    # Fill area
    ax.fill_between(
        time_ms, 0, alphas,
        alpha=0.3,
        color=COLORS["accent"],
        step="mid" if mode in ["segment", "temporal_features"] else None,
    )

    # Line
    if mode in ["segment", "temporal_features"]:
        ax.step(time_ms, alphas, where="mid", color=COLORS["accent"], linewidth=2)
    else:
        ax.plot(time_ms, alphas, color=COLORS["accent"], linewidth=2)

    # Zero line
    ax.axhline(y=0, color=COLORS["text_muted"], linewidth=0.5, linestyle="--")

    # Phase boundaries
    attack_ms = CONFIG.attack_end * (1500 / n)
    body_ms = CONFIG.body_end * (1500 / n)

    ax.axvline(x=attack_ms, color=COLORS["text_dim"], linewidth=1, linestyle=":", alpha=0.7)
    ax.axvline(x=body_ms, color=COLORS["text_dim"], linewidth=1, linestyle=":", alpha=0.7)

    # Phase labels
    y_max = max(abs(alphas.min()), abs(alphas.max()), 0.5)
    y_pos = y_max * 0.8

    for x, text in [(attack_ms / 2, "ATK"), ((attack_ms + body_ms) / 2, "BODY"), ((body_ms + 1500) / 2, "TAIL")]:
        ax.text(
            x, y_pos, text,
            ha="center", va="center",
            color=COLORS["text_muted"],
            fontsize=8, fontfamily="monospace",
            alpha=0.7,
        )

    ax.set_xlabel("TIME (ms)", color=COLORS["text_dim"], fontsize=9, fontfamily="monospace")
    ylabel = f"ALPHA" if not label else f"ALPHA ({label.upper()})"
    ax.set_ylabel(ylabel, color=COLORS["text_dim"], fontsize=9, fontfamily="monospace")
    ax.set_xlim(0, 1500)
    ax.set_ylim(-y_max * 1.2, y_max * 1.2)

    plt.tight_layout(pad=0.5)
    return fig


def create_feature_correlation_plot(
    correlations: np.ndarray,
    top_indices: np.ndarray,
    label: str,
) -> plt.Figure:
    """
    Bar chart showing top feature correlations for a property.

    Args:
        correlations: Full correlation array (1024,)
        top_indices: Indices of top features
        label: Property name
    """
    fig, ax = plt.subplots(figsize=(4, 2), facecolor=COLORS["bg"])
    _setup_ax(ax)

    top_corrs = correlations[top_indices]
    x = np.arange(len(top_indices))

    # Color by sign
    colors = [COLORS["accent"] if c > 0 else COLORS["text_dim"] for c in top_corrs]

    ax.bar(x, top_corrs, color=colors, edgecolor=COLORS["border"], linewidth=0.5)
    ax.axhline(y=0, color=COLORS["border"], linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in top_indices], fontsize=7, rotation=45)
    ax.set_xlabel("FEATURE IDX", color=COLORS["text_dim"], fontsize=8, fontfamily="monospace")
    ax.set_ylabel("CORR", color=COLORS["text_dim"], fontsize=8, fontfamily="monospace")
    ax.set_title(
        f"{label.upper()} TOP FEATURES",
        color=COLORS["text"],
        fontsize=9, fontfamily="monospace",
        pad=4,
    )

    plt.tight_layout(pad=0.5)
    return fig


def create_empty_plot(message: str = "NO DATA") -> plt.Figure:
    """Create an empty placeholder plot."""
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=COLORS["bg"])
    _setup_ax(ax)
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        color=COLORS["text_muted"],
        fontsize=12, fontfamily="monospace",
        transform=ax.transAxes,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return fig
