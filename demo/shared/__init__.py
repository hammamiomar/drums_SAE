"""Shared modules for v1 and v2 demos."""

from demo.shared.state import (
    SessionState,
    SteeringHistoryItem,
    build_property_params,
    format_alpha_display,
    get_active_properties,
    has_active_steering,
)
from demo.shared.theme import BRUTALIST_CSS
from demo.shared.vae import (
    DEVICE,
    decode_latents_to_audio,
    load_latent_stats,
    load_vae,
)
from demo.shared.viz import (
    COLORS,
    create_alpha_envelope,
    create_audio_triplet_display,
    create_empty_plot,
    create_feature_correlation_plot,
    create_spectrogram_comparison,
    create_waveform_overlay,
)

__all__ = [
    # Theme
    "BRUTALIST_CSS",
    # Visualization
    "COLORS",
    "create_waveform_overlay",
    "create_spectrogram_comparison",
    "create_alpha_envelope",
    "create_feature_correlation_plot",
    "create_empty_plot",
    "create_audio_triplet_display",
    # State
    "SessionState",
    "SteeringHistoryItem",
    "build_property_params",
    "format_alpha_display",
    "get_active_properties",
    "has_active_steering",
    # VAE
    "DEVICE",
    "load_vae",
    "decode_latents_to_audio",
    "load_latent_stats",
]
