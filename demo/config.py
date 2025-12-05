"""Configuration for Drums SAE Steering Demo."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Immutable configuration."""

    # Paths (relative to project root)
    checkpoint_path: str = "checkpoints/sae_step_50000.pt"
    feature_summary_path: str = "notebooks/feature_summary.csv"
    latent_data_path: str = "data/drums_encoded.npz"
    output_dir: str = "outputs/steering"

    # Audio
    sample_rate: int = 44100

    # Temporal structure
    n_timesteps: int = 32
    attack_end: int = 8
    body_end: int = 24

    # Steering properties (ordered by correlation strength)
    steering_labels: tuple[str, ...] = (
        "brightness",
        "depth",
        "loudness",
        "roughness",
        "warmth",
        "hardness",
        "sharpness",
        "boominess",
    )

    # Slider config
    alpha_min: float = -2.0
    alpha_max: float = 2.0
    alpha_default: float = 0.0
    alpha_step: float = 0.1

    # UI
    max_samples_in_dropdown: int = 100
    max_history: int = 10


# Temporal modes
TEMPORAL_MODES = {
    "uniform": "Same steering across all 32 timesteps",
    "segment": "Separate control for Attack (0-8) / Body (8-24) / Tail (24-32)",
    "envelope": "Linear interpolation from start alpha to end alpha",
    "temporal_features": "Attack-heavy features for transient, sustain-heavy for body",
}

# Mode to slider mapping
MODE_SLIDERS = {
    "uniform": ["alpha"],
    "segment": ["alpha_attack", "alpha_body", "alpha_tail"],
    "envelope": ["alpha_start", "alpha_end"],
    "temporal_features": ["alpha_attack", "alpha_sustain"],
}

# Slider labels by mode
SLIDER_LABELS = {
    "uniform": {"alpha": "ALPHA"},
    "segment": {
        "alpha_attack": "ATTACK [0-8]",
        "alpha_body": "BODY [8-24]",
        "alpha_tail": "TAIL [24-32]",
    },
    "envelope": {
        "alpha_start": "START",
        "alpha_end": "END",
    },
    "temporal_features": {
        "alpha_attack": "ATTACK FEATS",
        "alpha_sustain": "SUSTAIN FEATS",
    },
}


def get_project_root() -> Path:
    """Get project root directory."""
    # demo/config.py -> demo -> project_root
    return Path(__file__).resolve().parent.parent


CONFIG = Config()
PROJECT_ROOT = get_project_root()
