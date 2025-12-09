"""Configuration for V2 Demo - Bulk testing with direct feature manipulation."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ConfigV2:
    """Immutable configuration for v2 demo."""

    # Paths (relative to project root)
    checkpoint_path: str = "experiments/v2_main/checkpoints/sae_latest.pt"
    latent_data_path: str = "data/latents_v2.npz"
    features_path: str = "data/features_v2.csv"
    output_dir: str = "outputs/steering_v2"

    # Audio
    sample_rate: int = 44100

    # Temporal structure (v2: 16 timesteps = 0.74s)
    n_timesteps: int = 16
    attack_end: int = 4   # ~185ms (scaled from v1's 8/32)
    body_end: int = 12    # ~555ms (scaled from v1's 24/32)

    # Model architecture (from training)
    d_input: int = 64
    expansion_factor: int = 64  # 4096 features
    topk: int = 32

    # Steering config
    alpha_min: float = -2.0
    alpha_max: float = 2.0
    alpha_default: float = 0.5
    alpha_step: float = 0.1

    # Bulk testing defaults
    default_n_samples: int = 10
    max_n_samples: int = 50

    # Feature properties we can steer by
    # Maps property name -> best feature index (from Session 2 correlation analysis)
    property_features: dict = field(default_factory=lambda: {
        "sub_bass": 1852,      # ρ = 0.49, +336% when active
        "spectral_centroid": None,  # Will be populated from analysis
        "bass": None,
        "crest_factor": None,
        "rms": None,
    })

    # Properties available in features_v2.csv
    available_properties: tuple[str, ...] = (
        "spectral_centroid",
        "spectral_flatness",
        "sub_bass",
        "bass",
        "low_mid",
        "mid",
        "high_mid",
        "high",
        "crest_factor",
        "rms",
    )


def get_project_root() -> Path:
    """Get project root directory."""
    # demo/v2/config.py -> v2 -> demo -> project_root
    return Path(__file__).resolve().parent.parent.parent


CONFIG = ConfigV2()
PROJECT_ROOT = get_project_root()
