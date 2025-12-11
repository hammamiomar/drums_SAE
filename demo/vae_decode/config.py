"""Configuration for Drums SAE Steering Demo."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DemoConfig:
    """Immutable configuration for the consolidated demo."""

    # Audio
    sample_rate: int = 44100

    # Steering slider
    alpha_min: float = -2.0
    alpha_max: float = 2.0
    alpha_default: float = 1.0
    alpha_step: float = 0.1

    # Batch generation
    max_samples: int = 10
    default_samples: int = 5

    # Output
    output_dir: str = "outputs/steering"


# Human-readable display names for properties
PROPERTY_DISPLAY = {
    # V2 properties (DSP-extracted)
    "spectral_centroid": "Brightness",
    "rms": "Loudness",
    "crest_factor": "Punchiness",
    "bass": "Body / Warmth",
    # V1 properties (pre-computed)
    "brightness": "Brightness",
    "loudness": "Loudness",
    "boominess": "Boominess",
    "hardness": "Hardness",
    "depth": "Depth",
}


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent


CONFIG = DemoConfig()
PROJECT_ROOT = get_project_root()
