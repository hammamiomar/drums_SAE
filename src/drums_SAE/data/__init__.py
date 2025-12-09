"""Data processing utilities for drums SAE v2.

This package contains:
- preprocess: Audio preprocessing aligned with VAE encoding
- features: Per-timestep audio feature extraction
"""

from drums_SAE.data.preprocess import (
    PreprocessConfig,
    load_audio,
    preprocess_for_vae,
    preprocess_to_mono,
    get_timestep_bounds,
)
from drums_SAE.data.features import (
    FeatureConfig,
    TimestepFeatures,
    extract_timestep_features,
    should_include_timestep,
)

__all__ = [
    "PreprocessConfig",
    "load_audio",
    "preprocess_for_vae",
    "preprocess_to_mono",
    "get_timestep_bounds",
    "FeatureConfig",
    "TimestepFeatures",
    "extract_timestep_features",
    "should_include_timestep",
]
