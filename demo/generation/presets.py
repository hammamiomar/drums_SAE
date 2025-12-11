"""Presets and constants for the steered generation demo."""

# Prompt presets for quick selection
PROMPT_PRESETS = [
    "punchy kick drum",
    "crisp snare hit",
    "shimmering hi-hat",
    "deep 808 bass",
    "tight tom hit",
    "clap with reverb",
    "metallic percussion",
    "boomy floor tom",
    "bright crash cymbal",
    "subby 808 kick",
    "dry snare",
    "open hi-hat",
    "closed hi-hat",
    "rim shot",
    "electronic tom",
]

# Property configuration: (internal_name, ui_label, negative_label, positive_label)
PROPERTIES = [
    ("bass", "Bass", "less", "more"),
    ("spectral_centroid", "Brightness", "darker", "brighter"),
    ("rms", "Loudness", "quieter", "louder"),
    ("crest_factor", "Punchiness", "sustained", "transient"),
]

# Internal property names for iteration
PROPERTY_NAMES = [p[0] for p in PROPERTIES]

# Human-readable property names for display
PROPERTY_DISPLAY_NAMES = {
    "bass": "Bass Energy",
    "spectral_centroid": "Brightness (Hz)",
    "rms": "Loudness (RMS)",
    "crest_factor": "Punchiness",
}

# Property colors for plots
PROPERTY_COLORS = {
    "bass": "#e74c3c",           # Red
    "spectral_centroid": "#3498db",  # Blue
    "rms": "#2ecc71",            # Green
    "crest_factor": "#9b59b6",   # Purple
}

# Generation defaults
DEFAULT_STEPS = 100
DEFAULT_CFG = 7.0
DEFAULT_SEED = 42
DEFAULT_SCHEDULE = "middle"

# Slider configuration
SLIDER_RANGE = (-3.0, 3.0)
SLIDER_STEP = 0.1

# Sample rate (from Stable Audio Open)
SAMPLE_RATE = 44100

# Model paths (relative to project root)
SAE_CHECKPOINT = "experiments/v2_main/checkpoints/sae_latest.pt"
STEERING_VECTORS_PATH = "experiments/v2_main/eval/steering_vectors.npz"
