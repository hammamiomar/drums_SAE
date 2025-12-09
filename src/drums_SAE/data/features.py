"""Per-timestep audio feature extraction for drums SAE v2.

This module extracts audio features for each latent timestep (~46ms window).
Features are designed to correlate with steering intents:
- spectral_centroid -> brightness steering
- crest_factor -> punchiness steering
- sub_bass/bass -> boominess steering
- spectral_flatness -> noisiness steering

The key improvement over v1: features vary per-timestep instead of
duplicating whole-clip metadata 32 times.
"""

from dataclasses import dataclass, asdict
from typing import Literal

import librosa
import numpy as np

from drums_SAE.data.preprocess import PreprocessConfig, get_timestep_bounds


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature extraction."""

    # Phase detection
    silence_threshold_db: float = -40.0   # Below this = silence phase
    min_energy_db: float = -50.0          # Below this = exclude from training
    attack_decay_ratio: float = 0.5       # Drop to 50% of peak = end of attack

    # Frequency band boundaries (Hz)
    # Based on standard audio engineering bands for drums
    band_edges: tuple[int, ...] = (20, 80, 250, 600, 2500, 6000, 16000)
    band_names: tuple[str, ...] = (
        "sub_bass",   # 20-80 Hz: kick thump, sub frequencies
        "bass",       # 80-250 Hz: kick body, tom fundamentals
        "low_mid",    # 250-600 Hz: snare body, boxiness
        "mid",        # 600-2500 Hz: snare crack, presence
        "high_mid",   # 2500-6000 Hz: attack snap, hi-hat body
        "high",       # 6000-16000 Hz: air, sizzle, cymbal shimmer
    )


@dataclass
class TimestepFeatures:
    """Features for one latent timestep (~46ms of audio).

    This dataclass mirrors the schema in CLAUDE.md.
    Spectral features are None for silence timesteps.
    """

    # Identifiers
    sample_id: str
    timestep: int

    # Phase (categorical)
    phase: Literal["attack", "decay", "silence"]

    # Energy
    rms: float
    rms_db: float

    # Spectral (None if silence)
    spectral_centroid: float | None
    spectral_flatness: float | None

    # Band energies (fraction of total, sum to ~1.0)
    sub_bass: float | None
    bass: float | None
    low_mid: float | None
    mid: float | None
    high_mid: float | None
    high: float | None

    # Transient
    crest_factor: float | None
    zero_crossing_rate: float | None

    # Derived
    is_silence: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return asdict(self)


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS energy of an audio segment.

    Args:
        audio: (samples,) audio segment

    Returns:
        RMS value (linear scale)
    """
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_rms_db(rms: float, ref_max: float) -> float:
    """Convert RMS to dB relative to reference maximum.

    Args:
        rms: Linear RMS value
        ref_max: Reference maximum (typically max of full file)

    Returns:
        RMS in dB (will be <= 0 when ref_max is the true max)
    """
    return float(20 * np.log10(rms / (ref_max + 1e-10) + 1e-10))


def compute_band_energies(
    audio: np.ndarray,
    sr: int,
    config: FeatureConfig,
) -> dict[str, float]:
    """Compute energy ratio per frequency band using FFT.

    Energy ratios are normalized to sum to 1.0, representing the
    relative distribution of energy across frequency bands.

    Args:
        audio: (samples,) audio segment
        sr: Sample rate
        config: Feature configuration with band edges

    Returns:
        Dict mapping band name to energy fraction (sum ≈ 1.0)
    """
    n_fft = len(audio)

    # Compute power spectrum
    fft = np.fft.rfft(audio, n=n_fft)
    power = np.abs(fft) ** 2

    # Frequency bins
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)

    # Sum power in each band
    band_powers = {}
    for i, name in enumerate(config.band_names):
        low = config.band_edges[i]
        high = config.band_edges[i + 1]
        mask = (freqs >= low) & (freqs < high)
        band_powers[name] = float(power[mask].sum())

    # Normalize to sum to 1.0
    total = sum(band_powers.values()) + 1e-10
    return {name: power / total for name, power in band_powers.items()}


def compute_spectral_features(
    audio: np.ndarray,
    sr: int,
) -> tuple[float, float]:
    """Compute spectral centroid and flatness.

    Spectral centroid: "center of mass" of the spectrum (brightness)
    Spectral flatness: ratio of geometric to arithmetic mean (noisiness)

    Args:
        audio: (samples,) audio segment
        sr: Sample rate

    Returns:
        (spectral_centroid_hz, spectral_flatness)
    """
    # Use n_fft that fits the segment (2048 samples typical)
    n_fft = min(2048, len(audio))

    # Compute over whole segment (single frame)
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=len(audio)
    )
    flatness = librosa.feature.spectral_flatness(
        y=audio, n_fft=n_fft, hop_length=len(audio)
    )

    return float(centroid.mean()), float(flatness.mean())


def compute_crest_factor(audio: np.ndarray, rms: float) -> float:
    """Compute crest factor (peak/RMS ratio).

    Higher values indicate more transient/punchy sounds.
    Typical values: 3-10 for drums, higher for sharp attacks.

    Args:
        audio: (samples,) audio segment
        rms: Pre-computed RMS value

    Returns:
        Crest factor (dimensionless ratio)
    """
    peak = float(np.max(np.abs(audio)))
    return peak / (rms + 1e-10)


def compute_zero_crossing_rate(audio: np.ndarray) -> float:
    """Compute zero crossing rate.

    Higher values indicate more high-frequency content or noise.

    Args:
        audio: (samples,) audio segment

    Returns:
        Zero crossing rate (0 to 1)
    """
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=len(audio), hop_length=len(audio)
    )
    return float(zcr.mean())


def detect_phases(
    rms_per_timestep: np.ndarray,
    config: FeatureConfig,
    ref_max: float,
) -> list[str]:
    """Label each timestep as attack/decay/silence.

    Algorithm (from CLAUDE.md):
    1. Find peak RMS timestep
    2. Attack = timesteps 0 to first where RMS drops below 50% of peak
    3. Decay = after attack until silence threshold
    4. Silence = below -40 dB relative to max

    This models drum envelope: sharp attack, gradual decay, trailing silence.

    Args:
        rms_per_timestep: (n_timesteps,) RMS values per timestep
        config: Feature configuration
        ref_max: Reference maximum for dB conversion

    Returns:
        List of phase labels per timestep
    """
    n_timesteps = len(rms_per_timestep)

    # Convert to dB relative to file max
    rms_db = np.array([
        compute_rms_db(rms, ref_max) for rms in rms_per_timestep
    ])

    # Find peak timestep
    peak_idx = int(np.argmax(rms_per_timestep))
    peak_val = rms_per_timestep[peak_idx]

    # Find attack end: first timestep after peak where RMS < 50% of peak
    attack_threshold = peak_val * config.attack_decay_ratio
    attack_end = peak_idx

    for t in range(peak_idx, n_timesteps):
        if rms_per_timestep[t] < attack_threshold:
            attack_end = t
            break
    else:
        # Never dropped below threshold (unusual but possible)
        attack_end = n_timesteps

    # Label phases
    phases = []
    for t in range(n_timesteps):
        if rms_db[t] < config.silence_threshold_db:
            phases.append("silence")
        elif t <= attack_end:
            phases.append("attack")
        else:
            phases.append("decay")

    return phases


def extract_timestep_features(
    audio_mono: np.ndarray,        # (target_length,) preprocessed mono
    sample_id: str,
    preprocess_config: PreprocessConfig,
    feature_config: FeatureConfig,
) -> list[TimestepFeatures]:
    """Extract features for all timesteps in a preprocessed audio file.

    This is the main entry point for feature extraction. It performs
    two passes:
    1. Compute RMS for all timesteps (needed for phase detection)
    2. Extract all features per timestep

    Args:
        audio_mono: Preprocessed mono audio from preprocess_to_mono()
        sample_id: Unique identifier for this audio sample
        preprocess_config: Preprocessing configuration
        feature_config: Feature extraction configuration

    Returns:
        List of TimestepFeatures, one per timestep (length = n_timesteps)
    """
    sr = preprocess_config.sample_rate
    n_timesteps = preprocess_config.n_timesteps

    # Reference max for dB calculations (use file max)
    ref_max = float(np.max(np.abs(audio_mono)))

    # First pass: compute RMS for all timesteps (needed for phase detection)
    rms_values = []
    for t in range(n_timesteps):
        start, end = get_timestep_bounds(t, preprocess_config)
        segment = audio_mono[start:end]
        rms_values.append(compute_rms(segment))

    rms_array = np.array(rms_values)

    # Detect phases based on envelope
    phases = detect_phases(rms_array, feature_config, ref_max)

    # Second pass: extract all features
    features = []
    for t in range(n_timesteps):
        start, end = get_timestep_bounds(t, preprocess_config)
        segment = audio_mono[start:end]

        # Basic energy
        rms = rms_values[t]
        rms_db = compute_rms_db(rms, ref_max)

        phase = phases[t]
        is_silence = phase == "silence"

        # Spectral features (None if silence or near-zero energy)
        if is_silence or rms < 1e-10:
            spectral_centroid = None
            spectral_flatness = None
            band_energies = {name: None for name in feature_config.band_names}
            crest_factor = None
            zcr = None
        else:
            spectral_centroid, spectral_flatness = compute_spectral_features(
                segment, sr
            )
            band_energies = compute_band_energies(segment, sr, feature_config)
            crest_factor = compute_crest_factor(segment, rms)
            zcr = compute_zero_crossing_rate(segment)

        features.append(TimestepFeatures(
            sample_id=sample_id,
            timestep=t,
            phase=phase,
            rms=rms,
            rms_db=rms_db,
            spectral_centroid=spectral_centroid,
            spectral_flatness=spectral_flatness,
            sub_bass=band_energies.get("sub_bass"),
            bass=band_energies.get("bass"),
            low_mid=band_energies.get("low_mid"),
            mid=band_energies.get("mid"),
            high_mid=band_energies.get("high_mid"),
            high=band_energies.get("high"),
            crest_factor=crest_factor,
            zero_crossing_rate=zcr,
            is_silence=is_silence,
        ))

    return features


def should_include_timestep(
    features: TimestepFeatures,
    config: FeatureConfig,
) -> bool:
    """Determine if a timestep should be included in SAE training data.

    Excludes:
    - Silence timesteps (no useful signal)
    - Very low energy timesteps (noise floor)

    This filtering happens during training data creation, not during
    feature extraction. The full features parquet includes all timesteps
    for analysis purposes.

    Args:
        features: Extracted features for one timestep
        config: Feature configuration

    Returns:
        True if timestep should be included in training
    """
    if features.is_silence:
        return False
    if features.rms_db < config.min_energy_db:
        return False
    return True
