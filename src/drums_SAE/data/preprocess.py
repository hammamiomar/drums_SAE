"""Audio preprocessing for drums SAE v2.

This module provides a single source of truth for audio preprocessing.
Both VAE encoding and feature extraction MUST use these functions to
guarantee alignment between latents and metadata.

Critical insight: The VAE encoder has a finite receptive field of 2048 samples
per latent timestep. Each latent vector represents a local window of audio,
not the whole file. Therefore, metadata must be extracted per-timestep from
the SAME preprocessed audio that gets encoded.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from stable_audio_tools.inference.utils import prepare_audio


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for audio preprocessing.

    All preprocessing parameters are centralized here to ensure
    VAE encoding and feature extraction use identical settings.
    """

    sample_rate: int = 44100
    target_length: int = 32768  # v2: ~0.74s = 16 timesteps (v1 was 65536 = 32 timesteps)
    target_channels: int = 2    # VAE expects stereo
    vae_downsample_factor: int = 2048  # Samples per latent timestep

    @property
    def n_timesteps(self) -> int:
        """Number of latent timesteps after VAE encoding."""
        return self.target_length // self.vae_downsample_factor

    @property
    def samples_per_timestep(self) -> int:
        """Number of audio samples per latent timestep."""
        return self.vae_downsample_factor

    @property
    def seconds_per_timestep(self) -> float:
        """Duration of each timestep in seconds (~46.4ms)."""
        return self.vae_downsample_factor / self.sample_rate


def load_audio(
    path: str,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, int]:
    """Load audio file using torchaudio.

    Args:
        path: Path to audio file (wav, mp3, etc.)
        device: Device to load audio to

    Returns:
        audio: (channels, samples) tensor
        sr: Original sample rate
    """
    audio, sr = torchaudio.load(path)
    return audio.to(device), sr


def preprocess_for_vae(
    audio: torch.Tensor,       # (channels, samples) or (samples,)
    in_sr: int,
    config: PreprocessConfig,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Preprocess audio for VAE encoding.

    This is the canonical preprocessing function. It uses the same
    prepare_audio() function from stable-audio-tools that the original
    VAE training used.

    Processing steps:
    1. Move to device
    2. Resample to target_sr (44100 Hz)
    3. PadCrop to target_length (pad short audio with zeros at END)
    4. Add batch dimension
    5. Set channel count (mono -> stereo duplication)

    Args:
        audio: Raw audio tensor from torchaudio.load()
        in_sr: Original sample rate of the audio
        config: Preprocessing configuration
        device: Target device

    Returns:
        Preprocessed audio: (1, 2, target_length) - batch, stereo, samples
    """
    return prepare_audio(
        audio,
        in_sr=in_sr,
        target_sr=config.sample_rate,
        target_length=config.target_length,
        target_channels=config.target_channels,
        device=device,
    )


def preprocess_to_mono(
    audio: torch.Tensor,       # (channels, samples) or (samples,)
    in_sr: int,
    config: PreprocessConfig,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Preprocess audio to mono for feature extraction.

    CRITICAL: This function uses preprocess_for_vae() internally to
    guarantee identical resampling, padding, and sample boundaries.
    The only difference is converting stereo to mono at the end.

    Args:
        audio: Raw audio tensor from torchaudio.load()
        in_sr: Original sample rate of the audio
        config: Preprocessing configuration
        device: Target device

    Returns:
        Preprocessed mono audio: (target_length,) numpy array
    """
    # First preprocess exactly as for VAE (handles resampling, padding)
    stereo = preprocess_for_vae(audio, in_sr, config, device)

    # Convert to mono: (1, 2, samples) -> (samples,)
    # Average the stereo channels to preserve relative energy
    mono = stereo.squeeze(0).mean(dim=0)  # (2, samples) -> (samples,)

    return mono.cpu().numpy()


def get_timestep_bounds(
    timestep: int,
    config: PreprocessConfig,
) -> tuple[int, int]:
    """Get sample indices for a given timestep.

    Each VAE latent timestep corresponds to a contiguous window of
    samples in the preprocessed audio.

    Args:
        timestep: Timestep index (0 to n_timesteps-1)
        config: Preprocessing configuration

    Returns:
        (start_sample, end_sample) - exclusive end

    Example:
        For timestep 0 with default config: (0, 2048)
        For timestep 1: (2048, 4096)
        For timestep 15: (30720, 32768)
    """
    start = timestep * config.samples_per_timestep
    end = start + config.samples_per_timestep
    return start, end


def get_timestep_audio(
    audio_mono: np.ndarray,    # (target_length,)
    timestep: int,
    config: PreprocessConfig,
) -> np.ndarray:
    """Extract the audio segment for a specific timestep.

    Args:
        audio_mono: Full preprocessed mono audio
        timestep: Timestep index
        config: Preprocessing configuration

    Returns:
        Audio segment: (samples_per_timestep,) numpy array
    """
    start, end = get_timestep_bounds(timestep, config)
    return audio_mono[start:end]
