"""
VAE loading and audio decoding utilities.

This module is shared between v1 and v2 demos.
Provides functions for loading the Stable Audio VAE and decoding latents to audio.
"""

import numpy as np
import torch

# Device detection
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def load_vae(device: str | None = None):
    """
    Load the Stable Audio Open VAE.

    Args:
        device: Device to load model on. Defaults to auto-detected device.

    Returns:
        VAE model or None if loading fails
    """
    if device is None:
        device = DEVICE

    try:
        from stable_audio_tools import get_pretrained_model
        print("[VAE] Loading Stable Audio Open VAE...")
        model, _ = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        vae = model.pretransform.model.to(device)
        for p in vae.parameters():
            p.requires_grad = False
        print(f"[VAE] Loaded on {device}")
        return vae
    except Exception as e:
        print(f"[VAE] Failed to load: {e}")
        return None


def decode_latents_to_audio(
    z_norm: torch.Tensor,
    vae,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> np.ndarray | None:
    """
    Decode normalized latents to audio.

    Args:
        z_norm: Normalized latents, shape (n_timesteps, 64) or (batch, n_timesteps, 64)
        vae: Loaded VAE model
        latent_mean: Mean for denormalization, shape (64,)
        latent_std: Std for denormalization, shape (64,)

    Returns:
        1D mono audio array (float32, normalized to [-1, 1]), or None if VAE unavailable
    """
    if vae is None:
        return None

    from einops import rearrange

    # Add batch dim if needed
    if z_norm.dim() == 2:
        z_norm = z_norm.unsqueeze(0)

    # Denormalize: z = z_norm * std + mean
    z = z_norm * latent_std + latent_mean

    # Reshape for VAE: (batch, channels, timesteps)
    z = rearrange(z, "b t c -> b c t")

    with torch.no_grad():
        audio = vae.decode(z)

    # Convert to numpy and extract mono
    audio = audio.cpu().numpy()
    audio = np.squeeze(audio)  # Remove batch dim if present

    # If stereo (2, samples) or (samples, 2), take first channel
    if audio.ndim == 2:
        if audio.shape[0] <= 2:  # (channels, samples)
            audio = audio[0]
        else:  # (samples, channels)
            audio = audio[:, 0]

    # Ensure float32 in [-1, 1] range for Gradio
    audio = audio.astype(np.float32)
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()

    return audio


def load_latent_stats(npz_path: str, device: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load mean and std from latents NPZ file.

    Args:
        npz_path: Path to latents NPZ file
        device: Device to load tensors on

    Returns:
        Tuple of (mean, std) tensors
    """
    if device is None:
        device = DEVICE

    data = np.load(npz_path)
    latent_mean = torch.tensor(data["mean"], dtype=torch.float32).to(device)
    latent_std = torch.tensor(data["std"], dtype=torch.float32).to(device)
    return latent_mean, latent_std
