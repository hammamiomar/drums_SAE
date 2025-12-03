"""
Steering utilities for controlling drum generation via SAE features.

Usage:
    from drums_SAE.steering import ControlVectors, steer_latent

    cv = ControlVectors.from_checkpoint("checkpoints/sae_step_50000.pt", "feature_summary.csv")
    z_steered = steer_latent(z, cv["brightness"], alpha=0.5)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from drums_SAE.sae.model import AudioSae


@dataclass
class ControlVectors:
    """Container for control vectors with metadata."""

    vectors: dict[str, torch.Tensor]  # label -> (d_input,) unit vector
    correlations: dict[str, np.ndarray]  # label -> (d_hidden,) correlations
    labels: tuple[str, ...]

    def __getitem__(self, label: str) -> torch.Tensor:
        return self.vectors[label]

    def __contains__(self, label: str) -> bool:
        return label in self.vectors

    def keys(self):
        return self.vectors.keys()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        feature_summary_path: str,
        top_k: int = 20,
        device: str = "cpu",
    ) -> "ControlVectors":
        """
        Build control vectors from trained SAE and feature correlations.

        Args:
            checkpoint_path: Path to SAE checkpoint
            feature_summary_path: Path to feature_summary.csv from evaluation
            top_k: Number of top-correlated features to use per control vector
            device: Device to load model on
        """
        # Load SAE
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        cfg = checkpoint["config"]
        model = AudioSae(
            d_input=cfg["d_input"],
            expansion_factor=cfg["expansion_factor"],
            topk=cfg["topk"],
            topk_aux=cfg["topk_aux"],
            dead_threshold=cfg["dead_threshold"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        W_dec = model.decoder.weight.data  # (d_input, d_hidden)

        # Load correlations
        df = pd.read_csv(feature_summary_path)
        corr_cols = [c for c in df.columns if c.startswith("corr_")]
        labels = tuple(c.replace("corr_", "") for c in corr_cols)

        vectors = {}
        correlations = {}

        for label in labels:
            corr = df[f"corr_{label}"].values
            correlations[label] = corr

            # Skip degenerate labels (like binary reverb)
            if np.abs(corr).max() < 0.05:
                continue

            # Top-k features by absolute correlation
            top_idx = np.argsort(np.abs(corr))[-top_k:]
            weights = torch.tensor(corr[top_idx], dtype=torch.float32)

            # Weighted sum of decoder directions
            vec = W_dec[:, top_idx] @ weights
            vec = vec / vec.norm()  # Unit normalize
            vectors[label] = vec

        return cls(vectors=vectors, correlations=correlations, labels=labels)

    @classmethod
    def from_precomputed(cls, path: str) -> "ControlVectors":
        """Load precomputed control vectors from .npz file."""
        data = np.load(path, allow_pickle=True)
        vectors = {k: torch.tensor(v) for k, v in data["vectors"].item().items()}
        correlations = data["correlations"].item()
        labels = tuple(vectors.keys())
        return cls(vectors=vectors, correlations=correlations, labels=labels)

    def save(self, path: str):
        """Save control vectors to .npz file."""
        np.savez(
            path,
            vectors={k: v.numpy() for k, v in self.vectors.items()},
            correlations=self.correlations,
        )


def steer_latent(
    z: torch.Tensor,
    direction: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Steer a latent vector along a control direction.

    Args:
        z: Original latent(s), shape (d_input,) or (batch, d_input)
        direction: Unit control vector, shape (d_input,)
        alpha: Steering strength (positive = more of property, negative = less)

    Returns:
        Steered latent(s), same shape as input
    """
    return z + alpha * direction


def steer_latent_normalized(
    z: torch.Tensor,
    direction: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Steer while preserving the original latent norm.
    Useful if the VAE is sensitive to latent magnitude.
    """
    original_norm = z.norm(dim=-1, keepdim=True)
    z_steered = z + alpha * direction
    z_steered = z_steered / z_steered.norm(dim=-1, keepdim=True) * original_norm
    return z_steered


def interpolate_latents(
    z1: torch.Tensor,
    z2: torch.Tensor,
    steps: int = 5,
    spherical: bool = False,
) -> torch.Tensor:
    """
    Interpolate between two latents.

    Args:
        z1, z2: Latent vectors, shape (d_input,)
        steps: Number of interpolation steps
        spherical: Use spherical interpolation (slerp) instead of linear

    Returns:
        Interpolated latents, shape (steps, d_input)
    """
    ts = torch.linspace(0, 1, steps)

    if spherical:
        # Normalize for slerp
        z1_norm = z1 / z1.norm()
        z2_norm = z2 / z2.norm()
        omega = torch.acos(torch.clamp(z1_norm @ z2_norm, -1, 1))

        if omega.abs() < 1e-6:
            # Vectors are nearly identical, use linear
            return torch.stack([z1 * (1 - t) + z2 * t for t in ts])

        return torch.stack(
            [
                (torch.sin((1 - t) * omega) * z1 + torch.sin(t * omega) * z2)
                / torch.sin(omega)
                for t in ts
            ]
        )
    else:
        return torch.stack([z1 * (1 - t) + z2 * t for t in ts])


def create_steering_grid(
    z: torch.Tensor,
    control_vectors: ControlVectors,
    labels: list[str],
    alphas: list[float] = [-1.0, -0.5, 0, 0.5, 1.0],
) -> dict[str, torch.Tensor]:
    """
    Create a grid of steered latents for multiple properties.

    Returns:
        Dict mapping "label_alpha" -> steered latent
    """
    results = {"original": z}

    for label in labels:
        if label not in control_vectors:
            continue
        direction = control_vectors[label]
        for alpha in alphas:
            if alpha == 0:
                continue
            key = f"{label}_{alpha:+.1f}"
            results[key] = steer_latent(z, direction, alpha)

    return results
