"""Probe-based steering for controllable drum generation.

This module replaces the older correlation-based control vectors with
learned probe-derived steering directions. The key insight from the
Smule SAE paper: if a linear probe can predict a property from SAE features,
the probe weights define a meaningful steering direction for that property.

Advantages over correlation-based steering:
1. Probes are trained to discriminate classes, not just correlate
2. The steering direction is learned from data, not hand-crafted
3. Evaluation and steering use the same learned mapping

Usage:
    from drums_SAE.steering.probe_steer import (
        ProbeSteeringVectors,
        steer_with_probe,
        steer_with_probe_bidirectional,
    )

    # Load trained probe vectors
    vectors = ProbeSteeringVectors.load("experiments/v2_main/eval/steering_vectors.npz")

    # Steer toward "brighter" (positive alpha)
    z_bright = steer_with_probe_bidirectional(
        z, sae, vectors, "spectral_centroid", alpha=1.0
    )

    # Steer toward "darker" (negative alpha)
    z_dark = steer_with_probe_bidirectional(
        z, sae, vectors, "spectral_centroid", alpha=-1.0
    )
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from drums_SAE.sae.model import AudioSae


@dataclass
class ProbeSteeringVectors:
    """Container for probe-derived steering vectors with save/load.

    Stores both the contrast directions (for bidirectional steering)
    and the full probe weights (for per-class steering).

    Attributes:
        directions: Contrast directions, property -> (d_hidden,) unit vector
        weights: Full probe weights, property -> (n_bins, d_hidden)
        bin_edges: Discretization boundaries, property -> (n_bins + 1,)
        scaler_mean: Feature scaler mean, property -> (d_hidden,)
        scaler_std: Feature scaler std, property -> (d_hidden,)
    """

    directions: dict[str, np.ndarray]   # (d_hidden,) unit vectors
    weights: dict[str, np.ndarray]      # (n_bins, d_hidden)
    bin_edges: dict[str, np.ndarray]    # (n_bins + 1,)
    scaler_mean: dict[str, np.ndarray]  # (d_hidden,)
    scaler_std: dict[str, np.ndarray]   # (d_hidden,)

    @classmethod
    def load(cls, path: str | Path) -> "ProbeSteeringVectors":
        """Load from NPZ file saved by run_eval."""
        data = np.load(path)

        # Discover properties from the file keys (avoid pickle)
        properties = set()
        for key in data.files:
            if key.startswith("direction_"):
                prop = key.replace("direction_", "")
                properties.add(prop)

        directions = {}
        weights = {}
        bin_edges = {}
        scaler_mean = {}
        scaler_std = {}

        for prop in properties:
            directions[prop] = data[f"direction_{prop}"]
            weights[prop] = data[f"weights_{prop}"]
            bin_edges[prop] = data[f"bin_edges_{prop}"]
            scaler_mean[prop] = data[f"scaler_mean_{prop}"]
            scaler_std[prop] = data[f"scaler_std_{prop}"]

        return cls(
            directions=directions,
            weights=weights,
            bin_edges=bin_edges,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
        )

    def save(self, path: str | Path) -> None:
        """Save to NPZ file."""
        data = {}
        for prop in self.properties:
            data[f"direction_{prop}"] = self.directions[prop]
            data[f"weights_{prop}"] = self.weights[prop]
            data[f"bin_edges_{prop}"] = self.bin_edges[prop]
            data[f"scaler_mean_{prop}"] = self.scaler_mean[prop]
            data[f"scaler_std_{prop}"] = self.scaler_std[prop]

        np.savez(path, **data)

    @property
    def properties(self) -> list[str]:
        """List available steerable properties."""
        return list(self.directions.keys())

    def get_direction(self, property_name: str) -> np.ndarray:
        """Get the contrast direction for bidirectional steering.

        The direction points from low to high values of the property.
        - Positive alpha: steer toward high (e.g., brighter)
        - Negative alpha: steer toward low (e.g., darker)
        """
        return self.directions[property_name]

    def get_class_direction(
        self,
        property_name: str,
        target_bin: int,
    ) -> np.ndarray:
        """Get direction toward a specific bin class.

        Useful for steering to a particular level rather than
        just "more" or "less" of a property.

        Args:
            property_name: Which property
            target_bin: Bin index (0 = lowest, n_bins-1 = highest)

        Returns:
            Direction vector (d_hidden,)
        """
        weights = self.weights[property_name]
        direction = weights[target_bin].copy()

        # Normalize for consistent steering magnitude
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm

        return direction


def steer_with_probe(
    z: torch.Tensor,            # (n_timesteps, d_input) normalized latents
    sae: AudioSae,
    direction: np.ndarray,       # (d_hidden,) steering direction
    alpha: float = 1.0,
    preserve_residual: bool = True,
) -> torch.Tensor:
    """Steer latent using probe-derived direction with Gytis residual trick.

    The steering formula:
        f_steered = f + alpha * direction
        f_steered = rms_norm(f_steered)  # maintain hypersphere
        z_steered = decode(f_steered) + residual  # preserve quality

    Args:
        z: Normalized latent tensor, shape (T, 64) or (B, T, 64)
        sae: Trained SAE model
        direction: Unit steering direction in feature space
        alpha: Steering strength (negative for opposite direction)
        preserve_residual: If True, add back what SAE couldn't reconstruct

    Returns:
        Steered latent tensor, same shape as input

    Note:
        The Gytis residual trick is critical for quality. The SAE typically
        reconstructs ~88-94% of the signal variance; adding back the residual
        preserves fine details that would otherwise be lost during steering.
    """
    device = z.device
    direction_t = torch.from_numpy(direction).float().to(device)

    was_training = sae.training
    sae.training = False

    with torch.no_grad():
        # Encode to SAE features
        enc = sae.encode(z, return_aux=False)
        f = enc["f"]  # (T, d_hidden) RMS-normalized

        # Compute residual before modification (Gytis trick)
        z_reconstructed = sae.decode(f)
        residual = z - z_reconstructed

        # Apply steering direction
        # Direction broadcasts over batch/time dimensions
        f_steered = f + alpha * direction_t

        # Re-normalize to maintain RMSNorm constraint
        # (Decoder was trained on RMS-normalized inputs)
        f_steered = sae.rms_norm(f_steered)

        # Decode steered features
        z_steered = sae.decode(f_steered)

        # Add residual back to preserve fine details
        if preserve_residual:
            z_steered = z_steered + residual

    sae.training = was_training
    return z_steered


def steer_with_probe_bidirectional(
    z: torch.Tensor,
    sae: AudioSae,
    vectors: ProbeSteeringVectors,
    property_name: str,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Convenience function for bidirectional property steering.

    Uses the contrast direction (high - low weights) so:
    - alpha > 0: more of property (e.g., brighter, louder)
    - alpha < 0: less of property (e.g., darker, quieter)

    Args:
        z: Normalized latent tensor
        sae: Trained SAE model
        vectors: Loaded steering vectors
        property_name: Which property to steer
        alpha: Steering magnitude (positive = more, negative = less)

    Returns:
        Steered latent tensor
    """
    direction = vectors.get_direction(property_name)
    return steer_with_probe(z, sae, direction, alpha=alpha)


def steer_to_target_bin(
    z: torch.Tensor,
    sae: AudioSae,
    vectors: ProbeSteeringVectors,
    property_name: str,
    target_bin: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Steer toward a specific discretized bin.

    This enables more precise control than bidirectional steering.
    Instead of "brighter" or "darker", you can specify "brightness level 7".

    Args:
        z: Normalized latent tensor
        sae: Trained SAE model
        vectors: Loaded steering vectors
        property_name: Which property to steer
        target_bin: Target bin index (0 = lowest, n_bins-1 = highest)
        alpha: Steering strength

    Returns:
        Steered latent tensor
    """
    direction = vectors.get_class_direction(property_name, target_bin)
    return steer_with_probe(z, sae, direction, alpha=alpha)


def create_steered_triplet(
    z: torch.Tensor,
    sae: AudioSae,
    vectors: ProbeSteeringVectors,
    property_name: str,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create original, more, and less versions for A/B comparison.

    Useful for demos where you want to show:
    [Less] [Original] [More]

    Args:
        z: Normalized latent tensor
        sae: Trained SAE model
        vectors: Loaded steering vectors
        property_name: Which property to steer
        alpha: Steering magnitude

    Returns:
        (z_less, z_original, z_more) tuple of latent tensors
    """
    z_less = steer_with_probe_bidirectional(z, sae, vectors, property_name, -alpha)
    z_more = steer_with_probe_bidirectional(z, sae, vectors, property_name, alpha)

    return z_less, z, z_more
