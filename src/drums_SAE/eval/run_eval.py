"""Evaluation orchestration for Drums SAE.

Handles the full evaluation pipeline:
1. Load SAE checkpoint
2. Encode dataset through SAE to get sparse features
3. Train linear probes for each acoustic property
4. Save results and steering vectors for demo integration

Supports both V1 (whole-clip metadata) and V2 (per-timestep DSP features).
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from drums_SAE.sae.model import AudioSae
from drums_SAE.eval.probes import ProbeResult, train_probe, get_contrast_direction


# =============================================================================
# V1 vs V2 Property Schemas
# =============================================================================
#
# V1: Pre-computed whole-clip metadata (duplicated across timesteps)
#     Lower probe accuracy ceiling because same value applies to different latents
#
# V2: Per-timestep DSP features we extracted
#     Higher accuracy expected due to true per-timestep alignment
# =============================================================================

V1_PROPERTIES = (
    "brightness",   # Timbral brightness
    "loudness",     # Overall loudness
    "boominess",    # Low-frequency emphasis
    "hardness",     # Attack sharpness
    "depth",        # Spatial depth
)

V2_PROPERTIES = (
    "spectral_centroid",  # Brightness (Hz)
    "rms",                # Loudness (linear energy)
    "crest_factor",       # Punchiness (peak/RMS ratio)
    "bass",               # Body/warmth (80-250Hz band)
)


@dataclass
class EvalConfig:
    """Configuration for evaluation pipeline."""

    # Probe settings
    n_bins: int = 10
    test_size: float = 0.2
    random_state: int = 42

    # Filter settings (V2 has per-timestep silence labels)
    filter_silence: bool = True

    # Batch processing
    batch_size: int = 4096


@dataclass
class EvalResults:
    """Container for all evaluation results."""

    config: EvalConfig
    checkpoint_path: str
    version: str
    probe_results: dict[str, ProbeResult] = field(default_factory=dict)
    steering_directions: dict[str, np.ndarray] = field(default_factory=dict)

    def summary_df(self) -> pd.DataFrame:
        """Create summary DataFrame of probe accuracies."""
        rows = []
        for name, result in self.probe_results.items():
            row = result.to_dict()
            row["pass"] = result.accuracy_ratio > 2.0
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, output_dir: str | Path) -> None:
        """Save all results to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary JSON (human-readable metrics)
        summary = {
            "checkpoint": self.checkpoint_path,
            "version": self.version,
            "config": asdict(self.config),
            "probes": {
                name: result.to_dict()
                for name, result in self.probe_results.items()
            },
        }
        summary_path = output_dir / "eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save steering vectors for demo integration
        # Properties are discovered from key prefixes when loading
        steering_data = {}
        for prop_name, result in self.probe_results.items():
            steering_data[f"direction_{prop_name}"] = self.steering_directions[prop_name]
            steering_data[f"weights_{prop_name}"] = result.weights
            steering_data[f"bin_edges_{prop_name}"] = result.bin_edges
            steering_data[f"scaler_mean_{prop_name}"] = result.scaler_mean
            steering_data[f"scaler_std_{prop_name}"] = result.scaler_std

        steering_path = output_dir / "steering_vectors.npz"
        np.savez(steering_path, **steering_data)

        # Save CSV summary
        csv_path = output_dir / "probe_accuracies.csv"
        self.summary_df().to_csv(csv_path, index=False)

        print(f"Results saved to {output_dir}/")
        print(f"  - eval_summary.json")
        print(f"  - steering_vectors.npz")
        print(f"  - probe_accuracies.csv")


def get_device() -> str:
    """Auto-detect best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_data_paths_for_version(version: str) -> tuple[str, str]:
    """Get (latents_path, features_path) for a model version."""
    if version == "v1":
        return ("data/drums_encoded.npz", "data/drums_encoded_metadata.csv")
    elif version == "v2":
        return ("data/latents_v2.npz", "data/features_v2.csv")
    else:
        raise ValueError(f"Unknown version: {version}. Use 'v1' or 'v2'.")


def get_properties_for_version(version: str) -> tuple[str, ...]:
    """Get the property list for a model version."""
    if version == "v1":
        return V1_PROPERTIES
    elif version == "v2":
        return V2_PROPERTIES
    else:
        raise ValueError(f"Unknown version: {version}. Use 'v1' or 'v2'.")


def load_sae_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> AudioSae:
    """Load SAE model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Target device for model

    Returns:
        Loaded SAE model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    # Handle older checkpoints that may be missing newer config fields
    cfg.setdefault("topk_aux", 128)
    cfg.setdefault("dead_threshold", 10000)

    model = AudioSae(
        d_input=cfg["d_input"],
        expansion_factor=cfg["expansion_factor"],
        topk=cfg["topk"],
        topk_aux=cfg["topk_aux"],
        dead_threshold=cfg["dead_threshold"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.training = False

    return model


def encode_dataset(
    sae: AudioSae,
    latent_path: str,
    batch_size: int = 4096,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode all latents through SAE to get sparse features.

    Args:
        sae: Trained SAE model
        latent_path: Path to latents NPZ file
        batch_size: Batch size for encoding
        device: Compute device

    Returns:
        features: SAE feature activations, shape (n_samples, d_hidden)
        mean: Dataset mean used for normalization, shape (d_input,)
        std: Dataset std used for normalization, shape (d_input,)
    """
    # Load latents
    data = np.load(latent_path)
    latents = torch.from_numpy(data["latents"]).float()
    mean = torch.from_numpy(data["mean"]).float()
    std = torch.from_numpy(data["std"]).float()

    # Normalize (same as training)
    latents_norm = (latents - mean) / (std + 1e-8)

    # Encode in batches
    all_features = []
    n_samples = len(latents_norm)

    sae.training = False
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Encoding"):
            batch = latents_norm[i:i + batch_size].to(device)
            enc = sae.encode(batch, return_aux=False)
            # Use RMS-normalized features (f), not raw activations (h)
            all_features.append(enc["f"].cpu().numpy())

    features = np.concatenate(all_features, axis=0)

    return features, data["mean"], data["std"]


def run_evaluation(
    checkpoint_path: str,
    version: str,
    latent_path: str | None = None,
    features_path: str | None = None,
    config: EvalConfig | None = None,
    device: str | None = None,
) -> EvalResults:
    """Run the full evaluation pipeline.

    Args:
        checkpoint_path: Path to SAE checkpoint
        version: "v1" or "v2" - determines property schema and data paths
        latent_path: Override auto-detected latent file path
        features_path: Override auto-detected features file path
        config: Evaluation configuration
        device: Compute device (auto-detected if None)

    Returns:
        EvalResults containing all probe results and steering vectors
    """
    if config is None:
        config = EvalConfig()
    if device is None:
        device = get_device()

    # Get version-appropriate paths and properties
    default_latent, default_features = get_data_paths_for_version(version)
    latent_path = latent_path or default_latent
    features_path = features_path or default_features
    properties = get_properties_for_version(version)

    print(f"Version: {version}")
    print(f"Properties: {properties}")
    print(f"Device: {device}")
    print("-" * 60)

    # Load SAE
    print(f"Loading checkpoint: {checkpoint_path}")
    sae = load_sae_from_checkpoint(checkpoint_path, device)
    print(f"SAE: {sae.d_input} -> {sae.d_hidden} features (topk={sae.topk})")

    # Encode dataset
    print(f"Loading latents: {latent_path}")
    sae_features, _, _ = encode_dataset(
        sae=sae,
        latent_path=latent_path,
        batch_size=config.batch_size,
        device=device,
    )
    print(f"SAE features shape: {sae_features.shape}")

    # Load property values
    print(f"Loading features: {features_path}")
    if features_path.endswith(".parquet"):
        df = pd.read_parquet(features_path)
    else:
        df = pd.read_csv(features_path)

    # Verify alignment
    if len(sae_features) != len(df):
        raise ValueError(
            f"Latent/feature mismatch: {len(sae_features)} latents vs {len(df)} features"
        )

    # Filter silence (V2 only - V1 doesn't have per-timestep silence labels)
    mask = None
    if config.filter_silence and "is_silence" in df.columns:
        mask = ~df["is_silence"].values
        sae_features = sae_features[mask]
        df = df[mask].reset_index(drop=True)
        print(f"After filtering silence: {len(df):,} samples")

    # Train probes
    results = EvalResults(
        config=config,
        checkpoint_path=checkpoint_path,
        version=version,
    )

    print("-" * 60)
    print("Training probes:")

    for prop_name in properties:
        if prop_name not in df.columns:
            print(f"  {prop_name:20s}: SKIPPED (not in features)")
            continue

        values = df[prop_name].values
        valid_count = (~pd.isna(values)).sum()

        if valid_count < 100:
            print(f"  {prop_name:20s}: SKIPPED (only {valid_count} valid samples)")
            continue

        try:
            probe_result = train_probe(
                features=sae_features,
                property_values=values,
                property_name=prop_name,
                n_bins=config.n_bins,
                test_size=config.test_size,
                random_state=config.random_state,
            )

            results.probe_results[prop_name] = probe_result

            # Extract steering direction (contrast: high - low)
            direction = get_contrast_direction(probe_result)
            results.steering_directions[prop_name] = direction

            # Print summary
            status = "PASS" if probe_result.accuracy_ratio > 2.0 else "FAIL"
            print(
                f"  {prop_name:20s}: acc={probe_result.accuracy:.3f} "
                f"({probe_result.accuracy_ratio:.1f}x chance) [{status}]"
            )

        except Exception as e:
            print(f"  {prop_name:20s}: ERROR - {e}")

    print("-" * 60)

    # Summary statistics
    if results.probe_results:
        accuracies = [r.accuracy for r in results.probe_results.values()]
        ratios = [r.accuracy_ratio for r in results.probe_results.values()]
        passing = sum(1 for r in ratios if r > 2.0)

        print(f"Mean accuracy: {np.mean(accuracies):.3f}")
        print(f"Mean accuracy ratio: {np.mean(ratios):.1f}x chance")
        print(f"Passing probes (>2x chance): {passing}/{len(ratios)}")

    return results
