#!/usr/bin/env python3
"""Extract per-timestep audio features for SAE training.

This script processes all audio files in a directory and outputs
per-timestep features aligned with VAE latent timesteps.

The key improvement over v1: each timestep gets its OWN features
instead of duplicating whole-clip metadata 32 times.

Usage:
    python scripts/01_extract_features.py \\
        --input_dir data/GT/one_shot_percussive_sounds \\
        --output_path data/features_v2.parquet \\
        --target_length 32768

Output:
    - features_v2.parquet: Per-timestep features (~164K rows)
    - features_v2.json: Config for reproducibility
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drums_SAE.data.preprocess import (
    PreprocessConfig,
    load_audio,
    preprocess_to_mono,
)
from drums_SAE.data.features import (
    FeatureConfig,
    extract_timestep_features,
    should_include_timestep,
)


def iter_audio_files(input_dir: Path) -> Iterator[tuple[str, Path]]:
    """Iterate over audio files in sorted order.

    Sorting is critical for reproducibility and alignment with
    VAE encoding which uses the same ordering.

    Yields:
        (sample_id, path) tuples sorted by sample_id
    """
    wav_files = sorted(input_dir.rglob("*.wav"))
    for path in wav_files:
        sample_id = path.stem  # e.g., "10465"
        yield sample_id, path


def process_audio_file(
    sample_id: str,
    path: Path,
    preprocess_config: PreprocessConfig,
    feature_config: FeatureConfig,
    device: torch.device,
) -> list[dict]:
    """Process one audio file and return feature dictionaries.

    Returns:
        List of feature dicts, one per timestep
    """
    # Load and preprocess (using shared preprocessing for alignment)
    audio, sr = load_audio(str(path), device)
    audio_mono = preprocess_to_mono(audio, sr, preprocess_config, device)

    # Extract features for all timesteps
    timestep_features = extract_timestep_features(
        audio_mono,
        sample_id,
        preprocess_config,
        feature_config,
    )

    return [f.to_dict() for f in timestep_features]


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-timestep audio features for SAE training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing audio files (searches recursively for *.wav)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output parquet file path",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=32768,
        help="Target audio length in samples (32768 = 16 timesteps, 65536 = 32 timesteps)",
    )
    parser.add_argument(
        "--silence_threshold_db",
        type=float,
        default=-40.0,
        help="Silence threshold in dB for phase detection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detects if not specified.",
    )
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Device: {device}")

    # Configs
    preprocess_config = PreprocessConfig(target_length=args.target_length)
    feature_config = FeatureConfig(silence_threshold_db=args.silence_threshold_db)

    print(f"Target length: {preprocess_config.target_length} samples")
    print(f"Timesteps per file: {preprocess_config.n_timesteps}")
    print(f"Seconds per timestep: {preprocess_config.seconds_per_timestep:.4f}s")
    print(f"Silence threshold: {feature_config.silence_threshold_db} dB")
    print()

    # Collect audio files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    audio_files = list(iter_audio_files(input_dir))
    print(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print("No .wav files found. Check the input directory.")
        sys.exit(1)

    # Process files
    all_features = []
    errors = []

    for sample_id, path in tqdm(audio_files, desc="Extracting features"):
        try:
            features = process_audio_file(
                sample_id, path, preprocess_config, feature_config, device
            )
            all_features.extend(features)
        except Exception as e:
            errors.append((sample_id, str(e)))
            continue

    if errors:
        print(f"\nWarning: {len(errors)} files failed to process:")
        for sample_id, error in errors[:10]:  # Show first 10
            print(f"  {sample_id}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Create DataFrame
    df = pd.DataFrame(all_features)

    # Sort by sample_id (numeric) then timestep for consistent ordering
    # This matches the order in VAE encoding
    df["sample_id_int"] = pd.to_numeric(df["sample_id"], errors="coerce")
    df = df.sort_values(["sample_id_int", "timestep"]).drop(columns=["sample_id_int"])
    df = df.reset_index(drop=True)

    # Summary stats
    n_total = len(df)
    n_silence = df["is_silence"].sum()
    n_attack = (df["phase"] == "attack").sum()
    n_decay = (df["phase"] == "decay").sum()
    n_trainable = df.apply(
        lambda row: not row["is_silence"] and row["rms_db"] >= feature_config.min_energy_db,
        axis=1
    ).sum()

    print(f"\n{'='*50}")
    print(f"Extracted {n_total:,} timesteps from {len(audio_files):,} files")
    print(f"{'='*50}")
    print(f"  Attack:   {n_attack:>8,} ({n_attack/n_total:>6.1%})")
    print(f"  Decay:    {n_decay:>8,} ({n_decay/n_total:>6.1%})")
    print(f"  Silence:  {n_silence:>8,} ({n_silence/n_total:>6.1%})")
    print(f"{'='*50}")
    print(f"  Trainable (non-silence, >= {feature_config.min_energy_db} dB): {n_trainable:,}")
    print()

    # Feature statistics (non-silence only)
    non_silence = df[~df["is_silence"]]
    if len(non_silence) > 0:
        print("Feature statistics (non-silence timesteps):")
        for col in ["spectral_centroid", "spectral_flatness", "crest_factor", "sub_bass", "bass", "mid", "high"]:
            if col in non_silence.columns:
                vals = non_silence[col].dropna()
                if len(vals) > 0:
                    print(f"  {col:20s}: mean={vals.mean():8.4f}, std={vals.std():8.4f}")
        print()

    # Save features
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try parquet first, fall back to CSV if pyarrow not available
    if output_path.suffix == ".parquet":
        try:
            df.to_parquet(output_path, index=False)
        except ImportError:
            print("Warning: pyarrow not installed, saving as CSV instead")
            output_path = output_path.with_suffix(".csv")
            df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"Saved features to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Save config for reproducibility
    config_path = output_path.with_suffix(".json")
    config_data = {
        "preprocess": {
            "sample_rate": preprocess_config.sample_rate,
            "target_length": preprocess_config.target_length,
            "target_channels": preprocess_config.target_channels,
            "vae_downsample_factor": preprocess_config.vae_downsample_factor,
            "n_timesteps": preprocess_config.n_timesteps,
            "seconds_per_timestep": preprocess_config.seconds_per_timestep,
        },
        "features": {
            "silence_threshold_db": feature_config.silence_threshold_db,
            "min_energy_db": feature_config.min_energy_db,
            "attack_decay_ratio": feature_config.attack_decay_ratio,
            "band_edges": list(feature_config.band_edges),
            "band_names": list(feature_config.band_names),
        },
        "input_dir": str(input_dir.absolute()),
        "n_files": len(audio_files),
        "n_errors": len(errors),
        "n_timesteps_total": n_total,
        "n_trainable": int(n_trainable),
        "phase_distribution": {
            "attack": int(n_attack),
            "decay": int(n_decay),
            "silence": int(n_silence),
        },
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved config to: {config_path}")


if __name__ == "__main__":
    main()
