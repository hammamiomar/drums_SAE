#!/usr/bin/env python3
"""Encode audio files to VAE latents for SAE training.

This script uses the SAME preprocessing as feature extraction to
guarantee alignment between latent rows and metadata rows.

Usage:
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib \\
    python scripts/02_encode_latents.py \\
        --input_dir data/GT/one_shot_percussive_sounds \\
        --output_path data/latents_v2.npz \\
        --target_length 32768

Output:
    - latents_v2.npz: Contains 'latents' (N, 64), 'mean' (64,), 'std' (64,)
    - latents_v2.json: Config for reproducibility
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drums_SAE.data.preprocess import (
    PreprocessConfig,
    load_audio,
    preprocess_for_vae,
)


def iter_audio_files(input_dir: Path) -> Iterator[tuple[str, Path]]:
    """Iterate over audio files in sorted order.

    CRITICAL: Must use the same ordering as feature extraction
    to ensure row-by-row alignment.
    """
    wav_files = sorted(input_dir.rglob("*.wav"))
    for path in wav_files:
        sample_id = path.stem
        yield sample_id, path


def main():
    parser = argparse.ArgumentParser(
        description="Encode audio files to VAE latents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output npz file path",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=32768,
        help="Target audio length in samples (must match feature extraction)",
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

    # Config - MUST match feature extraction
    config = PreprocessConfig(target_length=args.target_length)
    print(f"Target length: {config.target_length} samples")
    print(f"Timesteps per file: {config.n_timesteps}")
    print()

    # Load VAE model
    print("Loading Stable Audio Open model...")
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)
    model.requires_grad_(False)
    vae = model.pretransform

    # Verify sample rate matches
    target_sr = model_config["sample_rate"]
    assert target_sr == config.sample_rate, (
        f"Sample rate mismatch: model expects {target_sr}, config has {config.sample_rate}"
    )
    print(f"Model sample rate: {target_sr}")
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
    latents_accumulator = []
    sample_ids = []
    errors = []

    with torch.no_grad():
        for sample_id, path in tqdm(audio_files, desc="Encoding latents"):
            try:
                # Load audio
                audio, sr = load_audio(str(path), device)

                # Preprocess using SHARED function (guarantees alignment)
                input_tensor = preprocess_for_vae(audio, sr, config, device)

                # Encode to latents
                # Output: (1, 64, n_timesteps)
                encoded_latents = vae.encode(input_tensor)

                # Flatten: (1, 64, n_timesteps) -> (n_timesteps, 64)
                flat_z = rearrange(encoded_latents, "b c t -> (b t) c").cpu().numpy()

                latents_accumulator.append(flat_z)
                sample_ids.extend([sample_id] * flat_z.shape[0])

            except Exception as e:
                errors.append((sample_id, str(e)))
                continue

    if errors:
        print(f"\nWarning: {len(errors)} files failed to process:")
        for sample_id, error in errors[:10]:
            print(f"  {sample_id}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Concatenate all latents
    all_latents = np.concatenate(latents_accumulator, axis=0)
    print(f"\nTotal latents: {all_latents.shape}")

    # Compute normalization statistics
    mean = np.mean(all_latents, axis=0)
    std = np.std(all_latents, axis=0)

    print(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Std range: [{std.min():.4f}, {std.max():.4f}]")

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        latents=all_latents,
        mean=mean,
        std=std,
    )
    print(f"\nSaved latents to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Save config for reproducibility
    config_path = output_path.with_suffix(".json")
    config_data = {
        "preprocess": {
            "sample_rate": config.sample_rate,
            "target_length": config.target_length,
            "target_channels": config.target_channels,
            "vae_downsample_factor": config.vae_downsample_factor,
            "n_timesteps": config.n_timesteps,
        },
        "model": {
            "name": "stabilityai/stable-audio-open-1.0",
            "sample_rate": target_sr,
        },
        "input_dir": str(input_dir.absolute()),
        "n_files": len(audio_files),
        "n_errors": len(errors),
        "n_latents_total": all_latents.shape[0],
        "latent_dim": all_latents.shape[1],
        "mean_stats": {
            "min": float(mean.min()),
            "max": float(mean.max()),
            "mean": float(mean.mean()),
        },
        "std_stats": {
            "min": float(std.min()),
            "max": float(std.max()),
            "mean": float(std.mean()),
        },
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved config to: {config_path}")

    # Alignment check reminder
    print("\n" + "=" * 60)
    print("IMPORTANT: Verify alignment with features!")
    print("  - Latent rows and feature rows should match 1:1")
    print(f"  - Expected: {len(audio_files)} files x {config.n_timesteps} timesteps")
    print(f"  - Actual: {all_latents.shape[0]} latent vectors")
    print("=" * 60)


if __name__ == "__main__":
    main()
