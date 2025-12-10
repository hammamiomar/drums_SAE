#!/usr/bin/env python3
"""Train SAE on v2 latent data with silence filtering.

Usage:
    # Default v2 training (64× expansion, topk=32, 100k steps)
    python scripts/03_train_sae.py

    # Custom experiment
    python scripts/03_train_sae.py \
        --experiment_name ablation_topk16 \
        --topk 16 \
        --num_steps 50000

    # v1-style training (for comparison)
    python scripts/03_train_sae.py \
        --data_path data/drums_encoded.npz \
        --no_filter_silence \
        --expansion_factor 16 \
        --topk 64

See CLAUDE.md for hyperparameter guidance and success criteria.
"""

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from drums_SAE.training.train import TrainConfig, train


def get_git_hash() -> str | None:
    """Get current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Train SAE on v2 latent data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment naming
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="v2_main",
        help="Name for this experiment (creates experiments/{name}/)",
    )

    # Data paths
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/latents_v2.npz",
        help="Path to latents NPZ file",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default="data/features_v2.csv",
        help="Path to features CSV (for silence filtering)",
    )
    parser.add_argument(
        "--no_filter_silence",
        action="store_true",
        help="Disable silence filtering (train on all timesteps)",
    )

    # Model architecture
    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=64,
        help="Expansion factor (d_hidden = 64 × factor)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=32,
        help="Number of features to activate per sample",
    )
    parser.add_argument(
        "--topk_aux",
        type=int,
        default=512,
        help="Number of dead features to revive with AuxK loss",
    )

    # Training
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100_000,
        help="Total training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw"],
        help="Optimizer: adam (default) or adamw (with weight decay)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )

    # Logging
    parser.add_argument(
        "--save_every",
        type=int,
        default=10_000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="drums_SAE",
        help="W&B project name",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging",
    )

    args = parser.parse_args()

    # Create experiment directory
    exp_dir = Path("experiments") / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Build config
    config = TrainConfig(
        data_path=args.data_path,
        features_path=args.features_path if not args.no_filter_silence else None,
        filter_silence=not args.no_filter_silence,
        expansion_factor=args.expansion_factor,
        topk=args.topk,
        topk_aux=args.topk_aux,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        lr=args.lr,
        num_steps=args.num_steps,
        save_every=args.save_every,
        checkpoint_dir=str(checkpoint_dir),
        wandb_project=args.wandb_project if not args.no_wandb else None,
        wandb_name=args.experiment_name,
    )

    # Print config summary
    print("=" * 60)
    print(f"SAE Training: {args.experiment_name}")
    print("=" * 60)
    print(f"  Data: {config.data_path}")
    print(f"  Features: {config.features_path}")
    print(f"  Filter silence: {config.filter_silence}")
    print(
        f"  Model: {config.d_input} → {config.d_input * config.expansion_factor} "
        f"({config.expansion_factor}× expansion)"
    )
    print(f"  TopK: {config.topk} (aux: {config.topk_aux})")
    print(f"  Optimizer: {config.optimizer} (lr={config.lr})")
    print(f"  Training: {config.num_steps:,} steps, batch={config.batch_size}")
    print(f"  Output: {exp_dir}/")
    print("=" * 60)

    # Save config before training (reproducibility)
    config_dict = asdict(config)
    config_dict["git_hash"] = get_git_hash()
    config_dict["timestamp"] = datetime.now().isoformat()

    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to {config_path}")

    # Train
    train(config)

    print(f"\nTraining complete! Checkpoints saved to {checkpoint_dir}/")


if __name__ == "__main__":
    main()
