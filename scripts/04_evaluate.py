#!/usr/bin/env python3
"""Evaluate SAE with linear probes.

Usage:
    # Evaluate a V2 model (default)
    python scripts/04_evaluate.py \
        --checkpoint experiments/v2_main/checkpoints/sae_latest.pt \
        --version v2

    # Evaluate a V1 model
    python scripts/04_evaluate.py \
        --checkpoint experiments/v1/checkpoints/sae_step_50000.pt \
        --version v1

    # Custom settings
    python scripts/04_evaluate.py \
        --checkpoint experiments/v2_main/checkpoints/sae_latest.pt \
        --version v2 \
        --n_bins 5 \
        --output_dir custom_eval

See EVAL_PLAN.md for methodology details and success criteria.
"""

import argparse
from pathlib import Path

from drums_SAE.eval.run_eval import (
    EvalConfig,
    run_evaluation,
    get_properties_for_version,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAE with linear probes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SAE checkpoint (.pt file)",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        choices=["v1", "v2"],
        help="Model version (determines property schema and data paths)",
    )

    # Data paths (optional, auto-detected from version)
    parser.add_argument(
        "--latent_path",
        type=str,
        default=None,
        help="Path to latents NPZ (default: auto from version)",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default=None,
        help="Path to features CSV (default: auto from version)",
    )

    # Probe settings
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="Number of bins for discretization",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for test set",
    )
    parser.add_argument(
        "--no_filter_silence",
        action="store_true",
        help="Disable silence filtering (include all timesteps)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/<exp>/eval)",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        checkpoint_path = Path(args.checkpoint)
        # experiments/v2_main/checkpoints/sae_step_100000.pt -> experiments/v2_main/eval
        args.output_dir = str(checkpoint_path.parent.parent / "eval")

    # Build config
    config = EvalConfig(
        n_bins=args.n_bins,
        test_size=args.test_size,
        filter_silence=not args.no_filter_silence,
    )

    # Print header
    print("=" * 60)
    print("SAE LINEAR PROBE EVALUATION")
    print("=" * 60)
    print()

    # Show which properties will be probed
    properties = get_properties_for_version(args.version)
    print(f"Will probe: {properties}")
    print()

    # Run evaluation
    results = run_evaluation(
        checkpoint_path=args.checkpoint,
        version=args.version,
        latent_path=args.latent_path,
        features_path=args.features_path,
        config=config,
    )

    # Save results
    print()
    results.save(args.output_dir)

    print()
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
