#!/usr/bin/env python3
"""Generate steered drum sounds from noise.

This script demonstrates diffusion-time SAE steering: generating audio
with controlled acoustic properties by intervening during the sampling loop.

Usage:
    # Basic generation with bass boost
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib uv run python scripts/generate_steered.py \
        --prompt "punchy kick drum" \
        --bass 1.5 \
        --output outputs/bassy_kick.wav

    # Darker hi-hat
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib uv run python scripts/generate_steered.py \
        --prompt "hi-hat" \
        --brightness -1.0 \
        --output outputs/dark_hihat.wav

    # Multiple properties
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib uv run python scripts/generate_steered.py \
        --prompt "snare hit" \
        --punchiness 1.0 \
        --loudness 0.5 \
        --output outputs/punchy_snare.wav

    # Generate comparison triplet (less, baseline, more)
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib uv run python scripts/generate_steered.py \
        --prompt "kick drum" \
        --bass 1.5 \
        --compare \
        --output outputs/kick_comparison

Available Properties:
    --bass       : Body/warmth (maps to V2 'bass')
    --brightness : Timbral brightness (maps to V2 'spectral_centroid')
    --loudness   : Overall level (maps to V2 'rms')
    --punchiness : Attack sharpness (maps to V2 'crest_factor')

Typical alpha values: -2.0 to 2.0 (negative = less, positive = more)
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torchaudio

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drums_SAE.diffusion import generate_steered_audio, load_sae, load_stable_audio
from drums_SAE.diffusion.generate import generate_comparison
from drums_SAE.steering.probe_steer import ProbeSteeringVectors


def setup_logging(verbose: bool = False):
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate steered drum sounds using SAE-based control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Prompt and output
    parser.add_argument(
        "--prompt",
        type=str,
        default="drum hit",
        help="Text prompt for generation (default: 'drum hit')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="steered_output.wav",
        help="Output file path (default: steered_output.wav)",
    )

    # Steering controls (user-friendly names)
    steering = parser.add_argument_group("Steering Controls")
    steering.add_argument(
        "--bass",
        type=float,
        default=0.0,
        help="Body/warmth: negative=thinner, positive=more bass (-2 to 2)",
    )
    steering.add_argument(
        "--brightness",
        type=float,
        default=0.0,
        help="Brightness: negative=darker, positive=brighter (-2 to 2)",
    )
    steering.add_argument(
        "--loudness",
        type=float,
        default=0.0,
        help="Loudness/energy: negative=quieter, positive=louder (-2 to 2)",
    )
    steering.add_argument(
        "--punchiness",
        type=float,
        default=0.0,
        help="Punchiness: negative=softer, positive=punchier (-2 to 2)",
    )

    # Generation settings
    gen = parser.add_argument_group("Generation Settings")
    gen.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Diffusion steps (default: 100)",
    )
    gen.add_argument(
        "--cfg_scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale (default: 7.0)",
    )
    gen.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed, -1 for random (default: 42)",
    )
    gen.add_argument(
        "--apply_at",
        type=str,
        default="middle",
        choices=["all", "early", "middle", "late"],
        help="When to apply steering (default: middle)",
    )

    # Comparison mode
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison triplet (less, baseline, more)",
    )

    # Model paths
    paths = parser.add_argument_group("Model Paths")
    paths.add_argument(
        "--sae_checkpoint",
        type=str,
        default="experiments/v2_main/checkpoints/sae_latest.pt",
        help="Path to SAE checkpoint",
    )
    paths.add_argument(
        "--vectors_path",
        type=str,
        default="experiments/v2_main/eval/steering_vectors.npz",
        help="Path to steering vectors",
    )

    # Misc
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    # Build property steering dict (map user-friendly names to V2 properties)
    property_steering = {}
    if args.bass != 0:
        property_steering["bass"] = args.bass
    if args.brightness != 0:
        property_steering["spectral_centroid"] = args.brightness
    if args.loudness != 0:
        property_steering["rms"] = args.loudness
    if args.punchiness != 0:
        property_steering["crest_factor"] = args.punchiness

    logger.info(f"Prompt: '{args.prompt}'")
    if property_steering:
        logger.info(f"Steering: {property_steering}")
    else:
        logger.info("Steering: None (baseline generation)")
    logger.info(f"Schedule: {args.apply_at} steps")

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.compare:
        # Comparison mode: generate triplet
        if not property_steering:
            logger.error("--compare requires at least one steering property")
            sys.exit(1)

        # Use first property for comparison
        prop_name = list(property_steering.keys())[0]
        alpha = list(property_steering.values())[0]

        logger.info(f"Generating comparison triplet for '{prop_name}' with alpha={alpha}")

        audio_less, audio_baseline, audio_more = generate_comparison(
            prompt=args.prompt,
            property_name=prop_name,
            alpha=abs(alpha),
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            device=device,
            apply_steering_at=args.apply_at,
        )

        # Save triplet with suffixes
        base_path = output_path.with_suffix("")
        sample_rate = 44100

        for audio, suffix in [
            (audio_less, "_less"),
            (audio_baseline, "_baseline"),
            (audio_more, "_more"),
        ]:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            out_file = f"{base_path}{suffix}.wav"
            torchaudio.save(out_file, audio.cpu(), sample_rate)
            logger.info(f"Saved: {out_file}")

    else:
        # Single generation
        audio = generate_steered_audio(
            prompt=args.prompt,
            property_steering=property_steering if property_steering else None,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            device=device,
            apply_steering_at=args.apply_at,
            sae_checkpoint=args.sae_checkpoint,
            vectors_path=args.vectors_path,
        )

        # Ensure audio is 2D for torchaudio
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Save
        sample_rate = 44100
        torchaudio.save(str(output_path), audio.cpu(), sample_rate)
        logger.info(f"Saved: {output_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
