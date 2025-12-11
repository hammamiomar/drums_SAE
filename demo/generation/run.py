#!/usr/bin/env python3
"""Entry point for the steered generation demo.

Sets up required environment variables (FFmpeg path on macOS) before
importing the main application.

Usage:
    # From project root
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib uv run python -m demo.generation.run

    # Or with explicit FFmpeg path (handled by this script)
    uv run python -m demo.generation.run

    # With public sharing
    uv run python -m demo.generation.run --share
"""

import os
import sys
import logging

# === CRITICAL: Set FFmpeg path BEFORE importing audio libraries ===
# This is required on macOS with custom FFmpeg installations
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/ffmpeg7/lib")

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import argparse

import torch

from demo.generation.app import create_demo, load_models


def main():
    """Main entry point for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Steered Drum Generation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for sharing",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on (default: auto-select available port)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detected if not specified.",
    )
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")

    # Load models (this takes ~30-60 seconds)
    logger.info("Loading models... (this may take a minute)")
    load_models(device=device)

    # Create and launch demo
    logger.info("Creating demo interface...")
    demo = create_demo()

    if args.port:
        logger.info(f"Launching demo on port {args.port}")
    else:
        logger.info("Launching demo (auto-selecting available port)...")

    if args.share:
        logger.info("Creating public link...")

    demo.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
