#!/usr/bin/env python3
"""
Entry point for Drums SAE Steering Demos.

Usage:
    # V1: Interactive single-sample steering (original demo)
    python demo/run.py --v1
    # or just:
    python demo/run.py

    # V2: Bulk testing with residual preservation
    python demo/run.py --v2

Note: On macOS, you may need to set the FFmpeg library path:
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib python demo/run.py --v2
"""

import argparse
import os
import sys
from pathlib import Path

# Set macOS audio library path
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/ffmpeg7/lib")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Drums SAE Steering Demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo/run.py --v1    # Interactive demo (32 timesteps)
    python demo/run.py --v2    # Bulk testing demo (16 timesteps, residual trick)
        """,
    )
    parser.add_argument(
        "--v1",
        action="store_true",
        help="Run V1 demo: Interactive single-sample steering with control vectors",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Run V2 demo: Bulk testing with direct feature manipulation and residual preservation",
    )

    args = parser.parse_args()

    # Default to v1 if neither specified
    if not args.v1 and not args.v2:
        args.v1 = True

    if args.v2:
        print("\n[DEMO] Launching V2 — Bulk Testing Demo")
        from demo.v2.app import main as v2_main
        v2_main()
    else:
        print("\n[DEMO] Launching V1 — Interactive Demo")
        # V1 still uses the old app.py at the root level for now
        from demo.app import main as v1_main
        v1_main()


if __name__ == "__main__":
    main()
