#!/usr/bin/env python3
"""
Entry point for Drums SAE Steering Demo.

Usage:
    python demo/run.py

    # Or with FFmpeg path on macOS:
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib python demo/run.py
"""

import os
import sys
from pathlib import Path

# Set macOS audio library path
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/ffmpeg7/lib")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Launch the consolidated demo."""
    print("\n[DEMO] Launching Drums SAE Steering Demo")
    from demo.app import main as demo_main
    demo_main()


if __name__ == "__main__":
    main()
