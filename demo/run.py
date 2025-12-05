#!/usr/bin/env python3
"""
Entry point for Drums SAE Steering Demo.

Usage:
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib python demo/run.py

Or from project root:
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib python -m demo.app
"""

import os
import sys
from pathlib import Path

# Set macOS audio library path
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/ffmpeg7/lib")

# Add project root to path (demo/run.py -> demo -> project_root)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from demo.app import main

if __name__ == "__main__":
    main()
