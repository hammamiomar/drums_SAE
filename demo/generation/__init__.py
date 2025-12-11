"""Steered drum generation demo package.

This demo generates drum sounds from noise using Stable Audio Open's
diffusion model, with SAE-based property steering to control acoustic
properties during generation.
"""

from .app import create_demo, load_models

__all__ = ["create_demo", "load_models"]
