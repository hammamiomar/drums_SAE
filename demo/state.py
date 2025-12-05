"""State management and steering helpers for the demo."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .config import CONFIG, MODE_SLIDERS


@dataclass
class SteeringHistoryItem:
    """A single entry in the steering history."""

    sample_idx: int
    mode: str
    property_name: str
    params: dict[str, float]
    timestamp: float = 0.0

    def summary(self) -> str:
        """Short text summary for display."""
        alphas = [f"{v:+.1f}" for v in self.params.values()]
        return f"{self.property_name}: {' / '.join(alphas)}"


@dataclass
class SessionState:
    """
    All mutable state for a demo session.

    Stored in gr.State and passed through event handlers.
    """

    # Sample state
    current_sample_idx: int = 0
    z_original: torch.Tensor | None = None
    z_steered: torch.Tensor | None = None

    # Audio state
    audio_original: np.ndarray | None = None
    audio_steered: np.ndarray | None = None
    current_playback: str = "A"  # "A" (original) or "B" (steered)

    # Steering state
    current_mode: str = "uniform"
    current_property: str = "brightness"

    # Property params: {property_name: {param_key: value}}
    # e.g., {"brightness": {"alpha": 0.5}, "depth": {"alpha_attack": 0.3, ...}}
    property_params: dict[str, dict[str, float]] = field(default_factory=dict)

    # History
    steering_history: list[SteeringHistoryItem] = field(default_factory=list)

    def get_params_for_property(self, prop: str, mode: str) -> dict[str, float]:
        """Get params for a property, initializing if needed."""
        if prop not in self.property_params:
            self.property_params[prop] = {}

        params = self.property_params[prop]
        # Ensure all keys for this mode exist
        for key in MODE_SLIDERS[mode]:
            if key not in params:
                params[key] = CONFIG.alpha_default
        return params

    def set_param(self, prop: str, key: str, value: float) -> None:
        """Set a single parameter value."""
        if prop not in self.property_params:
            self.property_params[prop] = {}
        self.property_params[prop][key] = value

    def add_to_history(self, item: SteeringHistoryItem) -> None:
        """Add item to history, maintaining max size."""
        self.steering_history.insert(0, item)
        if len(self.steering_history) > CONFIG.max_history:
            self.steering_history.pop()

    def clear_steering(self) -> None:
        """Reset all steering params to default."""
        self.property_params = {}
        self.z_steered = None
        self.audio_steered = None


def build_property_params(
    mode: str,
    labels: list[str],
    slider_values: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    Build property_params dict from slider values based on mode.

    This is the shared function used by both process_steering and save_audio
    to avoid the bug where save only handled uniform mode.

    Args:
        mode: One of "uniform", "segment", "envelope", "temporal_features"
        labels: List of property labels (e.g., ["brightness", "depth", ...])
        slider_values: Dict mapping slider names to values

    Returns:
        Dict mapping property name to param dict
        e.g., {"brightness": {"alpha": 0.5}, "depth": {"alpha_attack": 0.3, ...}}
    """
    property_params: dict[str, dict[str, float]] = {}

    for label in labels:
        if mode == "uniform":
            key = f"{label}_uniform"
            if key in slider_values:
                property_params[label] = {"alpha": slider_values[key]}

        elif mode == "segment":
            keys = [f"{label}_attack", f"{label}_body", f"{label}_tail"]
            if all(k in slider_values for k in keys):
                property_params[label] = {
                    "alpha_attack": slider_values[keys[0]],
                    "alpha_body": slider_values[keys[1]],
                    "alpha_tail": slider_values[keys[2]],
                }

        elif mode == "envelope":
            keys = [f"{label}_start", f"{label}_end"]
            if all(k in slider_values for k in keys):
                property_params[label] = {
                    "alpha_start": slider_values[keys[0]],
                    "alpha_end": slider_values[keys[1]],
                }

        elif mode == "temporal_features":
            keys = [f"{label}_temporal_attack", f"{label}_temporal_sustain"]
            if all(k in slider_values for k in keys):
                property_params[label] = {
                    "alpha_attack": slider_values[keys[0]],
                    "alpha_sustain": slider_values[keys[1]],
                }

    return property_params


def format_alpha_display(params: dict[str, float], mode: str) -> str:
    """Format alpha values for display in data panel."""
    if not params:
        return "---"

    if mode == "uniform":
        alpha = params.get("alpha", 0)
        return f"{alpha:+.2f}"

    elif mode == "segment":
        a = params.get("alpha_attack", 0)
        b = params.get("alpha_body", 0)
        t = params.get("alpha_tail", 0)
        return f"A:{a:+.1f} B:{b:+.1f} T:{t:+.1f}"

    elif mode == "envelope":
        s = params.get("alpha_start", 0)
        e = params.get("alpha_end", 0)
        return f"{s:+.1f} -> {e:+.1f}"

    elif mode == "temporal_features":
        a = params.get("alpha_attack", 0)
        s = params.get("alpha_sustain", 0)
        return f"ATK:{a:+.1f} SUS:{s:+.1f}"

    return "---"


def has_active_steering(property_params: dict[str, dict[str, float]]) -> bool:
    """Check if any property has non-zero steering."""
    for params in property_params.values():
        if any(abs(v) > 1e-6 for v in params.values()):
            return True
    return False


def get_active_properties(property_params: dict[str, dict[str, float]]) -> list[str]:
    """Get list of properties with non-zero steering."""
    active = []
    for label, params in property_params.items():
        if any(abs(v) > 1e-6 for v in params.values()):
            active.append(label)
    return active
