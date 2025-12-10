"""Evaluation module for Drums SAE linear probes.

This module implements the evaluation methodology from the Smule SAE paper:
train linear probes to predict discretized acoustic properties from SAE features.
High probe accuracy indicates the SAE learned interpretable representations.
"""

from drums_SAE.eval.probes import (
    ProbeResult,
    discretize_property,
    train_probe,
    get_contrast_direction,
)
from drums_SAE.eval.run_eval import (
    EvalConfig,
    EvalResults,
    V1_PROPERTIES,
    V2_PROPERTIES,
    run_evaluation,
    load_sae_from_checkpoint,
    encode_dataset,
)

__all__ = [
    # Probe training
    "ProbeResult",
    "discretize_property",
    "train_probe",
    "get_contrast_direction",
    # Evaluation orchestration
    "EvalConfig",
    "EvalResults",
    "V1_PROPERTIES",
    "V2_PROPERTIES",
    "run_evaluation",
    "load_sae_from_checkpoint",
    "encode_dataset",
]
