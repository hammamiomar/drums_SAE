"""Linear probe training for SAE feature interpretability.

Following the Smule SAE paper methodology (Section 2.2):
1. Discretize continuous properties into bins (quantile-based for balanced classes)
2. Train multiclass logistic regression probes on SAE features
3. Use probe weights as steering directions

Key insight: If a linear probe can predict a property from SAE features,
the SAE has learned to encode that property. The probe weights then define
the steering direction for controllable generation.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    """Results from training a single linear probe.

    Stores both evaluation metrics and the trained weights needed for steering.

    Attributes:
        property_name: Name of the acoustic property being probed
        accuracy: Classification accuracy on held-out test set
        chance_baseline: Random baseline accuracy (1/n_bins)
        accuracy_ratio: accuracy / chance_baseline (>2.0 is good, >3.0 is great)
        n_bins: Number of discretization bins used
        n_train: Number of training samples
        n_test: Number of test samples
        weights: Probe weights for steering, shape (n_bins, d_hidden)
        bin_edges: Quantile boundaries for discretization, shape (n_bins + 1,)
        scaler_mean: StandardScaler mean for feature normalization, shape (d_hidden,)
        scaler_std: StandardScaler std for feature normalization, shape (d_hidden,)
    """
    property_name: str
    accuracy: float
    chance_baseline: float
    accuracy_ratio: float
    n_bins: int
    n_train: int
    n_test: int
    weights: np.ndarray       # (n_bins, d_hidden)
    bin_edges: np.ndarray     # (n_bins + 1,)
    scaler_mean: np.ndarray   # (d_hidden,)
    scaler_std: np.ndarray    # (d_hidden,)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (excludes large arrays)."""
        return {
            "property_name": self.property_name,
            "accuracy": float(self.accuracy),
            "chance_baseline": float(self.chance_baseline),
            "accuracy_ratio": float(self.accuracy_ratio),
            "n_bins": self.n_bins,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def discretize_property(
    values: np.ndarray,      # (n_samples,)
    n_bins: int = 10,
    method: Literal["quantile", "uniform"] = "quantile",
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize continuous property values into categorical bins.

    Args:
        values: Continuous property values, shape (n_samples,)
        n_bins: Number of bins to create
        method: "quantile" for equal-count bins (preferred),
                "uniform" for equal-width bins

    Returns:
        labels: Integer bin labels 0 to n_bins-1, shape (n_samples,)
        bin_edges: Bin boundaries, shape (n_bins + 1,)

    Note:
        Quantile binning is strongly preferred as it ensures roughly
        equal samples per class, which:
        1. Makes accuracy interpretable (not dominated by majority class)
        2. Works well with class_weight='balanced' in LogisticRegression
        3. Matches the Smule paper methodology
    """
    if method == "quantile":
        # Compute quantile boundaries
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(values, percentiles)

        # Handle edge case: many identical values cause duplicate edges
        # Use unique edges and reduce n_bins if necessary
        bin_edges_unique = np.unique(bin_edges)
        if len(bin_edges_unique) < n_bins + 1:
            # Fall back to uniform binning if quantiles collapse
            bin_edges = np.linspace(values.min(), values.max(), n_bins + 1)
    else:
        bin_edges = np.linspace(values.min(), values.max(), n_bins + 1)

    # Assign labels using digitize
    # digitize returns 1-indexed by default with these edges, so we adjust
    labels = np.digitize(values, bin_edges[1:-1])

    return labels, bin_edges


def train_probe(
    features: np.ndarray,         # (n_samples, d_hidden)
    property_values: np.ndarray,  # (n_samples,)
    property_name: str,
    n_bins: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
) -> ProbeResult:
    """Train a linear probe to predict discretized property from SAE features.

    Args:
        features: SAE feature activations (use RMS-normalized f, not raw h)
        property_values: Continuous property values to predict
        property_name: Name for identification/logging
        n_bins: Number of bins for discretization
        test_size: Fraction of data for held-out test set
        random_state: Random seed for reproducibility
        max_iter: Maximum iterations for LogisticRegression convergence

    Returns:
        ProbeResult with accuracy metrics and trained weights for steering

    Implementation Notes:
        - StandardScaler is essential for LogisticRegression convergence
        - class_weight='balanced' handles any remaining class imbalance
        - solver='lbfgs' works well for multiclass with many features
        - Weights are stored in original (unscaled) feature space
    """
    # Remove NaN values
    valid_mask = ~np.isnan(property_values)
    features = features[valid_mask].copy()
    property_values = property_values[valid_mask].astype(np.float64)

    n_samples = len(property_values)
    if n_samples < 100:
        raise ValueError(f"Insufficient samples for {property_name}: {n_samples}")

    # Discretize into bins
    labels, bin_edges = discretize_property(property_values, n_bins)
    actual_n_bins = len(np.unique(labels))

    if actual_n_bins < 2:
        raise ValueError(f"Only {actual_n_bins} unique bins for {property_name}")

    # Train/test split with stratification to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Scale features (critical for logistic regression convergence)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression probe
    # Note: multi_class="multinomial" is now the default in sklearn >= 1.5
    clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate on held-out test set
    accuracy = clf.score(X_test_scaled, y_test)
    chance_baseline = 1.0 / actual_n_bins

    # Transform weights back to original feature scale for steering
    # clf.coef_ is in scaled space: w_scaled @ x_scaled = w_scaled @ ((x - mean) / std)
    # To use with unscaled features: w_original = w_scaled / std
    weights_original = clf.coef_ / scaler.scale_

    return ProbeResult(
        property_name=property_name,
        accuracy=accuracy,
        chance_baseline=chance_baseline,
        accuracy_ratio=accuracy / chance_baseline,
        n_bins=actual_n_bins,
        n_train=len(y_train),
        n_test=len(y_test),
        weights=weights_original,
        bin_edges=bin_edges,
        scaler_mean=scaler.mean_,
        scaler_std=scaler.scale_,
    )


def get_contrast_direction(
    probe_result: ProbeResult,
    low_bin: int = 0,
    high_bin: int = -1,
    normalize: bool = True,
) -> np.ndarray:
    """Extract steering direction from probe weights.

    The contrast direction points from low_bin semantics toward high_bin.
    For example, with spectral_centroid: low=dark, high=bright.
    Using this direction:
    - Positive alpha = more of property (brighter)
    - Negative alpha = less of property (darker)

    Args:
        probe_result: Trained probe result with weights
        low_bin: Index of "low" class (default: 0 = lowest property values)
        high_bin: Index of "high" class (default: -1 = highest property values)
        normalize: If True, return unit-normalized direction

    Returns:
        Steering direction, shape (d_hidden,)
    """
    weights = probe_result.weights

    if high_bin == -1:
        high_bin = len(weights) - 1

    # Direction from low to high
    direction = weights[high_bin] - weights[low_bin]

    if normalize:
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm

    return direction
