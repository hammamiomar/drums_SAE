# %% [markdown]
# # SAE Evaluation & Feature Analysis
#
# **Phase 2A:** Evaluation — Did training work?
# **Phase 2B:** Feature Analysis — What did it learn?
# **Phase 2C:** Steering Capability — Can we control generation?

# %% Imports and Config
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

try:
    # If running as regular .py script, __file__ is defined
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drums_SAE.sae.model import AudioSae
from drums_SAE.training.data import LatentDataset

sns.set_theme(style="whitegrid")
Path("plots").mkdir(exist_ok=True)


@dataclass
class Config:
    checkpoint_path: str = str(PROJECT_ROOT / "checkpoints/sae_step_50000.pt")
    data_path: str = str(PROJECT_ROOT / "data/drums_encoded.npz")
    metadata_path: str = str(PROJECT_ROOT / "data/drums_encoded_metadata.csv")
    n_timesteps: int = 32  # Latent vectors per audio file
    label_columns: tuple = (
        "brightness",
        "boominess",
        "warmth",
        "hardness",
        "depth",
        "roughness",
        "sharpness",
        "loudness",
        "reverb",
    )


config = Config()
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {DEVICE}")


# %% Load Model and Data
def load_checkpoint(path: str, device: str) -> tuple[AudioSae, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = AudioSae(
        d_input=cfg["d_input"],
        expansion_factor=cfg["expansion_factor"],
        topk=cfg["topk"],
        topk_aux=cfg["topk_aux"],
        dead_threshold=cfg["dead_threshold"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


model, train_cfg = load_checkpoint(config.checkpoint_path, DEVICE)
dataset = LatentDataset(config.data_path, normalize=True)
metadata = pd.read_csv(config.metadata_path)

print(f"Model: {model.d_input}→{model.d_hidden} (TopK={model.topk})")
print(f"Dataset: {len(dataset):,} latent vectors")
print(f"Audio samples: {len(dataset) // config.n_timesteps:,}")


# %% Collect SAE Outputs
@torch.no_grad()
def collect_outputs(model, dataset, device, batch_size=1024):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    x_all, x_hat_all, h_all, f_all = [], [], [], []

    for batch in tqdm(loader, desc="Forward pass"):
        out = model(batch.to(device), return_aux=False)
        x_all.append(batch)
        x_hat_all.append(out["x_hat"].cpu())
        h_all.append(out["h"].cpu())
        f_all.append(out["f"].cpu())

    return {
        "x": torch.cat(x_all),
        "x_hat": torch.cat(x_hat_all),
        "h": torch.cat(h_all),
        "f": torch.cat(f_all),
    }


out = collect_outputs(model, dataset, DEVICE)
x, x_hat, h, f = out["x"], out["x_hat"], out["h"], out["f"]

# %% [markdown]
# ## Phase 2A: Evaluation Metrics

# %% Core Metrics
explained_var = 1 - (x - x_hat).pow(2).sum(dim=-1) / (
    (x - x.mean(dim=-1, keepdim=True)).pow(2).sum(dim=-1) + 1e-8
)
mse = (x - x_hat).pow(2).mean(dim=-1)
l0 = (h > 0).float().sum(dim=-1)
feature_usage = (h > 0).float().sum(dim=0)
dead_mask = model.steps_since_fired.cpu() > model.dead_threshold

print("=" * 60)
print("PHASE 2A: EVALUATION")
print("=" * 60)
print(f"\nReconstruction:  R² = {explained_var.mean():.4f} ± {explained_var.std():.4f}")
print(f"                 MSE = {mse.mean():.6f}")
print(f"Sparsity:        L0 = {l0.mean():.1f} (target: {model.topk})")
print(f"                 {(l0 == model.topk).float().mean():.1%} exact")
print(
    f"Features:        {dead_mask.sum()}/{model.d_hidden} dead ({dead_mask.float().mean():.1%})"
)

# %% Phase 2A Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Reconstruction scatter
ax = axes[0, 0]
idx = np.random.choice(len(x), 5000, replace=False)
ax.scatter(x[idx].flatten(), x_hat[idx].flatten(), alpha=0.05, s=1)
ax.plot([-4, 4], [-4, 4], "r--", lw=1)
ax.set(xlabel="Original", ylabel="Reconstructed", title="Reconstruction Quality")

# R² distribution
ax = axes[0, 1]
ax.hist(explained_var.numpy(), bins=50, alpha=0.7, color="forestgreen")
ax.axvline(
    explained_var.mean(),
    color="red",
    ls="--",
    label=f"Mean: {explained_var.mean():.3f}",
)
ax.set(xlabel="Explained Variance (R²)", title="R² Distribution")
ax.legend()

# L0 distribution
ax = axes[1, 0]
l0_vals, l0_counts = np.unique(l0.numpy(), return_counts=True)
ax.bar(l0_vals, l0_counts, color="steelblue")
ax.axvline(model.topk, color="red", ls="--", label=f"Target: {model.topk}")
ax.set(
    xlabel="L0",
    ylabel="Count",
    title="L0 Distribution",
    xlim=(model.topk - 5, model.topk + 2),
)
ax.legend()

# Feature usage
ax = axes[1, 1]
sorted_usage = torch.sort(feature_usage, descending=True).values.numpy()
ax.bar(range(len(sorted_usage)), sorted_usage, width=1, color="steelblue", alpha=0.7)
ax.axhline(len(dataset) * 0.5, color="red", ls="--", label="50% of samples")
ax.set(xlabel="Feature (sorted)", ylabel="Usage", title="Feature Usage")
ax.legend()

plt.tight_layout()
plt.savefig("plots/01_phase2a_metrics.png")
plt.show()

# %% [markdown]
# ## Phase 2B: Feature Analysis


# %% Fast Spearman Correlation (vectorized)
def spearman_correlation(X: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
    """
    Compute Spearman correlations between columns of X and Y.
    X: (N, d1), Y: (N, d2) -> returns (d1, d2) correlation matrix.
    """

    def rank(t):
        return t.argsort(dim=0).argsort(dim=0).float()

    X_rank = rank(X)
    Y_rank = rank(Y)

    X_centered = X_rank - X_rank.mean(dim=0)
    Y_centered = Y_rank - Y_rank.mean(dim=0)

    X_norm = X_centered / (X_centered.std(dim=0) + 1e-10)
    Y_norm = Y_centered / (Y_centered.std(dim=0) + 1e-10)

    return (X_norm.T @ Y_norm / len(X)).numpy()


# %% Aggregate to Audio Sample Level
n_audio = len(h) // config.n_timesteps
h_agg = h.reshape(n_audio, config.n_timesteps, -1).mean(dim=1)
meta_agg = metadata.iloc[:: config.n_timesteps].reset_index(drop=True)

labels_tensor = torch.tensor(
    meta_agg[list(config.label_columns)].values, dtype=torch.float32
)

print(f"Aggregated: {n_audio:,} audio samples")

# %% Compute Correlations (both timestep-level and aggregated)
print("Computing correlations...")
corr_timestep = spearman_correlation(
    h, torch.tensor(metadata[list(config.label_columns)].values, dtype=torch.float32)
)
corr_agg = spearman_correlation(h_agg, labels_tensor)

print(f"\nCorrelation improvement after aggregation:")
print(f"{'Label':<12} {'Timestep |ρ|':>12} {'Aggregated |ρ|':>14} {'Gain':>8}")
print("-" * 50)
for i, label in enumerate(config.label_columns):
    before = np.abs(corr_timestep[:, i]).max()
    after = np.abs(corr_agg[:, i]).max()
    print(f"{label:<12} {before:>12.3f} {after:>14.3f} {after / before:>7.1f}x")

# %% Feature-Label Correlation Heatmap
top_feat = np.argsort(np.abs(corr_agg).max(axis=1))[-30:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
for ax, (corr, title) in zip(
    axes,
    [
        (corr_timestep, "Per-Timestep (Misaligned)"),
        (corr_agg, "Per-Audio-Sample (Aligned)"),
    ],
):
    sns.heatmap(
        corr[top_feat],
        xticklabels=config.label_columns,
        yticklabels=[f"F{i}" for i in top_feat],
        cmap="RdBu_r",
        center=0,
        vmin=-0.4,
        vmax=0.4,
        annot=True,
        fmt=".2f",
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(title)

plt.suptitle("Feature-Label Correlations (Top 30 Features)", fontsize=12)
plt.tight_layout()
plt.savefig("plots/02_correlations.png")
plt.show()


# %% Max-Activating Samples
def get_top_activating(h, k=10):
    """Return (d_hidden, k) indices and values of top-k activations per feature."""
    vals, idx = torch.topk(h.T, k=k, dim=1)
    return idx, vals


top_idx, top_vals = get_top_activating(h, k=10)


def show_feature(feat_id):
    """Display top-activating samples for a feature."""
    samples = top_idx[feat_id].numpy()
    acts = top_vals[feat_id].numpy()
    df = metadata.iloc[samples][["id"] + list(config.label_columns)].copy()
    df.insert(0, "activation", acts)
    df.insert(1, "audio_file", samples // config.n_timesteps)
    df.insert(2, "timestep", samples % config.n_timesteps)
    return df


# Example
print("\nFeature 216 (top activating samples):")
print(show_feature(216).head(10).to_string(index=False))

# %% [markdown]
# ## Phase 2C: Steering Capability

# %% Decoder Analysis
W_dec = model.decoder.weight.data.cpu()
dec_norms = W_dec.norm(dim=0)
is_unit_norm = dec_norms.std() < 0.01

print("=" * 60)
print("PHASE 2C: STEERING CAPABILITY")
print("=" * 60)
print(f"\nDecoder norms: {dec_norms.mean():.4f} ± {dec_norms.std():.4f}")
print(f"Unit normalized: {'Yes' if is_unit_norm else 'No'}")

# %% Temporal Analysis
h_temporal = h.reshape(n_audio, config.n_timesteps, -1)
early = h_temporal[:, :8, :].mean(dim=(0, 1))
late = h_temporal[:, 24:, :].mean(dim=(0, 1))
attack_ratio = early / (late + 1e-8)

attack_features = torch.argsort(attack_ratio, descending=True)[:10]
sustain_features = torch.argsort(attack_ratio)[:10]

print(f"\nAttack features (fire early): {attack_features.tolist()}")
print(f"Sustain features (fire late): {sustain_features.tolist()}")

# %% Temporal Profile Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
time_sec = np.linspace(0, 1.5, config.n_timesteps)
temporal_mean = h_temporal.mean(dim=0)

for ax, feats, title in [
    (axes[0], attack_features[:5], "Attack Features"),
    (axes[1], sustain_features[:5], "Sustain Features"),
]:
    for f in feats:
        ax.plot(time_sec, temporal_mean[:, f].numpy(), label=f"F{f}", alpha=0.8)
    ax.set(xlabel="Time (s)", ylabel="Mean Activation", title=title)
    ax.legend()

plt.tight_layout()
plt.savefig("plots/03_temporal.png")
plt.show()


# %% Control Vectors
def build_control_vector(label, corr, W_dec, top_k=20):
    """Build steering direction from top-correlated features."""
    label_idx = config.label_columns.index(label)
    c = corr[:, label_idx]
    top = np.argsort(np.abs(c))[-top_k:]
    weights = torch.tensor(c[top], dtype=torch.float32)
    vec = W_dec[:, top] @ weights
    return vec / vec.norm(), c[top]


control_vecs = {}
print(f"\n{'Label':<12} {'Max |ρ|':>8} {'# |ρ|>0.1':>10}")
print("-" * 35)
for label in config.label_columns:
    cv, top_corrs = build_control_vector(label, corr_agg, W_dec)
    control_vecs[label] = cv
    max_c = np.abs(corr_agg[:, config.label_columns.index(label)]).max()
    n_strong = (np.abs(corr_agg[:, config.label_columns.index(label)]) > 0.1).sum()
    print(f"{label:<12} {max_c:>8.3f} {n_strong:>10}")

# %% Control Vector Orthogonality
cv_matrix = torch.stack(list(control_vecs.values()))
cv_cos = (cv_matrix @ cv_matrix.T).numpy()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cv_cos,
    xticklabels=config.label_columns,
    yticklabels=config.label_columns,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    ax=ax,
)
ax.set_title("Control Vector Similarity")
plt.tight_layout()
plt.savefig("plots/04_control_vectors.png")
plt.show()

# %% Steering Viability Assessment
print("\n" + "=" * 60)
print("STEERING VIABILITY ASSESSMENT")
print("=" * 60)


def assess(label):
    c = corr_agg[:, config.label_columns.index(label)]
    max_abs = np.abs(c).max()
    if max_abs > 0.3:
        return "GOOD"
    elif max_abs > 0.15:
        return "MODERATE"
    else:
        return "WEAK"


print(f"\n{'Label':<12} {'Max |ρ|':>8} {'Viability':>12}")
print("-" * 35)
results = {}
for label in config.label_columns:
    c = corr_agg[:, config.label_columns.index(label)]
    results[label] = assess(label)
    print(f"{label:<12} {np.abs(c).max():>8.3f} {results[label]:>12}")

good = [l for l, v in results.items() if v == "GOOD"]
moderate = [l for l, v in results.items() if v == "MODERATE"]
weak = [l for l, v in results.items() if v == "WEAK"]

print(f"\n✅ Ready for steering: {good + moderate if good + moderate else 'None'}")
print(f"⚠️  Weak signal: {weak if weak else 'None'}")


# %% Feature Inspector
def inspect_feature(feat_id):
    """Full analysis of a single feature."""
    h_feat = h[:, feat_id]
    active = h_feat > 0

    print(f"\n{'=' * 50}")
    print(f"FEATURE {feat_id}")
    print(f"{'=' * 50}")
    print(f"Usage: {active.sum():,} / {len(h):,} ({active.float().mean():.2%})")
    print(f"Mean activation (when active): {h_feat[active].mean():.3f}")
    print(f"Max activation: {h_feat.max():.3f}")

    print(f"\nTop correlations:")
    for i, label in enumerate(config.label_columns):
        print(f"  {label:<12}: {corr_agg[feat_id, i]:+.3f}")

    print(f"\nTop-5 activating samples:")
    print(show_feature(feat_id).head(5).to_string(index=False))

    print(f"\nTemporal bias (early/late): {attack_ratio[feat_id]:.2f}x")


# Inspect most interesting features
for label in ["brightness", "loudness", "depth"]:
    best = np.argmax(np.abs(corr_agg[:, config.label_columns.index(label)]))
    print(f"\n>>> Best feature for '{label}': F{best}")
    inspect_feature(best)

# %% Summary Export
summary = pd.DataFrame(
    {
        "feature": range(model.d_hidden),
        "usage": feature_usage.numpy(),
        "usage_pct": (feature_usage / len(dataset) * 100).numpy(),
        "is_dead": dead_mask.numpy(),
        "attack_ratio": attack_ratio.numpy(),
        **{f"corr_{l}": corr_agg[:, i] for i, l in enumerate(config.label_columns)},
    }
)
summary.to_csv("feature_summary.csv", index=False)
print("\n✅ Saved feature_summary.csv")

# %% Final Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Phase 2A (Evaluation):
  • R² = {explained_var.mean():.3f} (target: >0.90) {"✅" if explained_var.mean() > 0.9 else "❌"}
  • L0 = {l0.mean():.1f} (target: {model.topk}) {"✅" if abs(l0.mean() - model.topk) < 1 else "❌"}
  • Dead features: {dead_mask.sum()}/{model.d_hidden} {"✅" if dead_mask.float().mean() < 0.05 else "❌"}

Phase 2B (Feature Analysis):
  • Correlations improved {(np.abs(corr_agg).max() / np.abs(corr_timestep).max()):.1f}x after aggregation
  • Max |ρ| = {np.abs(corr_agg).max():.3f}
  • Temporal features identified (attack vs sustain)

Phase 2C (Steering):
  • Decoder {"✅ unit normalized" if is_unit_norm else "⚠️ not unit normalized"}
  • Viable labels: {[l for l, v in results.items() if v != "WEAK"]}
  • Recommendation: {"Test steering!" if good else "Consider per-timestep labels"}
""")

# %%
