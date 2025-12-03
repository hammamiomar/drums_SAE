# %% [markdown]
# # Steering Demo
#
# Apply control vectors to drum latents and decode to audio.
#
# Prerequisites:
# - Trained SAE checkpoint
# - feature_summary.csv from evaluation notebook
# - Stable Audio Open VAE (for decoding)

# %% Imports
import os
import sys
from pathlib import Path

os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/usr/local/ffmpeg7/lib"

import numpy as np
import torch
import torchaudio
from IPython.display import Audio, display

try:
    # If running as regular .py script, __file__ is defined
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drums_SAE.sae.model import AudioSae
from drums_SAE.steering.steer import ControlVectors, create_steering_grid, steer_latent
from drums_SAE.training.data import LatentDataset

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {DEVICE}")

# %% Config
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/sae_step_50000.pt"
FEATURE_SUMMARY_PATH = PROJECT_ROOT / "notebooks/feature_summary.csv"
LATENT_DATA_PATH = PROJECT_ROOT / "data/drums_encoded.npz"
OUTPUT_DIR = PROJECT_ROOT / "outputs/steering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 44100  # Stable Audio Open sample rate
N_TIMESTEPS = 32

# %% Load SAE and Control Vectors
print("Loading SAE and control vectors...")
cv = ControlVectors.from_checkpoint(
    str(CHECKPOINT_PATH),
    str(FEATURE_SUMMARY_PATH),
    top_k=20,
    device=DEVICE,
)
print(f"Control vectors available for: {list(cv.keys())}")

# Also load the full SAE for encoding/decoding through SAE
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
cfg = checkpoint["config"]
sae = AudioSae(
    d_input=cfg["d_input"],
    expansion_factor=cfg["expansion_factor"],
    topk=cfg["topk"],
    topk_aux=cfg["topk_aux"],
    dead_threshold=cfg["dead_threshold"],
).to(DEVICE)
sae.load_state_dict(checkpoint["model_state_dict"])
sae.eval()

# %% Load latent data (normalized)
dataset = LatentDataset(str(LATENT_DATA_PATH), normalize=True)

# Load normalization stats for denormalization
latent_data = np.load(LATENT_DATA_PATH)
latent_mean = torch.tensor(latent_data["mean"], dtype=torch.float32)
latent_std = torch.tensor(latent_data["std"], dtype=torch.float32)

print(f"Dataset: {len(dataset):,} latent vectors")
print(f"Audio samples: {len(dataset) // N_TIMESTEPS:,}")

# %% Load VAE Decoder
print("\nLoading Stable Audio Open VAE...")

try:
    from einops import rearrange
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.models.utils import load_ckpt_state_dict

    # Load the pretrained model (this will download if needed)
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    vae = model.pretransform.model  # The autoencoder
    vae = vae.to(DEVICE)
    vae.eval()

    VAE_AVAILABLE = True
    print("✓ VAE loaded successfully")

except ImportError as e:
    print(f"⚠ Could not load VAE: {e}")
    print("Install with: pip install stable-audio-tools")
    print("Steering will work but you won't be able to decode to audio.")
    VAE_AVAILABLE = False

except Exception as e:
    print(f"⚠ Error loading VAE: {e}")
    VAE_AVAILABLE = False


# %% Decoding functions
def denormalize_latent(z_norm: torch.Tensor) -> torch.Tensor:
    """Convert normalized latent back to original scale."""
    return z_norm * latent_std + latent_mean


def decode_to_audio(z_norm: torch.Tensor) -> torch.Tensor:
    """
    Decode normalized latent(s) to audio.

    Args:
        z_norm: Normalized latent, shape (32, 64) for one audio sample
                or (batch, 32, 64) for multiple

    Returns:
        Audio waveform, shape (samples,) or (batch, samples)
    """
    if not VAE_AVAILABLE:
        raise RuntimeError("VAE not available. Cannot decode to audio.")

    # Ensure 3D: (batch, timesteps, channels)
    if z_norm.dim() == 2:
        z_norm = z_norm.unsqueeze(0)

    # Denormalize
    z = denormalize_latent(z_norm.cpu())

    # VAE expects (batch, channels, timesteps)
    z = rearrange(z, "b t c -> b c t").to(DEVICE)

    with torch.no_grad():
        audio = vae.decode(z)

    # Output is (batch, 1, samples) or (batch, 2, samples)
    audio = audio.squeeze(1)  # Remove channel dim if mono

    if audio.shape[0] == 1:
        audio = audio.squeeze(0)

    return audio.cpu()


def save_audio(audio: torch.Tensor, path: str, sample_rate: int = SAMPLE_RATE):
    """Save audio tensor to file."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(path, audio, sample_rate)
    print(f"Saved: {path}")


# %% Helper to get one audio sample's latents
def get_audio_latents(audio_idx: int) -> torch.Tensor:
    """Get all 32 latent vectors for one audio sample."""
    start = audio_idx * N_TIMESTEPS
    end = start + N_TIMESTEPS
    return torch.stack([dataset[i] for i in range(start, end)])  # (32, 64)


# %% [markdown]
# ## Steering Experiments
#
# Let's steer some drums!

# %% Pick a sample to steer
AUDIO_IDX = 42  # Change this to try different drums

z_original = get_audio_latents(AUDIO_IDX)
print(f"Original latent shape: {z_original.shape}")

# %% Steer and compare
LABEL = "brightness"
ALPHAS = [-1.0, -0.5, 0, 0.5, 1.0]

direction = cv[LABEL].unsqueeze(0)  # (1, 64) for broadcasting

steered_latents = {}
for alpha in ALPHAS:
    key = f"{LABEL}_{alpha:+.1f}"
    # Steer each timestep
    z_steered = steer_latent(z_original, direction, alpha)
    steered_latents[key] = z_steered
    print(f"{key}: mean diff = {(z_steered - z_original).abs().mean():.4f}")

# %% Decode to audio (if VAE available)
if VAE_AVAILABLE:
    print("\nDecoding steered latents to audio...")

    for key, z in steered_latents.items():
        audio = decode_to_audio(z)
        save_audio(audio, str(OUTPUT_DIR / f"audio_{AUDIO_IDX}_{key}.wav"))

    # Also save original
    audio_orig = decode_to_audio(z_original)
    save_audio(audio_orig, str(OUTPUT_DIR / f"audio_{AUDIO_IDX}_original.wav"))

    print(f"\n✓ Audio files saved to {OUTPUT_DIR}/")

# %% Listen in notebook (if in Jupyter)
if VAE_AVAILABLE:
    print(f"\n🎧 Listening comparison for '{LABEL}':\n")

    for alpha in ALPHAS:
        if alpha == 0:
            label = "original"
            z = z_original
        else:
            label = f"{LABEL} α={alpha:+.1f}"
            z = steered_latents[f"{LABEL}_{alpha:+.1f}"]

        audio = decode_to_audio(z)
        print(f"{label}:")
        display(Audio(audio.numpy(), rate=SAMPLE_RATE))

# %% Multi-property steering grid
if VAE_AVAILABLE:
    print("\n" + "=" * 50)
    print("MULTI-PROPERTY STEERING GRID")
    print("=" * 50)

    LABELS_TO_TEST = ["brightness", "depth", "loudness", "roughness"]
    ALPHA = 0.7  # Steering strength

    for label in LABELS_TO_TEST:
        if label not in cv:
            continue

        direction = cv[label].unsqueeze(0)

        z_plus = steer_latent(z_original, direction, alpha=ALPHA)
        z_minus = steer_latent(z_original, direction, alpha=-ALPHA)

        audio_plus = decode_to_audio(z_plus)
        audio_minus = decode_to_audio(z_minus)

        save_audio(audio_plus, str(OUTPUT_DIR / f"audio_{AUDIO_IDX}_{label}_plus.wav"))
        save_audio(
            audio_minus, str(OUTPUT_DIR / f"audio_{AUDIO_IDX}_{label}_minus.wav")
        )

        print(f"\n{label.upper()}:")
        print(f"  Less {label} (α=-{ALPHA}):")
        display(Audio(audio_minus.numpy(), rate=SAMPLE_RATE))
        print(f"  More {label} (α=+{ALPHA}):")
        display(Audio(audio_plus.numpy(), rate=SAMPLE_RATE))

# %% [markdown]
# ## Steering Through SAE
#
# Alternative: Steer in SAE feature space instead of latent space.
# This is more "principled" since we're manipulating the interpretable features directly.


# %% Steer in SAE feature space
def steer_through_sae(
    z: torch.Tensor,
    sae: AudioSae,
    feature_idx: int,
    delta: float,
) -> torch.Tensor:
    """
    Steer by directly modifying a SAE feature's activation.

    Args:
        z: Latent(s), shape (timesteps, d_input)
        sae: Trained SAE model
        feature_idx: Which feature to modify
        delta: Amount to add to the feature activation

    Returns:
        Steered latent(s)
    """
    with torch.no_grad():
        # Encode to SAE features
        z_device = z.to(DEVICE)
        out = sae.encode(z_device)
        h = out["h"]  # (timesteps, d_hidden)

        # Modify specific feature
        h_steered = h.clone()
        h_steered[:, feature_idx] = h_steered[:, feature_idx] + delta

        # Decode back
        # Note: We use h directly (pre-RMSNorm) for decoding
        # Need to apply RMSNorm first
        h_norm = h_steered.pow(2).mean(dim=-1, keepdim=True).sqrt()
        f_steered = h_steered / (h_norm + 1e-8)

        z_steered = sae.decode(f_steered)

    return z_steered.cpu()


# %% Example: Steer using a specific feature
# Find feature most correlated with brightness
import pandas as pd

feature_summary = pd.read_csv(FEATURE_SUMMARY_PATH)
brightness_corrs = feature_summary["corr_brightness"].values
best_brightness_feature = np.argmax(brightness_corrs)
print(
    f"Best brightness feature: F{best_brightness_feature} (ρ = {brightness_corrs[best_brightness_feature]:.3f})"
)

# Steer using that feature
z_feat_plus = steer_through_sae(z_original, sae, best_brightness_feature, delta=2.0)
z_feat_minus = steer_through_sae(z_original, sae, best_brightness_feature, delta=-2.0)

if VAE_AVAILABLE:
    print("\nSteering via SAE feature manipulation:")

    audio_feat_plus = decode_to_audio(z_feat_plus)
    audio_feat_minus = decode_to_audio(z_feat_minus)

    save_audio(
        audio_feat_plus,
        str(
            OUTPUT_DIR / f"audio_{AUDIO_IDX}_feature{best_brightness_feature}_plus.wav"
        ),
    )
    save_audio(
        audio_feat_minus,
        str(
            OUTPUT_DIR / f"audio_{AUDIO_IDX}_feature{best_brightness_feature}_minus.wav"
        ),
    )

    print(f"\nF{best_brightness_feature} (brightness) manipulation:")
    print("  Decreased:")
    display(Audio(audio_feat_minus.numpy(), rate=SAMPLE_RATE))
    print("  Increased:")
    display(Audio(audio_feat_plus.numpy(), rate=SAMPLE_RATE))

# %% [markdown]
# ## Batch Generation
#
# Generate a batch of steered samples for evaluation.

# %% Generate comparison set
if VAE_AVAILABLE:
    print("\n" + "=" * 50)
    print("BATCH GENERATION")
    print("=" * 50)

    N_SAMPLES = 5
    LABELS = ["brightness", "depth", "roughness"]
    ALPHA = 0.6

    sample_indices = np.random.choice(
        len(dataset) // N_TIMESTEPS, N_SAMPLES, replace=False
    )

    batch_dir = OUTPUT_DIR / "batch"
    batch_dir.mkdir(exist_ok=True)

    for idx in sample_indices:
        z = get_audio_latents(idx)

        # Original
        audio = decode_to_audio(z)
        save_audio(audio, str(batch_dir / f"sample_{idx:04d}_original.wav"))

        # Steered versions
        for label in LABELS:
            if label not in cv:
                continue
            direction = cv[label].unsqueeze(0)

            z_plus = steer_latent(z, direction, ALPHA)
            z_minus = steer_latent(z, direction, -ALPHA)

            audio_plus = decode_to_audio(z_plus)
            audio_minus = decode_to_audio(z_minus)

            save_audio(
                audio_plus, str(batch_dir / f"sample_{idx:04d}_{label}_plus.wav")
            )
            save_audio(
                audio_minus, str(batch_dir / f"sample_{idx:04d}_{label}_minus.wav")
            )

    print(
        f"\n✓ Generated {N_SAMPLES * (1 + len(LABELS) * 2)} audio files in {batch_dir}/"
    )

# %% Summary
print("\n" + "=" * 50)
print("STEERING DEMO COMPLETE")
print("=" * 50)
print(f"""
Files generated in: {OUTPUT_DIR}/

Two steering methods demonstrated:
1. Control vector steering (add direction to latent)
2. SAE feature steering (modify feature activation directly)

Next steps:
- Listen to outputs and evaluate quality
- Tune alpha values for best results
- Build interactive demo (Gradio/Streamlit)
""")
