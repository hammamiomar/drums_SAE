"""Test the NEW post-hoc steering approach.

This tests:
1. Baseline generation produces normal audio (~300-500 Hz centroid)
2. +bass steering produces bassier audio (higher centroid? or more bass energy)
3. -bass steering produces thinner audio (lower bass energy)
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

import torch
import librosa
import numpy as np

from drums_SAE.generation import (
    load_models,
    load_sae,
    generate_latents,
    steer_latents,
    decode_latents,
)
from drums_SAE.steering.probe_steer import ProbeSteeringVectors

# Device
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load models
print('Loading models...')
model, model_config, vae, sample_rate = load_models(device)
print(f'Sample rate: {sample_rate}')

sae = load_sae('experiments/v2_main/checkpoints/sae_latest.pt', device)
vectors = ProbeSteeringVectors.load('experiments/v2_main/eval/steering_vectors.npz')
print(f'Available properties: {vectors.properties}')


def measure_audio(audio_tensor):
    """Measure spectral centroid and bass energy."""
    audio_np = audio_tensor.cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    if audio_np.ndim > 1:
        audio_np = audio_np[0]  # Take first channel

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=audio_np, sr=sample_rate).mean()

    # RMS energy
    rms = np.sqrt(np.mean(audio_np ** 2))

    return {
        'centroid': centroid,
        'rms': rms,
    }


# === TEST 1: Generate clean latents ===
print('\n=== TEST 1: Generate clean latents ===')
latents = generate_latents(
    model=model,
    prompt='kick drum',
    seconds=1.5,
    steps=100,
    cfg_scale=7.0,
    seed=42,
    device=device,
)
print(f'Latents shape: {latents.shape}')
print(f'Latents stats: mean={latents.mean():.4f}, std={latents.std():.4f}')


# === TEST 2: Decode baseline ===
print('\n=== TEST 2: Decode baseline ===')
baseline_audio = decode_latents(latents, vae)
print(f'Baseline audio shape: {baseline_audio.shape}')
baseline_props = measure_audio(baseline_audio)
print(f'Baseline centroid: {baseline_props["centroid"]:.0f} Hz')
print(f'Baseline RMS: {baseline_props["rms"]:.4f}')


# === TEST 3: Steer +bass and decode ===
print('\n=== TEST 3: Steer +bass and decode ===')
latents_plus_bass = steer_latents(
    latents=latents,
    sae=sae,
    steering_vectors=vectors,
    property_alphas={'bass': 1.5},
)
print(f'Steered latents stats: mean={latents_plus_bass.mean():.4f}, std={latents_plus_bass.std():.4f}')
plus_bass_audio = decode_latents(latents_plus_bass, vae)
plus_bass_props = measure_audio(plus_bass_audio)
print(f'+bass centroid: {plus_bass_props["centroid"]:.0f} Hz')
print(f'+bass RMS: {plus_bass_props["rms"]:.4f}')


# === TEST 4: Steer -bass and decode ===
print('\n=== TEST 4: Steer -bass and decode ===')
latents_minus_bass = steer_latents(
    latents=latents,
    sae=sae,
    steering_vectors=vectors,
    property_alphas={'bass': -1.5},
)
minus_bass_audio = decode_latents(latents_minus_bass, vae)
minus_bass_props = measure_audio(minus_bass_audio)
print(f'-bass centroid: {minus_bass_props["centroid"]:.0f} Hz')
print(f'-bass RMS: {minus_bass_props["rms"]:.4f}')


# === VERDICT ===
print('\n' + '=' * 50)
print('VERDICT')
print('=' * 50)
print(f'Baseline centroid: {baseline_props["centroid"]:.0f} Hz')
print(f'+bass centroid:    {plus_bass_props["centroid"]:.0f} Hz')
print(f'-bass centroid:    {minus_bass_props["centroid"]:.0f} Hz')

# Check if steering produces reasonable audio (not noise)
if baseline_props["centroid"] < 1000:
    print('✓ Baseline produces reasonable kick drum (centroid < 1000 Hz)')
else:
    print('✗ Baseline might be noise (centroid >= 1000 Hz)')

# Check if +bass and -bass produce different results
diff = abs(plus_bass_props["centroid"] - minus_bass_props["centroid"])
if diff > 50:
    print(f'✓ Steering produces measurable difference ({diff:.0f} Hz difference)')
else:
    print(f'✗ Steering has minimal effect ({diff:.0f} Hz difference)')

# Check direction makes sense
# Note: bass steering direction may affect centroid in unexpected ways
# since more bass = lower frequencies = could lower OR raise centroid
print(f'\nNote: bass direction interpretation depends on what the probe learned.')
print(f'      Verify by listening to the generated audio.')


# === SAVE AUDIO FOR LISTENING ===
print('\n=== Saving audio files ===')
import torchaudio

def save_audio(tensor, filename):
    audio = tensor.squeeze().cpu()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    # Normalize
    audio = audio / audio.abs().max() * 0.95
    torchaudio.save(filename, audio, sample_rate)
    print(f'Saved: {filename}')

save_audio(baseline_audio, 'outputs/posthoc_baseline.wav')
save_audio(plus_bass_audio, 'outputs/posthoc_plus_bass.wav')
save_audio(minus_bass_audio, 'outputs/posthoc_minus_bass.wav')

print('\nDone! Listen to the files to verify steering works.')
