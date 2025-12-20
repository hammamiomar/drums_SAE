"""Test steering with ORIGINAL code (no normalization)."""
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

import torch
import librosa
import numpy as np
from drums_SAE.diffusion import load_stable_audio, load_sae, generate_steered_audio
from drums_SAE.steering.probe_steer import ProbeSteeringVectors

device = 'mps'
print('Loading models...')
model, config = load_stable_audio(device)
sae = load_sae('experiments/v2_main/checkpoints/sae_latest.pt', device)
vectors = ProbeSteeringVectors.load('experiments/v2_main/eval/steering_vectors.npz')

def measure_audio(audio_tensor):
    """Measure spectral centroid."""
    audio_np = audio_tensor.cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=0)
    centroid = librosa.feature.spectral_centroid(y=audio_np, sr=44100).mean()
    return centroid

# Generate baseline
print('\nGenerating baseline...')
baseline = generate_steered_audio(
    prompt='kick drum',
    property_steering=None,
    steps=100,
    seed=42,
    device=device,
    model=model,
    model_config=config,
)
print(f'Baseline centroid: {measure_audio(baseline):.0f} Hz')

# Generate with steering - MIDDLE schedule
print('\nGenerating with bass=1.5, schedule=middle...')
steered_middle = generate_steered_audio(
    prompt='kick drum',
    property_steering={'bass': 1.5},
    apply_steering_at='middle',
    steps=100,
    seed=42,
    device=device,
    model=model,
    model_config=config,
    sae=sae,
    steering_vectors=vectors,
)
print(f'Steered (+bass) centroid: {measure_audio(steered_middle):.0f} Hz')

# Generate with steering - NEGATIVE direction (should make darker)
print('\nGenerating with bass=-1.5, schedule=middle (ORIGINAL CODE)...')
steered_minus = generate_steered_audio(
    prompt='kick drum',
    property_steering={'bass': -1.5},
    apply_steering_at='middle',
    steps=100,
    seed=42,
    device=device,
    model=model,
    model_config=config,
    sae=sae,
    steering_vectors=vectors,
)
print(f'Steered (-bass) centroid: {measure_audio(steered_minus):.0f} Hz')

# Compare to old files
print('\n=== Comparing to OLD working files ===')
old_baseline, _ = librosa.load('outputs/kick_comparison_baseline.wav', sr=None)
old_more, _ = librosa.load('outputs/kick_comparison_more.wav', sr=None)
print(f'Old baseline centroid: {librosa.feature.spectral_centroid(y=old_baseline, sr=44100).mean():.0f} Hz')
print(f'Old steered centroid: {librosa.feature.spectral_centroid(y=old_more, sr=44100).mean():.0f} Hz')

print('\n=== VERDICT ===')
baseline_c = measure_audio(baseline)
plus_c = measure_audio(steered_middle)
minus_c = measure_audio(steered_minus)
print(f'Baseline: {baseline_c:.0f} Hz')
print(f'+bass: {plus_c:.0f} Hz (should be > baseline if working, OLD was 2968 Hz)')
print(f'-bass: {minus_c:.0f} Hz (should be < baseline if working, OLD was 291 Hz)')
if minus_c < baseline_c:
    print('✓ Negative direction works! (darker than baseline)')
else:
    print('✗ Negative direction broken')
