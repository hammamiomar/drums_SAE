# %% imports
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from collections import Counter

# set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

# %% setup paths
data_dir = Path("../data/one_shot_percussive_sounds")
analysis_dir = Path("../data/analysis")

# get all wav files
wav_files = sorted(data_dir.rglob("*.wav"))
print(f"found {len(wav_files)} audio files")

# get all json files
json_files = sorted(analysis_dir.rglob("*.json"))
print(f"found {len(json_files)} analysis files")

# %% load all durations from analysis jsons
print("loading durations from analysis files...")

durations = []
metadata_list = []

for json_path in json_files:
    with open(json_path) as f:
        data = json.load(f)

    # extract key info
    duration = data.get('duration', 0)
    durations.append(duration)

    # store full metadata for later
    metadata_list.append({
        'filename': json_path.stem,
        'duration': duration,
        'samplerate': data.get('samplerate', 16000),
        'channels': data.get('channels', 1),
        # acoustic features (useful for probe training later!)
        'brightness': data.get('brightness', 0),
        'hardness': data.get('hardness', 0),
        'depth': data.get('depth', 0),
        'roughness': data.get('roughness', 0),
        'warmth': data.get('warmth', 0),
        'sharpness': data.get('sharpness', 0),
        'boominess': data.get('boominess', 0),
        'loudness': data.get('loudness', 0),
        'temporal_centroid': data.get('temporal_centroid', 0),
        'log_attack_time': data.get('log_attack_time', 0),
    })

durations = np.array(durations)
print(f"loaded {len(durations)} duration values")

# %% analyze duration distribution
print("\n=== duration statistics ===")
print(f"mean: {durations.mean():.3f}s")
print(f"median: {np.median(durations):.3f}s")
print(f"std: {durations.std():.3f}s")
print(f"min: {durations.min():.3f}s")
print(f"max: {durations.max():.3f}s")

# percentiles
percentiles = [50, 75, 90, 95, 99]
print("\npercentiles:")
for p in percentiles:
    val = np.percentile(durations, p)
    print(f"  {p}th: {val:.3f}s")

# how many samples fit in different lengths?
for target_len in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    pct = (durations <= target_len).mean() * 100
    print(f"{target_len}s captures {pct:.1f}% of samples")

# %% visualize duration distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# histogram
axes[0].hist(durations, bins=100, alpha=0.7, edgecolor='black')
axes[0].axvline(1.0, color='red', linestyle='--', label='1.0s (proposed)')
axes[0].axvline(np.median(durations), color='green', linestyle='--', label=f'median ({np.median(durations):.2f}s)')
axes[0].set_xlabel('duration (seconds)')
axes[0].set_ylabel('count')
axes[0].set_title('drum sample duration distribution')
axes[0].legend()
axes[0].grid(alpha=0.3)

# cumulative distribution
sorted_durations = np.sort(durations)
cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100
axes[1].plot(sorted_durations, cumulative, linewidth=2)
axes[1].axvline(1.0, color='red', linestyle='--', label='1.0s')
axes[1].axhline(95, color='orange', linestyle=':', alpha=0.5, label='95% coverage')
axes[1].set_xlabel('duration (seconds)')
axes[1].set_ylabel('cumulative % of samples')
axes[1].set_title('cumulative duration distribution')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% load and visualize a few random samples
n_samples = 4
sample_indices = np.random.choice(len(wav_files), n_samples, replace=False)

fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3 * n_samples))

for idx, sample_idx in enumerate(sample_indices):
    wav_path = wav_files[sample_idx]

    # load audio
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(audio) / sr

    # waveform
    times = np.arange(len(audio)) / sr
    axes[idx, 0].plot(times, audio, linewidth=0.5)
    axes[idx, 0].set_xlabel('time (s)')
    axes[idx, 0].set_ylabel('amplitude')
    axes[idx, 0].set_title(f'{wav_path.name} | duration: {duration:.2f}s | sr: {sr}Hz')
    axes[idx, 0].grid(alpha=0.3)

    # spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=axes[idx, 1])
    axes[idx, 1].set_title(f'spectrogram - {wav_path.name}')
    fig.colorbar(img, ax=axes[idx, 1], format='%+2.0f dB')

plt.tight_layout()
plt.show()

# %% explore acoustic features from pre-computed analysis
# this is gold for phase 2 probe training!

feature_names = ['brightness', 'hardness', 'depth', 'roughness',
                 'warmth', 'sharpness', 'boominess', 'loudness',
                 'temporal_centroid', 'log_attack_time']

# extract features into arrays
features_dict = {name: [] for name in feature_names}

for meta in metadata_list:
    for name in feature_names:
        features_dict[name].append(meta[name])

# convert to numpy
for name in feature_names:
    features_dict[name] = np.array(features_dict[name])

# plot distributions
fig, axes = plt.subplots(2, 5, figsize=(16, 8))
axes = axes.flatten()

for idx, name in enumerate(feature_names):
    values = features_dict[name]

    axes[idx].hist(values, bins=50, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{name}')
    axes[idx].set_xlabel('value')
    axes[idx].set_ylabel('count')
    axes[idx].grid(alpha=0.3)

    # add stats as text
    mean_val = values.mean()
    std_val = values.std()
    axes[idx].text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                   transform=axes[idx].transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)

plt.suptitle('pre-computed acoustic features distribution', fontsize=14, y=1.00)
plt.tight_layout()
plt.show()

# %% correlation between acoustic features
# understanding feature relationships helps with probe training

import pandas as pd

# create dataframe
df = pd.DataFrame(features_dict)

# compute correlation matrix
corr_matrix = df.corr()

# plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1)
plt.title('acoustic feature correlation matrix')
plt.tight_layout()
plt.show()

# %% decision: recommended audio length
print("\n=== RECOMMENDATION ===")
print(f"based on the duration analysis:")
print(f"  - 95th percentile: {np.percentile(durations, 95):.3f}s")
print(f"  - 1.0s captures {(durations <= 1.0).mean() * 100:.1f}% of samples")
print(f"  - 1.5s captures {(durations <= 1.5).mean() * 100:.1f}% of samples")
print(f"  - 2.0s captures {(durations <= 2.0).mean() * 100:.1f}% of samples")
print()
print("RECOMMENDED: Use 1.0s if >80% coverage is acceptable")
print("             Use 1.5s for >90% coverage with minimal padding overhead")
print()

# %% check sample rate consistency
sample_rates = [meta['samplerate'] for meta in metadata_list]
sr_counter = Counter(sample_rates)
print("\nsample rate distribution:")
for sr, count in sr_counter.most_common():
    print(f"  {sr}Hz: {count} files ({count/len(sample_rates)*100:.1f}%)")

# %% summary statistics for the paper
print("\n=== DATASET SUMMARY ===")
print(f"total samples: {len(durations)}")
print(f"duration range: {durations.min():.3f}s - {durations.max():.3f}s")
print(f"mean duration: {durations.mean():.3f}s")
print(f"median duration: {np.median(durations):.3f}s")
print(f"primary sample rate: {sr_counter.most_common(1)[0][0]}Hz")
print(f"acoustic features available: {', '.join(feature_names)}")
print("\nready for latent extraction!")
