# Drums SAE

Training Sparse Autoencoders on audio latent spaces for interpretable drum sound generation.

This is a work in progress. Personal project following the methodology from the Smule Labs paper.

## What This Is

An implementation of SAEs on the latent space of Stable Audio Open's VAE, applied to drum sounds. The idea is to decompose dense latent representations into sparse, interpretable features that could eventually be used for steering audio generation.

The SAE training works. Steering is still being figured out.

## Based On

**Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders**
Parker et al., Smule Labs, 2024
[arXiv:2510.23802](https://arxiv.org/abs/2510.23802)

The paper demonstrates that SAEs can learn interpretable audio features from music latent spaces.

## Pipeline

```
Audio -> Stable Audio Open VAE -> Latents (64-dim x T) -> SAE -> Sparse Features
```

1. Extract per-timestep acoustic features from audio (spectral centroid, RMS, etc.)
2. Encode audio through the Stable Audio Open VAE
3. Train SAE on the resulting latents
4. Train linear probes to evaluate feature interpretability

## Project Structure

```
src/drums_SAE/
  data/         # Audio preprocessing, feature extraction
  sae/          # SAE model (TopK activation, RMSNorm)
  training/     # Training loop, dataloaders
  eval/         # Linear probes for evaluation
  steering/     # WIP - probe-based steering

scripts/
  01_extract_features.py
  02_encode_latents.py
  03_train_sae.py
  04_evaluate.py

experiments/    # Trained model checkpoints
demo/           # Gradio demo
```

## Requirements

- Python 3.12+
- PyTorch
- librosa, torchaudio
- Stable Audio Open (included as submodule)

Uses `uv` for package management.

On macOS with custom ffmpeg:
```bash
DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib uv run python ...
```

## Usage

```bash
# Extract features
python scripts/01_extract_features.py \
    --input_dir data/GT/one_shot_percussive_sounds \
    --output_path data/features_v2.csv

# Encode latents
python scripts/02_encode_latents.py \
    --input_dir data/GT/one_shot_percussive_sounds \
    --output_path data/latents_v2.npz

# Train SAE
python scripts/03_train_sae.py \
    --output_dir experiments/v2_main

# Evaluate with linear probes
python scripts/04_evaluate.py \
    --checkpoint experiments/v2_main/checkpoints/sae_step_100000.pt
```

## SAE Architecture

- TopK activation (default k=32)
- RMSNorm after encoder for steering stability
- AuxK loss to revive dead features
- 64x expansion factor (64 -> 4096 features)

Based on Anthropic's "Scaling Monosemanticity" with the RMSNorm addition from the Smule paper.

## Status

- [x] Feature extraction pipeline
- [x] VAE encoding with alignment
- [x] SAE training with TopK
- [x] Multiple trained models
- [ ] Linear probe evaluation (in progress)
- [ ] Probe-based steering
- [ ] Steering verification

## License

MIT

## Acknowledgments

- [Smule Labs](https://www.smule.com/) for the original SAE audio paper
- [Stability AI](https://stability.ai/) for Stable Audio Open
- [Anthropic](https://www.anthropic.com/) for SAE training practices from their interpretability research
