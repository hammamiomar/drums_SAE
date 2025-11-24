# download at https://zenodo.org/records/3665275, data should have folders "analysis" and "one_shot_percussive_sounds"
# %%
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.utils import prepare_audio
from tqdm import tqdm


def create_dataset_manifest(root_path: str, output_path="drums_manifest.csv"):
    root = Path(root_path) / "GT"
    data_path = root / "one_shot_percussive_sounds"
    analysis_path = root / "analysis"

    # get all wav files
    wav_files = sorted(list(data_path.rglob("*.wav")))
    print(f"found {len(wav_files)} audio files")

    # get all json files
    json_files = sorted(analysis_path.rglob("*.json"))
    print(f"found {len(json_files)} analysis files")

    records = []

    for wav_file in wav_files:
        # Logic: data/1/1234.wav -> analysis/1/1234_analysis.json
        try:
            relative_path = wav_file.relative_to(data_path)
            folder = relative_path.parts[0]  # e.g., "1"
            stem = relative_path.stem  # e.g., "1234"

            json_file = analysis_path / folder / f"{stem}_analysis.json"

            meta = {}
            if json_file.exists():
                with open(json_file, "r") as f:
                    meta = json.load(f)

            # Build record
            records.append(
                {
                    "id": int(stem),
                    "path": str(wav_file),
                    # Audio Features from JSON
                    "brightness": meta.get("brightness", 0.0),
                    "boominess": meta.get("boominess", 0.0),
                    "warmth": meta.get("warmth", 0.0),
                    "hardness": meta.get("hardness", 0.0),
                    "depth": meta.get("depth", 0.0),
                    "roughness": meta.get("roughness", 0.0),
                    "sharpness": meta.get("sharpness", 0.0),
                    "loudness": meta.get("loudness", -60.0),
                    "reverb": 1 if meta.get("reverb") else 0,
                }
            )
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue

    df = pd.DataFrame(records)
    # Sort by ID for consistency
    if "id" in df.columns:
        df.sort_values("id", inplace=True)

    print(f"Created manifest with shape {df.shape}")
    df.to_csv(output_path, index=False)
    return df


def encode_audio(
    manifest_csv, save_path="processed_dataset.npz"
):  # Save to single file
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    df = pd.read_csv(manifest_csv)

    # Load Model
    print("Loading Model...")
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device).eval()
    vae = model.pretransform

    target_sr = model_config["sample_rate"]
    target_length = 65536  # around 1.5 secs, fits the downsampling --> length/2048 (2048 is the downsampling). so we guarantee 32 latent vecs per drum sample

    latents_accumulator = []
    metadata_accumulator = []

    print("Processing Audio...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                wav, sr = torchaudio.load(row["path"])

                input_tensor = prepare_audio(
                    wav,
                    in_sr=sr,
                    target_sr=target_sr,
                    target_length=target_length,
                    target_channels=vae.io_channels,
                    device=device,
                )

                # Encode
                # Output shape: [1, 64, 32] (Batch, Channels (latent dim), Time)
                # 64 channels: dense dimension, where VAE concepts are superimposed. 32 time steps: each step is approx 0.046 secs of audio
                encoded_latents = vae.encode(input_tensor)
                # encoder is a stack of 1d strided convolutions. so, it has a finite receptive field, essentially a feature extractor.
                # so a vector depends only on the local window of audio -- not the whole file itself.

                flat_z = (
                    rearrange(encoded_latents, "b c t -> (b t) c")
                    .cpu()
                    .numpy()  # (1,64,32) -> (32,64)
                )  # flatten, because the SAE trains on individual vectors. we are assuming temporal independence. we are looking for features in 64 dim vector space!

                latents_accumulator.append(flat_z)

                # Duplicate metadata for every timestep
                for _ in range(flat_z.shape[0]):
                    metadata_accumulator.append(row.to_dict())

            except Exception as e:
                print(f"Skipping {row['path']}: {e}")

    # Concatenate
    all_latents = np.concatenate(latents_accumulator, axis=0)

    # Calculate Stats for standardization, for each of the 64 latent channels
    # we want to normalize the latent inputs so the sparsity penalty applies isotropically in the latent space.
    # eg: if chanel 1 has absurd high variance model will focus on it to lower MSE, and l1 penalty will crush other channels, as the activations look small relative
    mean = np.mean(all_latents, axis=0)
    std = np.std(all_latents, axis=0)

    print(f"Saving to {save_path}...")
    np.savez_compressed(save_path, latents=all_latents, mean=mean, std=std)

    pd.DataFrame(metadata_accumulator).to_csv(
        save_path.replace(".npz", "_metadata.csv"), index=False
    )

    print("Done.")


def load_data(path="processed_dataset.npz"):
    # Load the archive
    data = np.load(path)

    # Access arrays by the names we gave them in savez_compressed
    latents = data["latents"]  # The big training data
    mean = data["mean"]  # For normalization
    std = data["std"]  # For normalization

    print(f"Loaded latents: {latents.shape}")
    return latents, mean, std
