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


def create_dataset_manifest(root_path: str):
    data_path = Path(root_path) / "one_shot_percussive_sounds"
    analysis_path = Path(root_path) / "analysis"

    # get all wav files
    wav_files = sorted(data_path.rglob("*.wav"))
    print(f"found {len(wav_files)} audio files")

    # get all json files
    json_files = sorted(analysis_path.rglob("*.json"))
    print(f"found {len(json_files)} analysis files")

    records = []

    for wav_file in data_path.rglob("*.wav"):
        # wav_file = data/1/1234.wav
        relative_path = wav_file.relative_to(data_path)
        folder = relative_path.parts[0]
        stem = relative_path.stem  # 1234

        json_file = analysis_path / folder / f"{stem}_analysis.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                meta = json.load(f)

            records.append(
                {
                    "id": int(stem),
                    "path": str(wav_file),
                    "loudness": meta.get("loudness"),
                    "single_event": meta.get("single_event"),
                    "hardness": meta.get("hardness"),
                    "depth": meta.get("depth"),
                    "brightness": meta.get("brightness"),
                    "roughness": meta.get("roughness"),
                    "warmth": meta.get("warmth"),
                    "sharpness": meta.get("sharpness"),
                    "boominess": meta.get("boominess"),
                    "reverb": meta.get("reverb"),
                }
            )
    df = pd.DataFrame(records)
    df.set_index("id", inplace=True)
    df.sort_index(inplace=True)
    print(f"have records with shape {df.shape}")
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
    target_length = 65536

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
                # Output shape: [1, 64, 32] (Batch, Channels, Time)
                encoded_latents = vae.encode(input_tensor)

                flat_z = rearrange(encoded_latents, "b c t -> (b t) c").cpu().numpy()

                latents_accumulator.append(flat_z)

                # Duplicate metadata for every timestep
                for _ in range(flat_z.shape[0]):
                    metadata_accumulator.append(row.to_dict())

            except Exception as e:
                print(f"Skipping {row['path']}: {e}")

    # Concatenate
    all_latents = np.concatenate(latents_accumulator, axis=0)

    # Calculate Stats
    mean = np.mean(all_latents, axis=0)
    std = np.std(all_latents, axis=0)

    print(f"Saving to {save_path}...")
    np.savez_compressed(save_path, latents=all_latents, mean=mean, std=std)

    pd.DataFrame(metadata_accumulator).to_csv(
        save_path.replace(".npz", "_metadata.csv"), index=False
    )

    print("Done.")
