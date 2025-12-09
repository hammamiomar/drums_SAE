import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class LatentDataset(Dataset):
    """V1 dataset - loads all latents without filtering (legacy)."""

    def __init__(self, npz_path, normalize=True):
        data = np.load(npz_path)

        latents = torch.from_numpy(
            data["latents"]
        ).float()  # (N, 64) ensure from possible float64 to float32
        self.mean = torch.from_numpy(data["mean"]).float()  # (64, )
        self.std = torch.from_numpy(data["std"]).float()  # (64, )

        if normalize:  # normalizing, channel wise, like for each of the 64 columns
            self.latents = (latents - self.mean) / (self.std + 1e-8)
        else:
            self.latents = latents

        print(f"Loaded {len(self.latents):,} latent vecs from {npz_path}")

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]


class LatentDatasetV2(Dataset):
    """V2 dataset with silence filtering for aligned latents + features.

    Loads latents from NPZ and optionally filters out silence timesteps
    using the aligned features CSV. This focuses training on meaningful
    audio content (attack/decay) rather than wasting capacity on silence.

    Args:
        npz_path: Path to latents_v2.npz with 'latents', 'mean', 'std'
        features_path: Path to features_v2.csv with 'is_silence' column
        filter_silence: If True, exclude silence timesteps from training
        normalize: If True, normalize latents using stored mean/std
    """

    def __init__(
        self,
        npz_path: str,
        features_path: str | None = None,
        filter_silence: bool = True,
        normalize: bool = True,
    ):
        data = np.load(npz_path)
        latents = torch.from_numpy(data["latents"]).float()  # (N, 64)
        self.mean = torch.from_numpy(data["mean"]).float()  # (64,)
        self.std = torch.from_numpy(data["std"]).float()  # (64,)

        n_total = len(latents)

        # Filter silence using features CSV (row-aligned with latents)
        if filter_silence and features_path is not None:
            df = pd.read_csv(features_path, usecols=["is_silence"])
            trainable_mask = ~df["is_silence"].values
            latents = latents[trainable_mask]
            print(
                f"Filtered to {len(latents):,} trainable timesteps "
                f"({len(latents) / n_total * 100:.1f}% of {n_total:,})"
            )
        else:
            print(f"Loaded {len(latents):,} latent vecs (no filtering)")

        # Normalize per-channel using dataset statistics
        if normalize:
            self.latents = (latents - self.mean) / (self.std + 1e-8)
        else:
            self.latents = latents

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.latents[idx]


def create_dataloader(
    npz_path: str,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,  # 0 is fine for in-memory data
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for latent data (v1 legacy)."""
    dataset = LatentDataset(npz_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )


def create_dataloader_v2(
    npz_path: str,
    features_path: str | None = None,
    filter_silence: bool = True,
    batch_size: int = 4096,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for v2 latent data with silence filtering."""
    dataset = LatentDatasetV2(
        npz_path=npz_path,
        features_path=features_path,
        filter_silence=filter_silence,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )


def infinite_dataloader(dataloader: DataLoader):
    """
    Wrap a DataLoader to loop forever.

    Useful for step-based (not epoch-based) training.
    """
    while True:
        for batch in dataloader:
            yield batch
