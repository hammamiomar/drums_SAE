import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LatentDataset(Dataset):
    # for pre encoded audio latents.
    # since dataset is small, can load directly
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


def create_dataloader(
    npz_path: str,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,  # 0 is fine for in-memory data
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for latent data."""
    dataset = LatentDataset(npz_path)
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
