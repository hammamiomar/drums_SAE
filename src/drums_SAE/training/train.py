from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import wandb
from tqdm import tqdm

from drums_SAE.sae.model import AudioSae, compute_metrics, sae_loss
from drums_SAE.training.data import create_dataloader, infinite_dataloader


@dataclass
class TrainConfig:
    # Data
    data_path: str = "data/drums_encoded.npz"

    # Model
    d_input: int = 64
    expansion_factor: int = 16
    topk: int = 64
    topk_aux: int = 128
    dead_threshold: int = 10_000

    # Training
    batch_size: int = 256
    lr: float = 1e-4
    num_steps: int = 50_000
    auxk_coef: float = 1 / 32

    # Logging & Checkpoints
    log_every: int = 100
    save_every: int = 5_000
    checkpoint_dir: str = "checkpoints"

    # W&B
    wandb_project: str = "drums_SAE"
    wandb_name: str | None = None  # auto-generated if None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(config: TrainConfig):
    """Main training loop."""

    device = get_device()
    print(f"Using device: {device}")

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_name,
        config=asdict(config),
    )

    # Data
    dataloader = create_dataloader(
        config.data_path,
        batch_size=config.batch_size,
        shuffle=True,
    )
    data_iter = infinite_dataloader(dataloader)

    print(f"Dataset size: {len(dataloader.dataset):,} samples")
    print(f"Steps per epoch: {len(dataloader):,}")

    # Model
    model = AudioSae(
        d_input=config.d_input,
        expansion_factor=config.expansion_factor,
        topk=config.topk,
        topk_aux=config.topk_aux,
        dead_threshold=config.dead_threshold,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Hidden dim: {model.d_hidden}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training Loop
    model.train()

    pbar = tqdm(range(config.num_steps), desc="Training")
    for step in pbar:
        # Get batch
        batch = next(data_iter).to(device)

        # Forward pass
        output = model(batch, return_aux=True)
        losses = sae_loss(batch, output, model, auxk_coef=config.auxk_coef)

        # Backward pass
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        # Maintain unit norm decoder (IMPORTANT!)
        model._normalize_decoder()

        # Logging
        if step % config.log_every == 0:
            metrics = compute_metrics(output, model)

            log_dict = {
                "loss/total": losses["total"].item(),
                "loss/mse": losses["mse"].item(),
                **metrics,
            }

            # AuxK loss might not exist if no dead features
            if "auxk" in losses:
                log_dict["loss/auxk"] = losses["auxk"].item()

            wandb.log(log_dict, step=step)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{losses['total'].item():.4f}",
                    "dead": f"{metrics['features/dead_count']:.0f}",
                }
            )

        # Checkpointing
        if step > 0 and step % config.save_every == 0:
            save_checkpoint(model, optimizer, step, config, checkpoint_dir)

    # Final save
    save_checkpoint(model, optimizer, config.num_steps, config, checkpoint_dir)
    wandb.finish()

    print("Training complete!")
    return model


def save_checkpoint(model, optimizer, step, config, checkpoint_dir):
    """Save model checkpoint."""
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
    }

    path = checkpoint_dir / f"sae_step_{step}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")

    # Also save as 'latest'
    latest_path = checkpoint_dir / "sae_latest.pt"
    torch.save(checkpoint, latest_path)


def load_checkpoint(path: str, device: str = None):
    """Load model from checkpoint."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)
    config = TrainConfig(**checkpoint["config"])

    model = AudioSae(
        d_input=config.d_input,
        expansion_factor=config.expansion_factor,
        topk=config.topk,
        topk_aux=config.topk_aux,
        dead_threshold=config.dead_threshold,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    return model, config, checkpoint["step"]


# Entry point
if __name__ == "__main__":
    config = TrainConfig(
        # Override defaults here or use CLI args
        data_path="data/drums_encoded.npz",
        num_steps=50_000,
        batch_size=256,
    )

    train(config)
