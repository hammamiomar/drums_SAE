import torch
import torch.nn.functional as F
from torch import nn
from wandb.data_types import Audio


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x --> (batch,hidden_dim)
        # Root mean squared --> sqrt(mean(x**2)), norm is divide x by it
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm


class AudioSae(nn.Module):
    """
    Sparse Auto Encoder with RMSNorm for audio latents

    Architecture follows Anthropic's "Scaling Monosemanticity" (2024) with
        RMSNorm addition from the audio SAE paper for steering stability.

    Apply RMSNorm after the encoder activation, to force features onto hypersphere
        so then we can prevent high energy audio features dominatiing sparsity penalty
    Forward pass:
        1. x_centered = x - pre_bias
        2. pre_acts = encoder(x_centered) + latent_bias
        2. h = activation(pre_acts) # activations, relu or topk+relu
        3. f = RMSNorm(h)   # normalized for steering
        4. x_hat = decoder(f) + pre_bias

    Diagram:
    Input x
       │
       ▼
    ┌─────────────────────────────────────────────┐
    │              ENCODER                         │
    │         pre_acts = Wx + b                    │
    └─────────────────────────────────────────────┘
       │
       ├──────────────────────┬───────────────────┐
       │                      │                   │
       ▼                      ▼                   │
    ┌──────────┐        ┌───────────┐             │
    │  TopK    │        │  TopK     │             │
    │ (all     │        │ (dead     │             │
    │ features)│        │ features  │             │
    │          │        │  only)    │             │
    └──────────┘        └───────────┘             │
       │                      │                   │
       ▼                      ▼                   │
       h                    h_aux                 │
       │                      │                   │
       ▼                      ▼                   │
    ┌──────────┐        ┌───────────┐             │
    │ Decoder  │        │ Decoder   │             │
    └──────────┘        └───────────┘             │
       │                      │                   │
       ▼                      ▼                   │
     x_hat                aux_recon               │
       │                      │                   │
       ▼                      │                   │
    residual = x - x_hat ◄────┘                   │
                                                  │
                                                  │
    LOSSES:                                       │
      MSE:  ||x - x_hat||²  ◄─────────────────────┘
      AuxK: ||residual - aux_recon||²

    """

    def __init__(
        self,
        d_input: int,  # input dim, which is 64 for stable audio vae
        expansion_factor: int = 16,  # d_hidden = d_input * expansion_factor
        topk: int = 64,  # num features to activate
        topk_aux: int = 128,  # num of dead features to revive, for auxK loss
        dead_threshold: int = 10_000,  # steps without firing before feature dead
    ):
        super().__init__()

        self.d_input = d_input
        self.d_hidden = d_input * expansion_factor
        self.expansion_factor = expansion_factor
        self.topk = topk
        self.topk_aux = topk_aux
        self.dead_threshold = dead_threshold

        # Encoder: d_input -> d_hidden
        self.encoder = nn.Linear(self.d_input, self.d_hidden, bias=False)

        # Decoder: d_hidden -> d_input
        self.decoder = nn.Linear(self.d_hidden, self.d_input, bias=False)

        # Biases (separate for flexibility)
        self.pre_bias = nn.Parameter(torch.zeros(self.d_input))
        self.latent_bias = nn.Parameter(torch.zeros(self.d_hidden))

        # RMSNorm for steering stability
        self.rms_norm = RMSNorm()

        # track dead features, with a buffer.
        self.register_buffer(
            "steps_since_fired", torch.zeros(self.d_hidden, dtype=torch.long)
        )
        # Initialize encoder/decoder with tied init
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)

        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        """ """
        self.decoder.weight.data = F.normalize(
            self.decoder.weight.data, dim=0
        )  # dim 0, we are normalizing along each eature col, since decoder data is shape (d_input,d_hidden)
        # where d_hidden is the num of features.

    def _topk_activation(
        self,
        pre_acts: torch.Tensor,
        k: int,
        mask: torch.Tensor | None = None,
    ):
        if (
            mask is not None
        ):  # for auxK, so we can get rid of the alive nodes: focus on dead
            pre_acts = pre_acts * mask

        values, indices = torch.topk(pre_acts, k=k, dim=-1)

        values = F.relu(values)

        h = torch.zeros_like(pre_acts)
        h.scatter_(dim=-1, index=indices, src=values)

        return h, indices

    @torch.no_grad()
    def _update_dead_feature_stats(self, indices: torch.Tensor):
        self.steps_since_fired += 1

        # reset counters for features that fired
        fired = indices.flatten().unique()
        self.steps_since_fired[fired] = 0

    def get_dead_feature_mask(self) -> torch.Tensor:
        # 1 is dead, 0 alive
        return (self.steps_since_fired > self.dead_threshold).float()

    def encode(self, x: torch.Tensor, return_aux=False):
        # x of shape (batch,d_input)
        # return_aux : if true returen AuxK activations for dead features
        x_centered = x - self.pre_bias
        pre_acts = self.encoder(x_centered) + self.latent_bias

        h, indices = self._topk_activation(pre_acts, k=self.topk)

        if self.training:
            self._update_dead_feature_stats(indices)

        f = self.rms_norm(h)
        result = {"h": h, "f": f, "indices": indices, "pre_acts": pre_acts}

        # aux k: activate top-k DEAD features, for auxiliary loss.
        if return_aux and self.training:
            dead_mask = self.get_dead_feature_mask()
            if dead_mask.sum() > 0:
                h_aux, indices_aux = self._topk_activation(
                    pre_acts,
                    k=min(self.topk_aux, int(dead_mask.sum())),
                    mask=dead_mask,
                )
                result["h_aux"] = h_aux
                result["indices_aux"] = indices_aux
        return result

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f) + self.pre_bias

    def forward(self, x, return_aux=True):
        """
        returns:
            x_hat: reconstruction
            h: sparse activations, pre normalization
            f: normalized activations
            indices: active feature indices
            h_aux:AuxK features, if training. so its the leftover dead feautes that we wanna train with, to get what the main features missed
            residual: x - x_hat (for auxK loss)
        """
        enc = self.encode(x, return_aux=return_aux)
        x_hat = self.decode(enc["f"])

        return {
            "x_hat": x_hat,
            "residual": x - x_hat,
            **enc,
        }


# --- Loss funcs---


def sae_loss(
    x, output: dict[str, torch.Tensor], model: AudioSae, auxk_coef: float = 1 / 32
):
    """
    topK SAE loss, with AuxK for feature revival.

    Loss = MSE(x,x_hat) + auxk_coef * MSE(residual,aux,aux_reconstruction)

    we are giving the dead features some signal to reconstruct the main reconstructin diff
        Main features reconstruct:     x_hat = decode(topk(x))
        Residual (what's left over):   residual = x - x_hat
        Dead features reconstruct:     aux_recon = decode(topk_dead_only(x))

        AuxK Loss: ||residual - aux_recon||²
        This gives dead features a gradient signal! They learn to reconstruct what the main features missed.
        Once they get good at this, they might become useful enough to enter the top-k for some inputs → they "come alive."
    """
    mse = (x - output["x_hat"]).pow(2).mean()

    losses = {
        "mse": mse,
        "total": mse,
    }

    if "h_aux" in output and output["h_aux"] is not None:
        # decode using aux dead features
        f_aux = model.rms_norm(output["h_aux"])
        aux_recon = model.decode(f_aux)

        # target: residual, whatever the main features couldnt reconstruct
        residual = output["residual"].detach()  # dont backprop through

        auxk_loss = (residual - aux_recon).pow(2).mean()

        losses["auxk"] = auxk_loss
        losses["total"] = mse + auxk_coef * auxk_loss

    return losses


# --- Convenience Func


def compute_metrics(
    output: dict[str, torch.Tensor],
    model: AudioSae,
) -> dict[str, float]:
    h = output["h"]

    # L0: average number of active features
    l0 = (h > 0).float().sum(dim=-1).mean()

    # Sparsity: fraction of zeros
    sparsity = (h == 0).float().mean()

    # Dead features
    dead_count = (model.steps_since_fired > model.dead_threshold).sum()
    dead_fraction = dead_count / model.d_hidden

    return {
        "sparsity/l0": l0.item(),
        "sparsity/fraction_zero": sparsity.item(),
        "features/dead_count": dead_count.item(),
        "features/dead_fraction": dead_fraction.item(),
    }
