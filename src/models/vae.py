"""
Variational Autoencoder + MLP classifier.

Architecture: 19900 → 512 → 256 → (mu, logvar)(64) → 256 → 512 → 19900
Classifier on mu: 64 → 64 → 2

KL annealing is handled externally via the `beta` argument to `.loss()`.
During eval, reparameterize returns mu (deterministic inference).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 19900


def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> list[nn.Module]:
    return [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.LeakyReLU(0.2), nn.Dropout(dropout)]


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.latent_dim = latent_dim

        # Encoder shared trunk
        enc: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc.extend(_mlp_block(prev, h, dropout))
            prev = h
        self.encoder_shared = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder
        dec: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec.extend(_mlp_block(prev, h, dropout))
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

        # Classifier operates on mu (stable, no noise)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_shared(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu  # deterministic at eval time

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        logits = self.classifier(mu)
        return x_recon, mu, logvar, z, logits

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        logits: torch.Tensor,
        y: torch.Tensor,
        beta: float = 1.0,
        recon_weight: float = 1.0,
        cls_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        cls_loss = F.cross_entropy(logits, y)
        total = recon_weight * recon_loss + beta * kl_loss + cls_weight * cls_loss
        return total, recon_loss, kl_loss, cls_loss
