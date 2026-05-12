"""
Sparse Autoencoder + MLP classifier (ASD-SAENet style).

Architecture: 19900 → 512 → 256 → latent(64) → 256 → 512 → 19900
Classifier:   latent(64) → 64 → 2

Reference: Almuqhim & Saeed (2021), Front. Comput. Neurosci.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 19900


def _make_encoder(input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, latent_dim), nn.Sigmoid()]
    return nn.Sequential(*layers)


def _make_decoder(latent_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = latent_dim
    for h in reversed(hidden_dims):
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 64,
        sparsity_target: float = 0.05,
        sparsity_weight: float = 1e-3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.latent_dim = latent_dim

        self.encoder = _make_encoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = _make_decoder(latent_dim, hidden_dims, input_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        logits = self.classifier(z)
        return x_recon, z, logits

    def sparse_loss(self, z: torch.Tensor) -> torch.Tensor:
        """KL divergence between target sparsity rho and mean activation per neuron."""
        rho = self.sparsity_target
        rho_hat = z.mean(dim=0).clamp(1e-8, 1.0 - 1e-8)
        kl = rho * torch.log(torch.tensor(rho) / rho_hat) + \
             (1 - rho) * torch.log(torch.tensor(1 - rho) / (1 - rho_hat))
        return kl.sum()

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        logits: torch.Tensor,
        y: torch.Tensor,
        recon_weight: float = 1.0,
        cls_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(x_recon, x)
        cls_loss = F.cross_entropy(logits, y)
        sp_loss = self.sparse_loss(z)
        total = recon_weight * recon_loss + cls_weight * cls_loss + self.sparsity_weight * sp_loss
        return total, recon_loss, cls_loss, sp_loss
