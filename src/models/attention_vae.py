"""
Attention-VAE: VAE with a patch-based Multi-Head Self-Attention block in the encoder.

After the first projection layer (19900 → 512), the 512-dim vector is split into
n_patches=8 patches of dim=64. nn.MultiheadAttention is applied across the patch
sequence, then the output is flattened back to 512 and fed to the rest of the encoder.

This allows the model to learn which groups of FC features are most relevant,
and the attention weights can be inspected for XAI.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 19900


def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> list[nn.Module]:
    return [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.LeakyReLU(0.2), nn.Dropout(dropout)]


class PatchAttention(nn.Module):
    """
    Split a (B, D) vector into (B, n_patches, D/n_patches) patches,
    apply MultiheadAttention, flatten back to (B, D).
    Saves attention weights for inspection.
    """

    def __init__(self, dim: int, n_patches: int = 8, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert dim % n_patches == 0, f"dim ({dim}) must be divisible by n_patches ({n_patches})"
        self.n_patches = n_patches
        self.patch_dim = dim // n_patches  # 512 / 8 = 64

        self.attn = nn.MultiheadAttention(
            embed_dim=self.patch_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        B, D = x.shape
        patches = x.reshape(B, self.n_patches, self.patch_dim)  # (B, n_patches, patch_dim)
        out, weights = self.attn(patches, patches, patches, need_weights=True)  # (B, n_patches, patch_dim)
        self.attn_weights = weights.detach()  # (B, n_heads, n_patches, n_patches) averaged → (B, n_patches, n_patches)
        out_flat = out.reshape(B, D)  # (B, D)
        return self.norm(x + out_flat)  # residual connection + LayerNorm


class AttentionVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 64,
        n_patches: int = 8,
        n_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.latent_dim = latent_dim

        # First projection: input → hidden_dims[0] (512)
        self.input_proj = nn.Sequential(
            *_mlp_block(input_dim, hidden_dims[0], dropout)
        )

        # Patch-based attention on hidden_dims[0]-dim representation
        self.patch_attn = PatchAttention(hidden_dims[0], n_patches=n_patches, n_heads=n_heads)

        # Remaining encoder layers
        enc: list[nn.Module] = []
        prev = hidden_dims[0]
        for h in hidden_dims[1:]:
            enc.extend(_mlp_block(prev, h, dropout))
            prev = h
        self.encoder_rest = nn.Sequential(*enc) if enc else nn.Identity()
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder (mirror)
        dec: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec.extend(_mlp_block(prev, h, dropout))
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

        # Classifier on mu
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)        # (B, 512)
        h = self.patch_attn(h)        # (B, 512) with residual attention
        h = self.encoder_rest(h)      # (B, 256)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

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

    @property
    def attention_weights(self) -> torch.Tensor | None:
        """Last-batch attention weights from the patch attention block."""
        return self.patch_attn.attn_weights

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
