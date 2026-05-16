"""
PyTorch Dataset for ABIDE FC matrices.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

N_ROIS = 200
_UPPER_IDX = np.triu_indices(N_ROIS, k=1)  # (2, 19900) — cached once


def upper_triangle(fc_matrix: np.ndarray) -> np.ndarray:
    """Extract upper triangle (k=1) from a (200, 200) FC matrix → (19900,)."""
    return fc_matrix[_UPPER_IDX]


class ABIDEDataset(Dataset):
    """
    Parameters
    ----------
    X : ndarray (N, 200, 200) or (N, 19900)  — FC matrices or pre-vectorized
    y : ndarray (N,)                          — labels: 1=ASD, 0=Control
    noise_std : float  — std of Gaussian noise added during training (0 = off)
    augment   : bool   — enable noise augmentation

    If X is already flat (N, 19900) it is used as-is (e.g. after ComBat+scaler).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.02,
    ) -> None:
        if X.ndim == 3:
            vecs = np.stack([upper_triangle(x) for x in X], axis=0)
        else:
            vecs = X
        self.X = torch.from_numpy(vecs.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.augment and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.y[idx]


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup augmentation: blend two random samples.
    Returns (mixed_x, y_a, y_b, lam) — loss = lam*CE(y_a) + (1-lam)*CE(y_b).
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam
