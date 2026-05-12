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

    If X is already flat (N, 19900) it is used as-is (e.g. after ComBat+scaler).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim == 3:
            vecs = np.stack([upper_triangle(x) for x in X], axis=0)  # (N, 19900)
        else:
            vecs = X  # already (N, 19900)
        self.X = torch.from_numpy(vecs.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
