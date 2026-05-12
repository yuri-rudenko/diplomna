"""
Per-feature StandardScaler fitted on train, applied to val/test.
Operates on vectorized upper-triangle FC arrays (N, 19900).
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.dataset import upper_triangle


def vectorize(X: np.ndarray) -> np.ndarray:
    """(N, 200, 200) → (N, 19900) upper-triangle vectors."""
    return np.stack([upper_triangle(x) for x in X], axis=0).astype(np.float32)


def fit_scaler(X_train_vec: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train_vec)
    return scaler


def apply_scaler(
    scaler: StandardScaler,
    X_train_vec: np.ndarray,
    X_test_vec: np.ndarray,
    X_val_vec: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Transform train/val/test with a pre-fitted scaler."""
    out = [scaler.transform(X_train_vec), scaler.transform(X_test_vec)]
    if X_val_vec is not None:
        out.insert(1, scaler.transform(X_val_vec))
    return tuple(out)
