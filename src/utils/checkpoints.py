"""
Resolve checkpoints / scalers / best fold for an experiment, with fallback.

run_experiments.py writes per-experiment artifacts under
``results/experiments/{exp}/checkpoints/`` (prefixed with the experiment name),
while the older compare_models.py path used a flat ``results/checkpoints/``.
These helpers resolve either layout from a single ``--exp`` flag so no manual
file copying is ever required.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_DIR / "results" / "experiments"
CKPT_DIR = PROJECT_DIR / "results" / "checkpoints"          # legacy compare_models.py path
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def best_fold(model_type: str, exp_name: str | None = None, default: int = 0) -> int:
    """Return the fold index with the highest test AUC for ``model_type``.

    Reads ``results/experiments/{exp}/metrics/cv_per_fold.csv`` first, then the
    global ``results/metrics/cv_per_fold.csv``. Falls back to ``default``.
    """
    candidates = []
    if exp_name:
        candidates.append(EXPERIMENTS_DIR / exp_name / "metrics" / "cv_per_fold.csv")
    candidates.append(PROJECT_DIR / "results" / "metrics" / "cv_per_fold.csv")

    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        sub = df[df["model"] == model_type]
        if sub.empty:
            continue
        return int(sub.loc[sub["auc"].idxmax(), "fold"])

    print(f"  [warn] cv_per_fold.csv not found — using fold {default}")
    return default


def resolve_fold(fold, model_type: str, exp_name: str | None = None) -> int:
    """Map a ``--fold`` arg ('best' or an int-like) to a concrete fold index."""
    if isinstance(fold, str) and fold.lower() == "best":
        return best_fold(model_type, exp_name)
    return int(fold)


def resolve_checkpoint(
    model_name: str,
    fold_idx: int,
    exp_name: str | None = None,
) -> Path | None:
    """Locate ``{model}_fold{k}_best_auc.pth`` in the experiment or legacy dir."""
    candidates = []
    if exp_name:
        candidates.append(
            EXPERIMENTS_DIR / exp_name / "checkpoints"
            / f"{exp_name}_{model_name}_fold{fold_idx}_best_auc.pth"
        )
    candidates.append(CKPT_DIR / f"{model_name}_fold{fold_idx}_best_auc.pth")
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_scaler(fold_idx: int, exp_name: str | None = None) -> Path | None:
    """Locate the saved StandardScaler for a fold (experiment dir, then global)."""
    candidates = []
    if exp_name:
        candidates.append(
            EXPERIMENTS_DIR / exp_name / "checkpoints" / f"scaler_fold{fold_idx}.pkl"
        )
    candidates.append(PROCESSED_DIR / f"scaler_fold{fold_idx}.pkl")
    for c in candidates:
        if c.exists():
            return c
    return None
