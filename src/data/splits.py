"""
Build stratified 5-fold CV splits for ABIDE, stratified by (site, label).

Usage:
    python -m src.data.splits
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def _make_strat_label(pheno: pd.DataFrame, min_count: int = 2) -> np.ndarray:
    """
    Composite stratification label = site + '_' + diagnosis.
    Rare strata (< min_count) are merged into 'other_<label>' to avoid
    StratifiedKFold failures on single-sample strata.
    """
    raw = pheno["site"].astype(str) + "_" + pheno["label"].astype(str)
    counts = raw.value_counts()
    rare = set(counts[counts < min_count].index)
    merged = raw.apply(lambda s: f"other_{s.split('_')[-1]}" if s in rare else s)
    return merged.values


def make_cv_splits(
    pheno: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return list of (train_idx, test_idx) for n_splits folds,
    stratified by (site, label).

    Each tuple contains integer indices into pheno (and into X).
    """
    strat = _make_strat_label(pheno)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = [(tr, te) for tr, te in skf.split(np.arange(len(pheno)), strat)]

    # Print balance report
    y = pheno["label"].values
    for k, (tr, te) in enumerate(folds):
        asd_tr = y[tr].mean()
        asd_te = y[te].mean()
        sites_te = pheno["site"].iloc[te].nunique()
        print(
            f"  Fold {k}: train={len(tr)} (ASD%={asd_tr:.2f})  "
            f"test={len(te)} (ASD%={asd_te:.2f}, sites={sites_te})"
        )
    return folds


def save_splits(folds: list[tuple[np.ndarray, np.ndarray]]) -> Path:
    """Save folds to data/processed/splits.npz (includes n_splits for mismatch detection)."""
    out = PROCESSED_DIR / "splits.npz"
    payload: dict = {"n_splits": np.array([len(folds)])}
    for k, (tr, te) in enumerate(folds):
        payload[f"train_{k}"] = tr
        payload[f"test_{k}"] = te
    np.savez(out, **payload)
    print(f"Saved {len(folds)}-fold splits → {out}")
    return out


def load_splits(n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load previously saved splits, auto-creating them if splits.npz is missing."""
    splits_path = PROCESSED_DIR / "splits.npz"

    if not splits_path.exists():
        pheno_path = PROCESSED_DIR / "phenotype.csv"
        if not pheno_path.exists():
            raise FileNotFoundError(
                f"Neither splits.npz nor phenotype.csv found in {PROCESSED_DIR}.\n"
                "Run: python -m src.data.download_abide"
            )
        print(f"  [splits] splits.npz not found — creating {n_splits}-fold splits ...")
        pheno = pd.read_csv(pheno_path)
        folds = make_cv_splits(pheno, n_splits=n_splits)
        save_splits(folds)

    data = np.load(splits_path)

    # Detect n_splits mismatch between saved file and requested
    if "n_splits" in data:
        saved_n = int(data["n_splits"][0])
        if saved_n != n_splits:
            raise ValueError(
                f"splits.npz was created with n_splits={saved_n} "
                f"but load_splits(n_splits={n_splits}) was called.\n"
                f"Delete {splits_path} and rerun to regenerate with n_splits={n_splits}."
            )

    return [(data[f"train_{k}"], data[f"test_{k}"]) for k in range(n_splits)]


if __name__ == "__main__":
    pheno = pd.read_csv(PROCESSED_DIR / "phenotype.csv")
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"Building {n_splits}-fold CV splits for {len(pheno)} subjects ...")
    folds = make_cv_splits(pheno, n_splits=n_splits)
    save_splits(folds)
