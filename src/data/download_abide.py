"""
Download full ABIDE I dataset, build Fisher-Z FC matrices, save to disk.

Usage:
    python -m src.data.download_abide
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
N_ROIS = 200


def _load_ts(item) -> np.ndarray | None:
    """Load ROI time-series from either ndarray or file path (dual nilearn support)."""
    try:
        if isinstance(item, np.ndarray):
            arr = item
        else:
            arr = np.loadtxt(item)
        if arr.ndim != 2 or arr.shape[1] != N_ROIS:
            return None
        if np.isnan(arr).any():
            return None
        if arr.shape[0] < 50:
            return None
        return np.asarray(arr, dtype=np.float64)
    except Exception:
        return None


def _pearson_fc(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation FC matrix (N_ROIS x N_ROIS)."""
    fc = np.corrcoef(ts.T)
    return np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)


def _fisher_z(fc: np.ndarray) -> np.ndarray:
    """Fisher-Z transform; diagonal set to 0 before arctanh."""
    np.fill_diagonal(fc, 0.0)
    fc_clipped = np.clip(fc, -0.999, 0.999)
    return np.arctanh(fc_clipped)


def download_and_build_fc(n_subjects: int | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Download ABIDE I (quality_checked=True) via nilearn, compute Fisher-Z FC matrices.

    Returns
    -------
    X : ndarray, shape (N, 200, 200)
    pheno : DataFrame with columns [subject_id, label, site, age, sex]
    """
    from nilearn import datasets

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Fetching ABIDE I (n_subjects={n_subjects or 'all'}) ...")
    bunch = datasets.fetch_abide_pcp(
        data_dir=str(DATA_DIR),
        pipeline="cpac",
        derivatives=["rois_cc200"],
        band_pass_filtering=True,
        global_signal_regression=False,
        quality_checked=True,
        n_subjects=n_subjects,
        verbose=1,
    )

    pheno_raw = pd.DataFrame(bunch.phenotypic)
    pheno_raw.columns = [str(c) for c in pheno_raw.columns]

    print(f"[2/3] Building FC matrices for {len(bunch.rois_cc200)} subjects ...")
    fc_matrices: list[np.ndarray] = []
    valid_idx: list[int] = []

    for i, item in enumerate(bunch.rois_cc200):
        ts = _load_ts(item)
        if ts is None:
            print(f"      [skip] idx {i}")
            continue
        fc = _pearson_fc(ts)
        fc_z = _fisher_z(fc)
        fc_matrices.append(fc_z)
        valid_idx.append(i)

    X = np.stack(fc_matrices, axis=0)  # (N, 200, 200)

    pheno = pd.DataFrame({
        "subject_id": pheno_raw.iloc[valid_idx]["SUB_ID"].astype(str).values,
        "label": (pheno_raw.iloc[valid_idx]["DX_GROUP"].astype(int) == 1).astype(int).values,
        "site": pheno_raw.iloc[valid_idx]["SITE_ID"].astype(str).values,
        "age": pd.to_numeric(pheno_raw.iloc[valid_idx]["AGE_AT_SCAN"], errors="coerce").values,
        "sex": pheno_raw.iloc[valid_idx]["SEX"].astype(int).values,
    })

    print(f"[3/3] Saving to {PROCESSED_DIR} ...")
    np.save(PROCESSED_DIR / "fc_matrices.npy", X)
    pheno.to_csv(PROCESSED_DIR / "phenotype.csv", index=False)

    n_asd = int(pheno["label"].sum())
    n_ctrl = int((pheno["label"] == 0).sum())
    print(f"\nDone — {len(X)} subjects  |  ASD: {n_asd}  |  Control: {n_ctrl}")
    print(f"Sites ({pheno['site'].nunique()}): {sorted(pheno['site'].unique())}")

    # Save dataset statistics for the report
    _save_dataset_stats(X, pheno)

    # Auto-create 5-fold splits so downstream code never fails on missing splits.npz
    try:
        from src.data.splits import make_cv_splits, save_splits
        print("\n[+] Auto-creating 5-fold CV splits ...")
        folds = make_cv_splits(pheno, n_splits=5)
        save_splits(folds)
    except Exception as e:
        print(f"  [warn] Could not auto-create splits: {e}")
        print("  Run manually: python -m src.data.splits")

    return X, pheno


def _save_dataset_stats(X: np.ndarray, pheno: pd.DataFrame) -> None:
    """Compute and save dataset_stats.json for the diploma report."""
    import json

    y = pheno["label"]
    age = pheno["age"]

    def age_stats(mask) -> dict:
        a = age[mask].dropna()
        return {
            "mean": round(float(a.mean()), 2),
            "std":  round(float(a.std()),  2),
            "min":  round(float(a.min()),  2),
            "max":  round(float(a.max()),  2),
            "n":    int(a.notna().sum()),
        }

    site_table: dict = {}
    for site in sorted(pheno["site"].unique()):
        mask = pheno["site"] == site
        site_table[site] = {
            "total":   int(mask.sum()),
            "asd":     int((y[mask] == 1).sum()),
            "control": int((y[mask] == 0).sum()),
        }

    stats = {
        "n_total":        int(len(X)),
        "n_asd":          int((y == 1).sum()),
        "n_control":      int((y == 0).sum()),
        "n_sites":        int(pheno["site"].nunique()),
        "sites":          sorted(pheno["site"].unique().tolist()),
        "site_table":     site_table,
        "fc_shape":       list(X.shape),       # [N, 200, 200]
        "n_features":     int(200 * 199 // 2), # 19900 upper-triangle
        "age_all":        age_stats(slice(None)),
        "age_asd":        age_stats(y == 1),
        "age_control":    age_stats(y == 0),
        "sex_total_male":     int((pheno["sex"] == 1).sum()),
        "sex_total_female":   int((pheno["sex"] == 2).sum()),
        "sex_asd_male":       int(((y == 1) & (pheno["sex"] == 1)).sum()),
        "sex_asd_female":     int(((y == 1) & (pheno["sex"] == 2)).sum()),
        "sex_control_male":   int(((y == 0) & (pheno["sex"] == 1)).sum()),
        "sex_control_female": int(((y == 0) & (pheno["sex"] == 2)).sum()),
    }

    out = PROCESSED_DIR / "dataset_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Dataset stats saved → {out}")


def load_processed() -> tuple[np.ndarray, pd.DataFrame]:
    """Load previously saved FC matrices and phenotype."""
    X = np.load(PROCESSED_DIR / "fc_matrices.npy")
    pheno = pd.read_csv(PROCESSED_DIR / "phenotype.csv")
    return X, pheno


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    download_and_build_fc(n_subjects=n)
