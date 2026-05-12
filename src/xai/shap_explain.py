"""
SHAP-based explanation for the trained models.

Pipeline:
  1. Load best-fold checkpoint of a model
  2. Wrap model so SHAP sees only P(ASD) output
  3. Run DeepExplainer with train background (100 subjects)
  4. shap_values: (N_test, 19900) — importance per FC connection
  5. Aggregate to ROI importance (200,) via mean |SHAP| per ROI
  6. Save shap_values.npy, roi_importance.npy, shap_top20_table.csv

Usage:
    python -m src.xai.shap_explain --model attention_vae --fold best
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT_DIR / "results" / "checkpoints"
FIGURES_DIR = PROJECT_DIR / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_ROIS = 200
_UPPER_IDX = np.triu_indices(N_ROIS, k=1)  # (2, 19900)


# ---------------------------------------------------------------------------
# Model wrapper for SHAP
# ---------------------------------------------------------------------------

class _ASDProbWrapper(nn.Module):
    """Return P(ASD) as a (B, 1) tensor — required by shap.DeepExplainer."""

    def __init__(self, model, model_type: str) -> None:
        super().__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        if self.model_type == "sae":
            _, _, logits = self.model(x)
        else:
            _, mu, _, _, logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[:, 1:2]  # (B, 1)
        return probs


# ---------------------------------------------------------------------------
# ROI aggregation
# ---------------------------------------------------------------------------

def shap_to_roi_importance(shap_values: np.ndarray) -> np.ndarray:
    """
    Aggregate connection-level SHAP values to ROI importance.

    Parameters
    ----------
    shap_values : (N, 19900)  — SHAP values per upper-triangle connection

    Returns
    -------
    roi_importance : (N, 200)  — mean |SHAP| across all connections involving each ROI
    """
    N = shap_values.shape[0]
    roi_importance = np.zeros((N, N_ROIS), dtype=np.float32)
    rows, cols = _UPPER_IDX

    for i in range(N_ROIS):
        # All connection indices where ROI i appears (as either endpoint)
        conn_mask = (rows == i) | (cols == i)
        roi_importance[:, i] = np.abs(shap_values[:, conn_mask]).mean(axis=1)

    return roi_importance


# ---------------------------------------------------------------------------
# Main explain function
# ---------------------------------------------------------------------------

def explain_model(
    model_type: str = "attention_vae",
    fold: str = "best",
    n_background: int = 100,
    device: torch.device | None = None,
) -> dict:
    """
    Run SHAP explanation on a trained model.

    Parameters
    ----------
    model_type : 'sae' | 'vae' | 'attention_vae'
    fold       : 'best' (pick fold with highest AUC) or int fold index
    """
    import shap

    from src.data.download_abide import load_processed
    from src.data.splits import load_splits
    from src.data.preprocessing import apply_scaler, fit_scaler, vectorize
    from src.data.harmonize import apply_combat, fit_combat

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_raw, pheno = load_processed()
    folds = load_splits()

    # Pick fold
    if fold == "best":
        fold_idx = _best_fold(model_type)
    else:
        fold_idx = int(fold)

    train_idx, test_idx = folds[fold_idx]
    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train = pheno["label"].values[train_idx]
    y_test = pheno["label"].values[test_idx]

    X_train_vec = vectorize(X_train_raw)
    X_test_vec = vectorize(X_test_raw)

    # ComBat
    try:
        sites_tr = pheno["site"].values[train_idx]
        sites_te = pheno["site"].values[test_idx]
        X_train_vec, combat_params = fit_combat(
            X_train_vec, sites_tr,
            pheno["age"].values[train_idx], pheno["sex"].values[train_idx]
        )
        X_test_vec = apply_combat(
            combat_params, X_test_vec, sites_te,
            pheno["age"].values[test_idx], pheno["sex"].values[test_idx]
        )
    except ImportError:
        print("  [warn] neuroCombat not available — skipping harmonization")

    scaler = fit_scaler(X_train_vec)
    X_train_n, X_test_n = apply_scaler(scaler, X_train_vec, X_test_vec)

    # Load model
    model = _load_model(model_type, fold_idx, device)
    wrapper = _ASDProbWrapper(model, model_type).to(device)
    wrapper.eval()

    # Background: random 100 train subjects
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train_n), size=min(n_background, len(X_train_n)), replace=False)
    background = torch.FloatTensor(X_train_n[bg_idx]).to(device)
    test_tensor = torch.FloatTensor(X_test_n).to(device)

    print(f"Running DeepSHAP: background={len(background)}, test={len(test_tensor)} ...")
    explainer = shap.DeepExplainer(wrapper, background)
    shap_values_raw = explainer.shap_values(test_tensor, check_additivity=False)

    # shap_values_raw may be list (one per output) or array depending on shap version
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[0]
    else:
        shap_values = shap_values_raw

    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().numpy()

    shap_values = np.array(shap_values, dtype=np.float32)
    if shap_values.ndim == 3:       # (N, features, 1) → (N, features)
        shap_values = shap_values[..., 0]
    # shap_values: (N_test, 19900)

    # ROI aggregation
    roi_importance = shap_to_roi_importance(shap_values)  # (N_test, 200)
    mean_roi_importance = roi_importance.mean(axis=0)      # (200,)

    # Save
    tag = f"{model_type}_fold{fold_idx}"
    np.save(FIGURES_DIR / f"shap_values_{tag}.npy", shap_values)
    np.save(FIGURES_DIR / f"roi_importance_{tag}.npy", mean_roi_importance)

    top20_idx = np.argsort(mean_roi_importance)[::-1][:20]
    top20_df = pd.DataFrame({
        "rank": range(1, 21),
        "roi_index": top20_idx,
        "mean_abs_shap": mean_roi_importance[top20_idx],
    })
    top20_path = FIGURES_DIR / "shap_top20_table.csv"
    top20_df.to_csv(top20_path, index=False)

    print(f"\nTop-20 ROIs by |SHAP|:")
    print(top20_df.to_string(index=False))
    print(f"\nSaved SHAP artifacts → {FIGURES_DIR}")

    return {
        "shap_values": shap_values,
        "roi_importance": mean_roi_importance,
        "top20": top20_df,
        "y_test": y_test,
    }


def _best_fold(model_type: str) -> int:
    """Return fold index with highest test AUC from cv_per_fold.csv."""
    metrics_path = PROJECT_DIR / "results" / "metrics" / "cv_per_fold.csv"
    if not metrics_path.exists():
        print("  [warn] cv_per_fold.csv not found — using fold 0")
        return 0
    df = pd.read_csv(metrics_path)
    subset = df[df["model"] == model_type]
    if subset.empty:
        return 0
    return int(subset.loc[subset["auc"].idxmax(), "fold"])


def _load_model(model_type: str, fold_idx: int, device: torch.device):
    """Load model from checkpoint."""
    from src.models.sparse_ae import SparseAutoencoder
    from src.models.vae import VariationalAutoencoder
    from src.models.attention_vae import AttentionVAE

    model_map = {
        "sae": SparseAutoencoder,
        "vae": VariationalAutoencoder,
        "attention_vae": AttentionVAE,
    }
    model = model_map[model_type]().to(device)
    ckpt_path = CKPT_DIR / f"{model_type}_fold{fold_idx}_best_auc.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="attention_vae", choices=["sae", "vae", "attention_vae"])
    parser.add_argument("--fold", default="best")
    parser.add_argument("--n-background", type=int, default=100)
    args = parser.parse_args()
    explain_model(model_type=args.model, fold=args.fold, n_background=args.n_background)
