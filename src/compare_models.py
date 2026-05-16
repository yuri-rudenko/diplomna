"""
Main entry point: 5-fold CV training and comparison of SAE / VAE / Attention-VAE.

Usage:
    python -m src.compare_models                          # full run
    python -m src.compare_models --cv 5 --epochs 150
    python -m src.compare_models --plot-only              # regenerate figures from saved csv
    python -m src.compare_models --no-harmonize           # skip ComBat
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch.utils.data import DataLoader

from src.data.dataset import ABIDEDataset
from src.data.download_abide import load_processed
from src.data.harmonize import fit_and_apply_combat
from src.data.preprocessing import apply_scaler, fit_scaler, vectorize
from src.data.splits import load_splits
from src.models.attention_vae import AttentionVAE
from src.models.sparse_ae import SparseAutoencoder
from src.models.vae import VariationalAutoencoder
from src.training.trainer import CKPT_DIR, evaluate, train_model

PROJECT_DIR = Path(__file__).resolve().parent.parent

# ── Reproducibility ──────────────────────────────────────────────────────────
_SEED = 42
random.seed(_SEED)
import numpy as _np_seed; _np_seed.random.seed(_SEED)
import torch as _torch_seed
_torch_seed.manual_seed(_SEED)
if _torch_seed.cuda.is_available():
    _torch_seed.cuda.manual_seed_all(_SEED)
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
FIGURES_DIR = PROJECT_DIR / "results" / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "sae": (SparseAutoencoder, "sae"),
    "vae": (VariationalAutoencoder, "vae"),
    "attention_vae": (AttentionVAE, "vae"),
}
# Ensemble is computed from the three models above — not a separate trainable model
ALL_MODEL_NAMES = list(MODELS.keys()) + ["ensemble"]


# ---------------------------------------------------------------------------
# One fold × one model
# ---------------------------------------------------------------------------

def run_fold(
    model_name: str,
    fold_idx: int,
    X_train_n: np.ndarray,
    X_test_n: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    n_epochs: int = 150,
) -> dict:
    ModelClass, model_type = MODELS[model_name]
    model = ModelClass()
    tag = f"{model_name}_fold{fold_idx}"

    train_ds = ABIDEDataset(X_train_n, y_train, augment=True, noise_std=0.02)
    test_ds  = ABIDEDataset(X_test_n,  y_test,  augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

    print(f"\n  --- {model_name} fold {fold_idx} ---")
    train_model(
        model, train_loader, test_loader,
        model_type=model_type,
        tag=tag,
        n_epochs=n_epochs,
        device=device,
    )

    # Load best checkpoint (by AUC) for final test evaluation
    ckpt = CKPT_DIR / f"{tag}_best_auc.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)

    metrics = evaluate(model, test_loader, device, model_type)
    metrics_clean = {k: v for k, v in metrics.items() if k not in ("probs", "labels")}
    metrics_clean.update({"model": model_name, "fold": fold_idx})

    # Save per-fold probs for ROC curve plotting
    np.save(FIGURES_DIR / f"probs_{tag}.npy", metrics["probs"])
    np.save(FIGURES_DIR / f"labels_{tag}.npy", metrics["labels"])

    print(
        f"  {model_name} fold {fold_idx} → AUC={metrics['auc']:.4f}  "
        f"F1={metrics['f1']:.4f}  BalAcc={metrics['balanced_acc']:.4f}"
    )
    return metrics_clean


# ---------------------------------------------------------------------------
# Full CV run
# ---------------------------------------------------------------------------

def run_cv(
    n_cv: int = 5,
    harmonize: bool = True,
    n_epochs: int = 150,
) -> pd.DataFrame:
    X_raw, pheno = load_processed()
    folds = load_splits(n_splits=n_cv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Subjects: {len(X_raw)}  |  Folds: {n_cv}\n")

    all_rows: list[dict] = []

    for k, (train_idx, test_idx) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"FOLD {k+1}/{n_cv}")
        print(f"{'='*50}")

        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train = pheno["label"].values[train_idx]
        y_test = pheno["label"].values[test_idx]

        X_train_vec = vectorize(X_train_raw)
        X_test_vec = vectorize(X_test_raw)

        # ComBat harmonization (fit on train only)
        if harmonize:
            sites_tr = pheno["site"].values[train_idx]
            sites_te = pheno["site"].values[test_idx]
            print(f"  [combat] train sites ({len(set(sites_tr))}): {sorted(set(sites_tr))}")
            print(f"  [combat] test  sites ({len(set(sites_te))}): {sorted(set(sites_te))}")
            site_counts = pd.Series(sites_tr).value_counts()
            small = site_counts[site_counts < 3]
            if not small.empty:
                print(f"  [combat] WARNING: sites with <3 train subjects: {small.to_dict()}")
            try:
                X_train_vec, X_test_vec = fit_and_apply_combat(
                    X_train_vec, X_test_vec,
                    sites_tr, sites_te,
                    pheno["age"].values[train_idx],
                    pheno["age"].values[test_idx],
                    pheno["sex"].values[train_idx],
                    pheno["sex"].values[test_idx],
                )
                print(f"  [combat] OK — train={X_train_vec.shape}  test={X_test_vec.shape}")
            except Exception as e:
                import traceback
                print(f"  [combat] FAILED on fold {k} ({type(e).__name__}: {e}):")
                traceback.print_exc()
                print("  [combat] continuing without harmonization")

        # StandardScaler (fit on train only) — save for SHAP reuse
        scaler = fit_scaler(X_train_vec)
        X_train_n, X_test_n = apply_scaler(scaler, X_train_vec, X_test_vec)
        _scaler_path = PROJECT_DIR / "data" / "processed" / f"scaler_fold{k}.pkl"
        joblib.dump(scaler, _scaler_path)
        print(f"  Scaler saved → {_scaler_path}")

        # Reshape back to (N, 200, 200) for ABIDEDataset
        # ABIDEDataset calls upper_triangle internally — pass raw FC
        # But after harmonize+scaler we have flat (N, 19900) — pass as-is
        # ABIDEDataset expects (N, 200, 200); wrap to accept flat input
        fold_probs: dict[str, np.ndarray] = {}
        for model_name in MODELS:
            row = run_fold(
                model_name, k,
                X_train_n, X_test_n,
                y_train, y_test,
                device, n_epochs,
            )
            all_rows.append(row)
            fold_probs[model_name] = np.load(
                FIGURES_DIR / f"probs_{model_name}_fold{k}.npy"
            )

        # Ensemble: average probabilities of all three models
        ens_probs  = np.mean(list(fold_probs.values()), axis=0)
        ens_labels = np.load(FIGURES_DIR / f"labels_sae_fold{k}.npy")
        ens_preds  = (ens_probs >= 0.5).astype(int)
        from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
        ens_row = {
            "model": "ensemble",
            "fold":  k,
            "auc":          float(roc_auc_score(ens_labels, ens_probs)),
            "f1":           float(f1_score(ens_labels, ens_preds, zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(ens_labels, ens_preds)),
            "acc":          float((ens_preds == ens_labels).mean()),
            "sensitivity":  float(((ens_preds==1)&(ens_labels==1)).sum() / max((ens_labels==1).sum(),1)),
            "specificity":  float(((ens_preds==0)&(ens_labels==0)).sum() / max((ens_labels==0).sum(),1)),
        }
        all_rows.append(ens_row)
        np.save(FIGURES_DIR / f"probs_ensemble_fold{k}.npy", ens_probs)
        np.save(FIGURES_DIR / f"labels_ensemble_fold{k}.npy", ens_labels)
        print(f"  ensemble fold {k} → AUC={ens_row['auc']:.4f}  F1={ens_row['f1']:.4f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(METRICS_DIR / "cv_per_fold.csv", index=False)
    _make_summary(df)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_summary(df: pd.DataFrame) -> None:
    metrics = ["auc", "f1", "balanced_acc", "sensitivity", "specificity"]
    rows = []
    for model_name in ALL_MODEL_NAMES:
        sub = df[df["model"] == model_name]
        row: dict = {"model": model_name}
        for m in metrics:
            if m in sub.columns:
                row[m] = f"{sub[m].mean():.4f} ± {sub[m].std():.4f}"
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("model")
    summary.to_csv(METRICS_DIR / "comparison_table.csv")
    print("\n" + "="*60)
    print("RESULTS  (mean ± std over 5 folds)")
    print("="*60)
    print(summary.to_string())


def plot_roc_curves(n_cv: int = 5) -> None:
    """Overlay ROC curves for all models (mean ± std across folds)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"sae": "#1f77b4", "vae": "#ff7f0e",
              "attention_vae": "#2ca02c", "ensemble": "#d62728"}

    for model_name in ALL_MODEL_NAMES:
        color = colors.get(model_name, "#888888")
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)

        for k in range(n_cv):
            tag = f"{model_name}_fold{k}"
            probs_path = FIGURES_DIR / f"probs_{tag}.npy"
            labels_path = FIGURES_DIR / f"labels_{tag}.npy"
            if not probs_path.exists():
                continue
            probs = np.load(probs_path)
            labels = np.load(labels_path)
            fpr, tpr, _ = roc_curve(labels, probs)
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(labels, probs))
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        if not tprs:
            continue

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                label=f"{model_name} (AUC={mean_auc:.3f}±{std_auc:.3f})")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=color, alpha=0.15)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — ASD vs Control (5-fold CV)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def plot_metrics_bars() -> None:
    """Bar chart of AUC / F1 / BalAcc with error bars."""
    df = pd.read_csv(METRICS_DIR / "cv_per_fold.csv")
    metrics = ["auc", "f1", "balanced_acc"]
    model_names = ALL_MODEL_NAMES
    x = np.arange(len(metrics))
    width = 0.22
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        sub = df[df["model"] == model_name]
        means = [sub[m].mean() for m in metrics]
        stds = [sub[m].std() for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, label=model_name,
               color=color, alpha=0.85, capsize=4)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(["AUC", "F1", "Balanced Acc"])
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — ASD Classification (5-fold CV)")
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "metrics_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def plot_confusion_matrices(n_cv: int = 5) -> None:
    """Confusion matrix for each model (fold with best AUC)."""
    df = pd.read_csv(METRICS_DIR / "cv_per_fold.csv")
    model_names = ALL_MODEL_NAMES
    labels_display = ["Control", "ASD"]

    fig, axes = plt.subplots(1, len(model_names), figsize=(18, 4))
    colors = {"sae": "#1f77b4", "vae": "#ff7f0e", "attention_vae": "#2ca02c"}

    for ax, model_name in zip(axes, model_names):
        # Pick fold with best AUC for this model
        sub = df[df["model"] == model_name]
        best_fold = int(sub.loc[sub["auc"].idxmax(), "fold"])
        tag = f"{model_name}_fold{best_fold}"

        probs_path = FIGURES_DIR / f"probs_{tag}.npy"
        labels_path = FIGURES_DIR / f"labels_{tag}.npy"
        if not probs_path.exists():
            ax.set_visible(False)
            continue

        probs  = np.load(probs_path)
        labels = np.load(labels_path)
        preds  = (probs >= 0.5).astype(int)
        cm = confusion_matrix(labels, preds)

        disp = ConfusionMatrixDisplay(cm, display_labels=labels_display)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        best_auc = float(sub.loc[sub["auc"].idxmax(), "auc"])
        ax.set_title(f"{model_name}\n(fold {best_fold}, AUC={best_auc:.3f})", fontsize=10)

    fig.suptitle("Confusion Matrices — Best Fold per Model")
    fig.tight_layout()
    out = FIGURES_DIR / "confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def plot_training_curves(n_cv: int = 5) -> None:
    """Loss and AUC curves per model (one fold for illustration)."""
    fig, axes = plt.subplots(2, len(MODELS), figsize=(14, 7), sharex=False)
    colors = {"sae": "#1f77b4", "vae": "#ff7f0e", "attention_vae": "#2ca02c"}

    for col, model_name in enumerate(MODELS):
        # Use fold 0 history for illustration
        hist_path = CKPT_DIR / f"{model_name}_fold0_history.json"
        if not hist_path.exists():
            continue
        with open(hist_path) as f:
            hist = json.load(f)

        c = colors[model_name]
        axes[0, col].plot(hist["train_loss"], color=c)
        axes[0, col].set_title(f"{model_name}\ntrain loss")
        axes[0, col].set_xlabel("epoch")
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(hist["val_auc"], color=c)
        axes[1, col].set_title("val AUC")
        axes[1, col].set_xlabel("epoch")
        axes[1, col].set_ylim(0.4, 1.0)
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle("Training curves (fold 0)", y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--no-harmonize", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot_roc_curves(args.cv)
        plot_metrics_bars()
        plot_confusion_matrices(args.cv)
        plot_training_curves(args.cv)
    else:
        run_cv(n_cv=args.cv, harmonize=not args.no_harmonize, n_epochs=args.epochs)
        plot_roc_curves(args.cv)
        plot_metrics_bars()
        plot_confusion_matrices(args.cv)
        plot_training_curves(args.cv)
