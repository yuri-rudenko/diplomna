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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from src.data.dataset import ABIDEDataset
from src.data.download_abide import load_processed
from src.data.harmonize import apply_combat, fit_combat
from src.data.preprocessing import apply_scaler, fit_scaler, vectorize
from src.data.splits import load_splits
from src.models.attention_vae import AttentionVAE
from src.models.sparse_ae import SparseAutoencoder
from src.models.vae import VariationalAutoencoder
from src.training.trainer import CKPT_DIR, evaluate, train_model

PROJECT_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
FIGURES_DIR = PROJECT_DIR / "results" / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "sae": (SparseAutoencoder, "sae"),
    "vae": (VariationalAutoencoder, "vae"),
    "attention_vae": (AttentionVAE, "vae"),
}


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

    train_ds = ABIDEDataset(X_train_n, y_train)
    test_ds = ABIDEDataset(X_test_n, y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

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
            try:
                sites_tr = pheno["site"].values[train_idx]
                sites_te = pheno["site"].values[test_idx]
                X_train_vec, combat_params = fit_combat(
                    X_train_vec, sites_tr,
                    pheno["age"].values[train_idx],
                    pheno["sex"].values[train_idx],
                )
                X_test_vec = apply_combat(
                    combat_params, X_test_vec, sites_te,
                    pheno["age"].values[test_idx],
                    pheno["sex"].values[test_idx],
                )
                print("  ComBat harmonization applied")
            except ImportError:
                print("  [warn] neuroCombat not installed — skipping harmonization")
            except Exception as e:
                print(f"  [warn] ComBat failed: {e} — skipping harmonization")

        # StandardScaler (fit on train only)
        scaler = fit_scaler(X_train_vec)
        X_train_n, X_test_n = apply_scaler(scaler, X_train_vec, X_test_vec)

        # Reshape back to (N, 200, 200) for ABIDEDataset
        # ABIDEDataset calls upper_triangle internally — pass raw FC
        # But after harmonize+scaler we have flat (N, 19900) — pass as-is
        # ABIDEDataset expects (N, 200, 200); wrap to accept flat input
        for model_name in MODELS:
            row = run_fold(
                model_name, k,
                X_train_n, X_test_n,
                y_train, y_test,
                device, n_epochs,
            )
            all_rows.append(row)

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
    for model_name in MODELS:
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
    colors = {"sae": "#1f77b4", "vae": "#ff7f0e", "attention_vae": "#2ca02c"}

    for model_name, color in colors.items():
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
    model_names = list(MODELS.keys())
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        sub = df[df["model"] == model_name]
        means = [sub[m].mean() for m in metrics]
        stds = [sub[m].std() for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, label=model_name,
               color=color, alpha=0.85, capsize=4)

    ax.set_xticks(x + width)
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
        plot_training_curves(args.cv)
    else:
        run_cv(n_cv=args.cv, harmonize=not args.no_harmonize, n_epochs=args.epochs)
        plot_roc_curves(args.cv)
        plot_metrics_bars()
        plot_training_curves(args.cv)
