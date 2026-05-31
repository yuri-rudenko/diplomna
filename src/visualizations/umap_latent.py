"""
UMAP projection of the latent space (mu) for trained VAE and Attention-VAE.

Loads best-fold checkpoints, extracts mu for all test subjects, projects
to 2D with UMAP, and saves scatter plots colored by ASD/Control.

Usage:
    python -m src.visualizations.umap_latent                      # both models, fold 0
    python -m src.visualizations.umap_latent --models vae         # only VAE
    python -m src.visualizations.umap_latent --fold best          # auto-pick best fold
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_DIR / "results" / "figures"
CKPT_DIR    = PROJECT_DIR / "results" / "checkpoints"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "vae":           ("vae",           "VariationalAutoencoder"),
    "attention_vae": ("attention_vae", "AttentionVAE"),
}

COLORS = {0: "#2196F3", 1: "#F44336"}   # Control: blue, ASD: red
LABELS = {0: "Control", 1: "ASD"}


# ---------------------------------------------------------------------------
# Extract mu from a trained model
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_mu(
    model,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu_array, labels_array) for all batches in loader."""
    model.eval()
    all_mu, all_y = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        if model_type == "sae":
            z = model.encode(X_batch)          # SAE: single latent vector
            all_mu.append(z.cpu().numpy())
        else:
            mu, _ = model.encode(X_batch)      # VAE / Attention-VAE
            all_mu.append(mu.cpu().numpy())
        all_y.append(y_batch.numpy())
    return np.concatenate(all_mu), np.concatenate(all_y)


# ---------------------------------------------------------------------------
# UMAP + scatter
# ---------------------------------------------------------------------------

def plot_umap_latent(
    model_name: str,
    fold,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    exp_name: str | None = None,
) -> None:
    import umap as umap_lib

    from src.data.download_abide import load_processed
    from src.data.preprocessing import apply_scaler, fit_scaler, vectorize
    from src.data.splits import load_splits
    from src.data.dataset import ABIDEDataset
    from src.utils.checkpoints import resolve_checkpoint, resolve_fold, resolve_scaler

    model_type, class_name = MODEL_CONFIGS[model_name]
    fold_idx = resolve_fold(fold, model_type, exp_name)

    # ---- load data & scaler ------------------------------------------------
    X_raw, pheno = load_processed()
    folds = load_splits()
    train_idx, test_idx = folds[fold_idx]

    X_train_vec = vectorize(X_raw[train_idx])
    X_test_vec  = vectorize(X_raw[test_idx])
    y_test = pheno["label"].values[test_idx]

    import joblib
    _scaler_path = resolve_scaler(fold_idx, exp_name)
    if _scaler_path is not None:
        scaler = joblib.load(_scaler_path)
    else:
        scaler = fit_scaler(X_train_vec)

    _, X_test_n = apply_scaler(scaler, X_train_vec, X_test_vec)

    test_ds     = ABIDEDataset(X_test_n, y_test)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # ---- load model --------------------------------------------------------
    device = torch.device("cpu")

    if class_name == "VariationalAutoencoder":
        from src.models.vae import VariationalAutoencoder
        model = VariationalAutoencoder()
    else:
        from src.models.attention_vae import AttentionVAE
        model = AttentionVAE()

    ckpt = resolve_checkpoint(model_name, fold_idx, exp_name)
    if ckpt is None:
        print(f"  [skip] checkpoint not found for {model_name} fold {fold_idx} "
              f"(exp={exp_name})")
        return
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)

    print(f"  Extracting mu for {model_name} fold {fold_idx} ...")
    mu_arr, y_arr = extract_mu(model, test_loader, device, model_type)
    print(f"  mu shape: {mu_arr.shape}  → running UMAP ...")

    # ---- UMAP --------------------------------------------------------------
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    emb = reducer.fit_transform(mu_arr)   # (N_test, 2)

    # ---- scatter plot ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in (0, 1):
        mask = y_arr == label
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=COLORS[label], label=LABELS[label],
            alpha=0.65, s=30, linewidths=0,
        )

    ax.set_title(
        f"UMAP of latent space — {model_name} (fold {fold_idx})\n"
        f"n_test={len(y_arr)}, latent_dim={mu_arr.shape[1]}",
        fontsize=11,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=1.5, frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out = FIGURES_DIR / f"umap_latent_{model_name}_fold{fold_idx}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Combined 2-panel figure: VAE + Attention-VAE side by side
# ---------------------------------------------------------------------------

def plot_umap_combined(fold, exp_name: str | None = None) -> None:
    """Side-by-side UMAP for VAE and Attention-VAE on the same fold."""
    import umap as umap_lib
    import joblib

    from src.data.download_abide import load_processed
    from src.data.preprocessing import apply_scaler, fit_scaler, vectorize
    from src.data.splits import load_splits
    from src.data.dataset import ABIDEDataset
    from src.models.vae import VariationalAutoencoder
    from src.models.attention_vae import AttentionVAE
    from src.utils.checkpoints import resolve_checkpoint, resolve_fold, resolve_scaler

    # Both panels share one fold; resolve 'best' against the VAE's metrics.
    fold_idx = resolve_fold(fold, "vae", exp_name)

    X_raw, pheno = load_processed()
    folds = load_splits()
    train_idx, test_idx = folds[fold_idx]

    X_train_vec = vectorize(X_raw[train_idx])
    X_test_vec  = vectorize(X_raw[test_idx])
    y_test = pheno["label"].values[test_idx]

    _scaler_path = resolve_scaler(fold_idx, exp_name)
    scaler = joblib.load(_scaler_path) if _scaler_path is not None else fit_scaler(X_train_vec)
    _, X_test_n = apply_scaler(scaler, X_train_vec, X_test_vec)

    test_ds     = ABIDEDataset(X_test_n, y_test)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    device      = torch.device("cpu")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    model_pairs = [
        ("vae",           VariationalAutoencoder(), axes[0]),
        ("attention_vae", AttentionVAE(),           axes[1]),
    ]

    for name, model, ax in model_pairs:
        ckpt = resolve_checkpoint(name, fold_idx, exp_name)
        if ckpt is None:
            ax.set_visible(False)
            continue

        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        mu_arr, y_arr = extract_mu(model, test_loader, device, "vae")

        reducer = umap_lib.UMAP(n_components=2, n_neighbors=15,
                                min_dist=0.1, random_state=42)
        emb = reducer.fit_transform(mu_arr)

        for label in (0, 1):
            mask = y_arr == label
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=COLORS[label], label=LABELS[label],
                       alpha=0.65, s=28, linewidths=0)

        ax.set_title(f"{name}\nlatent_dim={mu_arr.shape[1]}", fontsize=11)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(markerscale=1.4, frameon=True)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"UMAP latent space — fold {fold_idx}", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / f"umap_latent_combined_fold{fold_idx}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined UMAP → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=["vae", "attention_vae"],
        choices=["vae", "attention_vae"],
    )
    parser.add_argument("--fold", default="0",
                        help="Fold index or 'best' (default: 0)")
    parser.add_argument("--exp", default=None,
                        help="Experiment name from run_experiments.py (e.g. aug_full). "
                             "Resolves checkpoints/scalers from "
                             "results/experiments/{exp}/checkpoints/")
    parser.add_argument("--combined", action="store_true",
                        help="Also generate side-by-side combined figure")
    args = parser.parse_args()

    for m in args.models:
        plot_umap_latent(m, args.fold, exp_name=args.exp)

    if args.combined or len(args.models) >= 2:
        plot_umap_combined(args.fold, exp_name=args.exp)
