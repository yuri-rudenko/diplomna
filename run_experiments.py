"""
Experiment runner: trains all three models under each named configuration
and saves results to results/experiments/{name}/.

Usage:
    python run_experiments.py                            # run all experiments
    python run_experiments.py --exp baseline aug_full    # run specific ones
    python run_experiments.py --compare                  # compare all saved
    python run_experiments.py --list                     # list experiments
    python run_experiments.py --cv 2 --epochs 10         # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataset import ABIDEDataset, mixup_batch
from src.data.download_abide import load_processed
from src.data.harmonize import fit_and_apply_combat
from src.data.preprocessing import apply_scaler, fit_scaler, vectorize
from src.data.splits import load_splits
from src.experiments import EXPERIMENTS, ExperimentConfig
from src.models.attention_vae import AttentionVAE
from src.models.sparse_ae import SparseAutoencoder
from src.models.vae import VariationalAutoencoder
from src.training.schedules import beta_schedule, cls_weight_schedule
from src.training.trainer import evaluate

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

RESULTS_DIR = PROJECT_DIR / "results" / "experiments"

# ── Global reproducibility seed ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Environment detection ─────────────────────────────────────────────────────
def _detect_env() -> str:
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    if os.path.exists("/content"):
        return "colab"
    return "local"


def _env_info() -> dict:
    return {
        "env":        _detect_env(),
        "python":     platform.python_version(),
        "torch":      torch.__version__,
        "cuda":       torch.cuda.is_available(),
        "device":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "platform":   platform.system(),
        "seed":       SEED,
    }


# ---------------------------------------------------------------------------
# Per-experiment directories
# ---------------------------------------------------------------------------

def exp_dirs(exp_name: str) -> tuple[Path, Path, Path]:
    base    = RESULTS_DIR / exp_name
    ckpts   = base / "checkpoints"
    metrics = base / "metrics"
    figs    = base / "figures"
    for d in (ckpts, metrics, figs):
        d.mkdir(parents=True, exist_ok=True)
    return ckpts, metrics, figs


# ---------------------------------------------------------------------------
# Model factory — respects config hidden_dims / latent_dim / dropout
# ---------------------------------------------------------------------------

def make_model(model_name: str, cfg: ExperimentConfig) -> tuple[nn.Module, str]:
    kwargs = dict(
        hidden_dims=cfg.hidden_dims,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
    )
    if model_name == "sae":
        return SparseAutoencoder(**kwargs), "sae"
    elif model_name == "vae":
        return VariationalAutoencoder(**kwargs), "vae"
    else:  # attention_vae
        return AttentionVAE(**kwargs), "vae"


# ---------------------------------------------------------------------------
# Training loop (inline, config-aware)
# ---------------------------------------------------------------------------

def _forward(model, X, y, model_type, beta, cls_weight,
             label_smoothing, y_b=None, lam=1.0):
    if model_type == "sae":
        x_recon, z, logits = model(X)
        recon = F.mse_loss(x_recon, X)
        sp    = model.sparse_loss(z)
        if cls_weight > 0:
            cls_a = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            cls_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing) \
                    if y_b is not None else cls_a
            cls = lam * cls_a + (1 - lam) * cls_b
        else:
            cls = torch.tensor(0.0, device=X.device)
        total = recon + cls_weight * cls + model.sparsity_weight * sp
    else:
        x_recon, mu, logvar, z, logits = model(X)
        recon = F.mse_loss(x_recon, X)
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if cls_weight > 0:
            cls_a = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            cls_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing) \
                    if y_b is not None else cls_a
            cls = lam * cls_a + (1 - lam) * cls_b
        else:
            cls = torch.tensor(0.0, device=X.device)
        total = recon + beta * kl + cls_weight * cls
    return total, logits


def train_experiment(
    model: nn.Module,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    tag: str,
    ckpt_dir: Path,
    device: torch.device,
) -> dict:
    model = model.to(device)

    OptimizerClass = AdamW if cfg.optimizer == "adamw" else Adam

    def _make_opt():
        return OptimizerClass(model.parameters(),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)

    optimizer = _make_opt()
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.phase1_epochs,
                                  eta_min=cfg.lr * 0.1)

    best_auc = best_f1 = 0.0
    patience_ctr = 0
    phase2 = False
    history: dict = {"train_loss": [], "val_auc": [], "val_f1": [], "val_balanced_acc": []}

    bar = tqdm(range(cfg.n_epochs), desc=tag, unit="ep",
               dynamic_ncols=True, leave=True)
    t0 = time.time()

    for epoch in bar:
        beta = beta_schedule(epoch, warmup_epochs=cfg.kl_warmup_epochs)
        cw   = cls_weight_schedule(epoch, phase1_epochs=cfg.phase1_epochs,
                                   cls_weight_max=cfg.cls_weight_max)

        if epoch == cfg.phase1_epochs and not phase2:
            phase2 = True
            optimizer = _make_opt()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)
            patience_ctr = 0
            best_auc = 0.0

        # ── train one epoch ────────────────────────────────────────────────
        model.train()
        total_loss = correct = n = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            y_b2, lam = None, 1.0
            if cfg.use_mixup and cw > 0:
                X_b, y_b, y_b2, lam = mixup_batch(X_b, y_b, cfg.mixup_alpha)
                y_b2 = y_b2.to(device)

            loss, logits = _forward(model, X_b, y_b, model_type, beta, cw,
                                    cfg.label_smoothing, y_b2, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == y_b).sum().item()
            n += len(y_b)

        train_loss = total_loss / len(train_loader)

        # ── validate ───────────────────────────────────────────────────────
        val = evaluate(model, val_loader, device, model_type)

        if phase2:
            scheduler.step(val["auc"])
        else:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val["auc"])
        history["val_f1"].append(val["f1"])
        history["val_balanced_acc"].append(val["balanced_acc"])

        lr_cur = optimizer.param_groups[0]["lr"]
        phase  = "recon" if cw == 0 else f"joint(cw={cw:.1f})"
        mark   = " ★" if val["auc"] >= best_auc else ""
        bar.set_postfix(loss=f"{train_loss:.4f}",
                        auc=f"{val['auc']:.4f}{mark}",
                        phase=phase, lr=f"{lr_cur:.0e}",
                        p=f"{patience_ctr}/{cfg.patience}")

        if val["auc"] > best_auc:
            best_auc = val["auc"]
            torch.save(model.state_dict(), ckpt_dir / f"{tag}_best_auc.pth")
            patience_ctr = 0
        elif phase2:
            patience_ctr += 1

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save(model.state_dict(), ckpt_dir / f"{tag}_best_f1.pth")

        if patience_ctr >= cfg.patience:
            bar.set_description(f"{tag} [early stop]")
            bar.close()
            break

    train_time = time.time() - t0
    tqdm.write(f"  ✓ {tag} | AUC={best_auc:.4f}  F1={best_f1:.4f}  "
               f"time={train_time:.0f}s")

    history["final_epoch"]   = epoch          # last epoch index (0-based)
    history["n_epochs_run"]  = epoch + 1      # how many epochs actually ran
    history["train_time_s"]  = round(train_time, 1)
    history["best_val_auc"]  = best_auc
    history["best_val_f1"]   = best_f1

    with open(ckpt_dir / f"{tag}_history.json", "w") as f:
        json.dump(history, f)

    return history


# ---------------------------------------------------------------------------
# Run one full experiment (all models, all folds)
# ---------------------------------------------------------------------------

MODEL_NAMES = ["sae", "vae", "attention_vae"]


def run_experiment(cfg: ExperimentConfig, n_cv: int = 5) -> pd.DataFrame:
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT: {cfg.name}")
    print(f"  {cfg.description}")
    print(f"{'#'*60}")

    ckpt_dir, metrics_dir, figs_dir = exp_dirs(cfg.name)
    X_raw, pheno = load_processed()
    folds  = load_splits(n_splits=n_cv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}  Subjects: {len(X_raw)}  Folds: {n_cv}")

    # ── Save config.json for full reproducibility ─────────────────────────────
    config_path = metrics_dir / "config.json"
    # Count model parameters
    def _count_params(model_cls, kwargs):
        m = model_cls(**kwargs)
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    arch_kwargs = dict(hidden_dims=cfg.hidden_dims, latent_dim=cfg.latent_dim, dropout=cfg.dropout)
    model_params = {
        "sae":           _count_params(SparseAutoencoder, arch_kwargs),
        "vae":           _count_params(VariationalAutoencoder, arch_kwargs),
        "attention_vae": _count_params(AttentionVAE, arch_kwargs),
    }

    config_data = {
        "experiment":   asdict(cfg),
        "n_cv":         n_cv,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment":  _env_info(),
        "model_params": model_params,
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"  Config saved → {config_path}")
    print(f"  Model params — SAE: {model_params['sae']:,}  "
          f"VAE: {model_params['vae']:,}  "
          f"AttVAE: {model_params['attention_vae']:,}")

    all_rows: list[dict] = []

    for k, (train_idx, test_idx) in enumerate(folds):
        print(f"\n  {'='*50}")
        print(f"  FOLD {k+1}/{n_cv}")
        print(f"  {'='*50}")

        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train = pheno["label"].values[train_idx]
        y_test  = pheno["label"].values[test_idx]

        X_train_vec = vectorize(X_train_raw)
        X_test_vec  = vectorize(X_test_raw)

        # ComBat
        if cfg.harmonize:
            try:
                sites_tr = pheno["site"].values[train_idx]
                sites_te = pheno["site"].values[test_idx]
                X_train_vec, X_test_vec = fit_and_apply_combat(
                    X_train_vec, X_test_vec,
                    sites_tr, sites_te,
                    pheno["age"].values[train_idx], pheno["age"].values[test_idx],
                    pheno["sex"].values[train_idx], pheno["sex"].values[test_idx],
                )
                print(f"  [combat] OK — train={X_train_vec.shape} test={X_test_vec.shape}")
            except Exception:
                print("  [combat] FAILED:")
                traceback.print_exc()
                print("  [combat] continuing without harmonization")

        scaler = fit_scaler(X_train_vec)
        X_train_n, X_test_n = apply_scaler(scaler, X_train_vec, X_test_vec)

        # Save scaler — needed by shap_explain.py
        import joblib as _jl
        _scaler_path = ckpt_dir / f"scaler_fold{k}.pkl"
        _jl.dump(scaler, _scaler_path)

        # Save fold metadata (train/test sizes, ASD balance)
        y_all = pheno["label"].values
        fold_meta = {
            "fold":       k,
            "train_size": int(len(train_idx)),
            "test_size":  int(len(test_idx)),
            "train_asd_pct": float(y_all[train_idx].mean()),
            "test_asd_pct":  float(y_all[test_idx].mean()),
            "train_sites":   sorted(pheno["site"].iloc[train_idx].unique().tolist()),
            "test_sites":    sorted(pheno["site"].iloc[test_idx].unique().tolist()),
        }
        all_fold_meta = json.loads(
            (metrics_dir / "fold_info.json").read_text()
        ) if (metrics_dir / "fold_info.json").exists() else []
        all_fold_meta.append(fold_meta)
        with open(metrics_dir / "fold_info.json", "w") as _f:
            json.dump(all_fold_meta, _f, indent=2)

        fold_probs: dict[str, np.ndarray] = {}

        for model_name in MODEL_NAMES:
            tag = f"{cfg.name}_{model_name}_fold{k}"
            model, model_type = make_model(model_name, cfg)

            train_ds = ABIDEDataset(X_train_n, y_train,
                                    augment=cfg.augment, noise_std=cfg.noise_std)
            test_ds  = ABIDEDataset(X_test_n,  y_test,  augment=False)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
            test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

            print(f"\n  --- {model_name} fold {k} ---")
            train_experiment(model, model_type, train_loader, test_loader,
                             cfg, tag, ckpt_dir, device)

            # Load best checkpoint for evaluation
            ckpt = ckpt_dir / f"{tag}_best_auc.pth"
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.to(device)

            metrics = evaluate(model, test_loader, device, model_type)
            row = {k2: v for k2, v in metrics.items()
                   if k2 not in ("probs", "labels")}
            row.update({"experiment": cfg.name, "model": model_name, "fold": k})
            all_rows.append(row)

            np.save(figs_dir / f"probs_{tag}.npy",  metrics["probs"])
            np.save(figs_dir / f"labels_{tag}.npy", metrics["labels"])
            fold_probs[model_name] = metrics["probs"]

            print(f"  {model_name} fold {k} → "
                  f"AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
                  f"BalAcc={metrics['balanced_acc']:.4f}")

        # Ensemble
        ens_probs  = np.mean(list(fold_probs.values()), axis=0)
        ens_labels = np.load(figs_dir / f"labels_{cfg.name}_sae_fold{k}.npy")
        ens_preds  = (ens_probs >= 0.5).astype(int)
        from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
        ens_row = {
            "experiment": cfg.name, "model": "ensemble", "fold": k,
            "auc":          float(roc_auc_score(ens_labels, ens_probs)),
            "f1":           float(f1_score(ens_labels, ens_preds, zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(ens_labels, ens_preds)),
            "acc":          float((ens_preds == ens_labels).mean()),
            "sensitivity":  float(((ens_preds==1)&(ens_labels==1)).sum()
                                  / max((ens_labels==1).sum(), 1)),
            "specificity":  float(((ens_preds==0)&(ens_labels==0)).sum()
                                  / max((ens_labels==0).sum(), 1)),
        }
        all_rows.append(ens_row)
        np.save(figs_dir / f"probs_{cfg.name}_ensemble_fold{k}.npy",  ens_probs)
        np.save(figs_dir / f"labels_{cfg.name}_ensemble_fold{k}.npy", ens_labels)
        print(f"  ensemble fold {k} → AUC={ens_row['auc']:.4f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(metrics_dir / "cv_per_fold.csv", index=False)
    _save_summary(df, metrics_dir, cfg.name)
    return df


def _save_summary(df: pd.DataFrame, metrics_dir: Path, exp_name: str) -> None:
    metric_cols = ["auc", "f1", "balanced_acc", "sensitivity", "specificity"]
    rows = []
    for model_name in MODEL_NAMES + ["ensemble"]:
        sub = df[df["model"] == model_name]
        if sub.empty:
            continue
        row: dict = {"experiment": exp_name, "model": model_name}
        for m in metric_cols:
            if m in sub.columns:
                row[m] = f"{sub[m].mean():.4f} ± {sub[m].std():.4f}"
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(metrics_dir / "summary.csv", index=False)
    print(f"\n  Summary saved → {metrics_dir / 'summary.csv'}")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Compare all saved experiments
# ---------------------------------------------------------------------------

def compare_all() -> pd.DataFrame:
    """Read all summary.csv files and merge into one ranked comparison table."""
    if not RESULTS_DIR.exists():
        print("No experiments directory found.")
        return pd.DataFrame()

    rows = []
    per_fold_rows = []

    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        summary_path = exp_dir / "metrics" / "summary.csv"
        per_fold_path = exp_dir / "metrics" / "cv_per_fold.csv"
        if summary_path.exists():
            rows.append(pd.read_csv(summary_path))
        if per_fold_path.exists():
            pf = pd.read_csv(per_fold_path)
            pf["experiment"] = exp_dir.name
            per_fold_rows.append(pf)

    if not rows:
        print("No experiment results found. Run experiments first.")
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    out = RESULTS_DIR / "comparison_all.csv"
    df.to_csv(out, index=False)

    # Also write merged per-fold data to results/metrics/ for significance.py
    if per_fold_rows:
        all_pf = pd.concat(per_fold_rows, ignore_index=True)
        global_metrics_dir = PROJECT_DIR / "results" / "metrics"
        global_metrics_dir.mkdir(parents=True, exist_ok=True)
        all_pf.to_csv(global_metrics_dir / "cv_per_fold_all_experiments.csv", index=False)

        # Also write the baseline experiment's folds as the default significance.py target
        baseline_pf = all_pf[all_pf["experiment"] == "baseline"] if "baseline" in all_pf["experiment"].values else all_pf
        baseline_pf.to_csv(global_metrics_dir / "cv_per_fold.csv", index=False)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS — ensemble AUC ranking")
    print("=" * 80)
    ens = df[df["model"] == "ensemble"].copy()
    ens["auc_mean"] = ens["auc"].str.split(" ").str[0].astype(float)
    ens = ens.sort_values("auc_mean", ascending=False)
    print(ens[["experiment", "auc", "f1", "balanced_acc"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS — per model breakdown")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nSaved → {out}")
    return df


# ---------------------------------------------------------------------------
# Plot-only: regenerate figures from saved npy/csv (no retraining needed)
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works on Kaggle/Colab/headless
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


def plot_experiment(exp_name: str, n_cv: int = 5) -> None:
    """Regenerate all figures for one experiment from saved .npy/.json/.csv files."""
    base      = RESULTS_DIR / exp_name
    figs_dir  = base / "figures"
    ckpt_dir  = base / "checkpoints"
    metrics_dir = base / "metrics"

    if not figs_dir.exists():
        print(f"  [skip] {exp_name}: figures dir not found ({figs_dir})")
        return

    out_dir = base / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_names = MODEL_NAMES + ["ensemble"]
    colors = {"sae": "#1f77b4", "vae": "#ff7f0e",
               "attention_vae": "#2ca02c", "ensemble": "#d62728"}

    # ── 1. ROC curves ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    any_roc = False
    for mname in model_names:
        tprs, aucs, mean_fpr = [], [], np.linspace(0, 1, 100)
        for k in range(n_cv):
            tag = f"{exp_name}_{mname}_fold{k}"
            p = figs_dir / f"probs_{tag}.npy"
            l = figs_dir / f"labels_{tag}.npy"
            if not p.exists():
                continue
            probs  = np.load(p)
            labels = np.load(l)
            fpr, tpr, _ = roc_curve(labels, probs)
            aucs.append(roc_auc_score(labels, probs))
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        if not tprs:
            continue
        any_roc = True
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr  = np.std(tprs, axis=0)
        c = colors.get(mname, "#888")
        ax.plot(mean_fpr, mean_tpr, c=c, lw=2,
                label=f"{mname} (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=c, alpha=0.12)
    if any_roc:
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {exp_name} ({n_cv}-fold CV)")
        ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
        print(f"  Saved roc_curves.png")
    plt.close(fig)

    # ── 2. Metrics bar chart ─────────────────────────────────────────────────
    per_fold_csv = metrics_dir / "cv_per_fold.csv"
    if per_fold_csv.exists():
        df = pd.read_csv(per_fold_csv)
        metrics_cols = ["auc", "f1", "balanced_acc"]
        x = np.arange(len(metrics_cols))
        width = 0.22
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for i, mname in enumerate(model_names):
            sub = df[df["model"] == mname]
            if sub.empty:
                continue
            means = [sub[m].mean() for m in metrics_cols]
            stds  = [sub[m].std()  for m in metrics_cols]
            ax2.bar(x + i * width, means, width, yerr=stds,
                    label=mname, color=colors.get(mname, "#888"),
                    alpha=0.85, capsize=4)
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(["AUC", "F1", "Balanced Acc"])
        ax2.set_ylabel("Score"); ax2.set_ylim(0.4, 1.0)
        ax2.set_title(f"Metrics — {exp_name}")
        ax2.legend(); ax2.grid(axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(out_dir / "metrics_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved metrics_bars.png")

    # ── 3. Confusion matrices ────────────────────────────────────────────────
    if per_fold_csv.exists():
        df = pd.read_csv(per_fold_csv)
        plotable = [m for m in model_names
                    if not df[df["model"] == m].empty]
        fig3, axes = plt.subplots(1, len(plotable), figsize=(4.5 * len(plotable), 4))
        if len(plotable) == 1:
            axes = [axes]
        for ax3, mname in zip(axes, plotable):
            sub = df[df["model"] == mname]
            best_fold = int(sub.loc[sub["auc"].idxmax(), "fold"])
            tag = f"{exp_name}_{mname}_fold{best_fold}"
            p = figs_dir / f"probs_{tag}.npy"
            l = figs_dir / f"labels_{tag}.npy"
            if not p.exists():
                ax3.set_visible(False); continue
            probs  = np.load(p); labels = np.load(l)
            preds  = (probs >= 0.5).astype(int)
            cm     = confusion_matrix(labels, preds)
            ConfusionMatrixDisplay(cm, display_labels=["Control", "ASD"]).plot(
                ax=ax3, colorbar=False, cmap="Blues")
            ax3.set_title(f"{mname}\n(fold {best_fold}, "
                           f"AUC={sub.loc[sub['auc'].idxmax(),'auc']:.3f})", fontsize=9)
        fig3.suptitle(f"Confusion Matrices — {exp_name}")
        fig3.tight_layout()
        fig3.savefig(out_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved confusion_matrices.png")

    # ── 4. Training curves ───────────────────────────────────────────────────
    hist_models = [m for m in MODEL_NAMES
                   if (ckpt_dir / f"{exp_name}_{m}_fold0_history.json").exists()]
    if hist_models:
        fig4, axes4 = plt.subplots(2, len(hist_models),
                                    figsize=(5 * len(hist_models), 7))
        if len(hist_models) == 1:
            axes4 = [[axes4[0]], [axes4[1]]]
        for col, mname in enumerate(hist_models):
            hist_path = ckpt_dir / f"{exp_name}_{mname}_fold0_history.json"
            with open(hist_path) as f:
                hist = json.load(f)
            c = colors.get(mname, "#888")
            axes4[0][col].plot(hist["train_loss"], color=c)
            axes4[0][col].set_title(f"{mname}\ntrain loss"); axes4[0][col].grid(True, alpha=0.3)
            axes4[1][col].plot(hist["val_auc"], color=c)
            axes4[1][col].set_title("val AUC"); axes4[1][col].set_ylim(0.4, 1.0)
            axes4[1][col].grid(True, alpha=0.3)
        fig4.suptitle(f"Training curves — {exp_name} (fold 0)")
        fig4.tight_layout()
        fig4.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig4)
        print(f"  Saved training_curves.png")

    print(f"  All figures → {out_dir}")


def plot_all_experiments(n_cv: int = 5) -> None:
    """Regenerate figures for every saved experiment directory."""
    if not RESULTS_DIR.exists():
        print("No experiments directory found.")
        return
    exp_names = [d.name for d in sorted(RESULTS_DIR.iterdir())
                 if d.is_dir() and (d / "figures").exists()]
    if not exp_names:
        print("No completed experiments found in", RESULTS_DIR)
        return
    print(f"Regenerating figures for {len(exp_names)} experiments: {exp_names}")
    for name in exp_names:
        print(f"\n── {name} ──")
        plot_experiment(name, n_cv=n_cv)
    compare_all()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run named ablation experiments and compare results."
    )
    parser.add_argument("--exp", nargs="+", default=None,
                        help="Experiment names to run (default: all)")
    parser.add_argument("--cv",     type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override n_epochs for all experiments")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all saved experiment results")
    parser.add_argument("--list",    action="store_true",
                        help="List available experiment names")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate all figures from saved data (no training)")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        for name, cfg in EXPERIMENTS.items():
            print(f"  {name:<20s} — {cfg.description}")
        sys.exit(0)

    if args.compare:
        compare_all()
        sys.exit(0)

    if args.plot_only:
        names = args.exp if args.exp else None
        if names:
            for n in names:
                print(f"\n── Plotting {n} ──")
                plot_experiment(n, n_cv=args.cv)
        else:
            plot_all_experiments(n_cv=args.cv)
        sys.exit(0)

    # Select experiments to run
    names = args.exp if args.exp else list(EXPERIMENTS.keys())
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print(f"Unknown experiments: {unknown}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    env = _detect_env()
    print(f"\nEnvironment : {env}")
    print(f"Running     : {len(names)} experiment(s): {names}")
    print(f"CV folds    : {args.cv}  |  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Seed        : {SEED}\n")

    failed = []
    for name in names:
        cfg = EXPERIMENTS[name]
        if args.epochs is not None:
            cfg.n_epochs = args.epochs
        try:
            run_experiment(cfg, n_cv=args.cv)
        except Exception:
            print(f"\n[ERROR] Experiment '{name}' failed:")
            traceback.print_exc()
            failed.append(name)

    if failed:
        print(f"\nFailed experiments: {failed}")

    print("\n\nAll done. Next steps:")
    print("  python run_experiments.py --compare")
    print("  python run_experiments.py --plot-only")
