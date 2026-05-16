"""
Ablation: Two-stage VAE training.

Stage 1: Train pure VAE (no classifier, only recon + KL).
Stage 2: Freeze encoder, train MLP classifier on frozen mu.

Usage:
    python -m src.training.two_stage --cv 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.dataset import ABIDEDataset
from src.data.download_abide import load_processed
from src.data.harmonize import fit_and_apply_combat
from src.data.preprocessing import apply_scaler, fit_scaler, vectorize
from src.data.splits import load_splits
from src.models.vae import VariationalAutoencoder
from src.training.schedules import beta_schedule
from src.training.trainer import CKPT_DIR, evaluate

METRICS_DIR = Path(__file__).resolve().parents[2] / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


class FrozenEncoderClassifier(nn.Module):
    """Thin MLP on top of a frozen VAE encoder."""

    def __init__(self, vae: VariationalAutoencoder) -> None:
        super().__init__()
        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(vae.latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, _ = self.vae.encode(x)
        return self.classifier(mu)


def train_stage1(
    vae: VariationalAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tag: str,
    device: torch.device,
    n_epochs: int = 80,
    lr: float = 1e-3,
    kl_warmup: int = 30,
) -> None:
    """Train VAE on reconstruction + KL only (no classifier)."""
    vae.train()
    optimizer = Adam(vae.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    best_recon = float("inf")

    for epoch in range(n_epochs):
        vae.train()
        total_loss = 0.0
        beta = beta_schedule(epoch, warmup_epochs=kl_warmup)

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, _, logits = vae(X_batch)
            loss, *_ = vae.loss(
                X_batch, x_recon, mu, logvar, logits, y_batch.to(device),
                beta=beta, recon_weight=1.0, cls_weight=0.0,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_recon:
            best_recon = avg_loss
            torch.save(vae.state_dict(), CKPT_DIR / f"{tag}_stage1.pth")

        if epoch % 10 == 0:
            print(f"  [stage1/{tag}] epoch {epoch:3d} | recon+KL loss={avg_loss:.4f} (β={beta:.2f})")


def train_stage2(
    frozen_cls: FrozenEncoderClassifier,
    train_loader: DataLoader,
    tag: str,
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-3,
) -> None:
    """Train only the classifier head on frozen encoder."""
    optimizer = Adam(frozen_cls.classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        frozen_cls.train()
        total_loss = correct = n = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = frozen_cls(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == y_batch).sum().item()
            n += len(y_batch)

        if epoch % 10 == 0:
            print(f"  [stage2/{tag}] epoch {epoch:3d} | cls_loss={total_loss/len(train_loader):.4f} | acc={correct/n:.3f}")

    torch.save(frozen_cls.state_dict(), CKPT_DIR / f"{tag}_stage2.pth")


def run_two_stage(n_cv: int = 5, harmonize: bool = True) -> None:
    X_raw, pheno = load_processed()
    folds = load_splits(n_splits=n_cv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_metrics: list[dict] = []

    for k, (train_idx, test_idx) in enumerate(folds):
        print(f"\n=== Two-stage VAE  Fold {k} ===")
        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]
        y_train = pheno["label"].values[train_idx]
        y_test = pheno["label"].values[test_idx]

        X_train_vec = vectorize(X_train_raw)
        X_test_vec = vectorize(X_test_raw)

        if harmonize:
            try:
                X_train_vec, X_test_vec = fit_and_apply_combat(
                    X_train_vec, X_test_vec,
                    pheno["site"].values[train_idx], pheno["site"].values[test_idx],
                    pheno["age"].values[train_idx],  pheno["age"].values[test_idx],
                    pheno["sex"].values[train_idx],  pheno["sex"].values[test_idx],
                )
                print(f"  [combat] harmonization applied (fold {k})")
            except Exception as e:
                print(f"  [combat] skipped: {e}")

        scaler = fit_scaler(X_train_vec)
        X_train_n, X_test_n = apply_scaler(scaler, X_train_vec, X_test_vec)

        train_ds = ABIDEDataset(X_train_n, y_train)
        test_ds = ABIDEDataset(X_test_n, y_test)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

        tag = f"two_stage_vae_fold{k}"
        vae = VariationalAutoencoder().to(device)

        train_stage1(vae, train_loader, test_loader, tag, device)
        vae.load_state_dict(torch.load(CKPT_DIR / f"{tag}_stage1.pth", map_location=device))

        frozen_cls = FrozenEncoderClassifier(vae).to(device)
        train_stage2(frozen_cls, train_loader, tag, device)

        # Evaluate: wrap frozen_cls as something evaluate() can handle
        # We need a compatible forward wrapper for evaluate()
        class _EvalWrapper(nn.Module):
            def __init__(self, fc):
                super().__init__()
                self.fc = fc
            def forward(self, x):
                logits = self.fc(x)
                mu, _ = self.fc.vae.encode(x)
                return None, mu, None, None, logits

        wrapper = _EvalWrapper(frozen_cls).to(device)
        metrics = evaluate(wrapper, test_loader, device, "vae")
        metrics_clean = {k: v for k, v in metrics.items() if k not in ("probs", "labels")}
        metrics_clean["fold"] = k
        fold_metrics.append(metrics_clean)
        print(f"  Fold {k} AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}")

    df = pd.DataFrame(fold_metrics)
    out_path = METRICS_DIR / "two_stage_vae.csv"
    df.to_csv(out_path, index=False)
    print(f"\nTwo-stage VAE results (mean±std):")
    for col in ["auc", "f1", "balanced_acc"]:
        print(f"  {col}: {df[col].mean():.4f} ± {df[col].std():.4f}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--no-harmonize", action="store_true")
    args = parser.parse_args()
    run_two_stage(n_cv=args.cv, harmonize=not args.no_harmonize)
