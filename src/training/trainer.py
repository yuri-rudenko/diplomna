"""
Training loop for SAE / VAE / Attention-VAE.

train_model() handles:
- Two-phase training (reconstruction → joint)
- KL annealing for VAE/Attention-VAE
- Early stopping by val AUC
- Checkpoint saving (best AUC + best F1)
- tqdm progress bar per epoch
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.schedules import beta_schedule, cls_weight_schedule

PROJECT_DIR = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT_DIR / "results" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

ModelType = Any  # SparseAutoencoder | VariationalAutoencoder | AttentionVAE


def _forward_loss(
    model: ModelType,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    model_type: str,
    beta: float,
    cls_weight: float,
    recon_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_type == "sae":
        x_recon, z, logits = model(X_batch)
        total, *_ = model.loss(X_batch, x_recon, z, logits, y_batch,
                               recon_weight=recon_weight, cls_weight=cls_weight)
    else:
        x_recon, mu, logvar, z, logits = model(X_batch)
        total, *_ = model.loss(X_batch, x_recon, mu, logvar, logits, y_batch,
                               beta=beta, recon_weight=recon_weight, cls_weight=cls_weight)
    return total, logits


def train_one_epoch(
    model: ModelType,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str,
    beta: float,
    cls_weight: float,
    epoch_bar: tqdm,
) -> tuple[float, float]:
    model.train()
    total_loss = correct = n = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss, logits = _forward_loss(model, X_batch, y_batch, model_type, beta, cls_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y_batch).sum().item()
        n += len(y_batch)

        epoch_bar.set_postfix(loss=f"{total_loss/( epoch_bar.n or 1):.4f}", refresh=False)

    return total_loss / len(loader), correct / n


@torch.no_grad()
def evaluate(
    model: ModelType,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
) -> dict[str, float | np.ndarray]:
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        if model_type == "sae":
            _, _, logits = model(X_batch)
        else:
            _, mu, _, _, logits = model(X_batch)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)

    return {
        "auc":          float(roc_auc_score(labels, probs)),
        "f1":           float(f1_score(labels, preds, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "acc":          float((preds == labels).mean()),
        "sensitivity":  float(((preds == 1) & (labels == 1)).sum() / max((labels == 1).sum(), 1)),
        "specificity":  float(((preds == 0) & (labels == 0)).sum() / max((labels == 0).sum(), 1)),
        "probs":  probs,
        "labels": labels,
    }


def train_model(
    model: ModelType,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type: str,
    tag: str = "model",
    n_epochs: int = 150,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    patience: int = 20,
    phase1_epochs: int = 30,
    cls_weight_max: float = 5.0,
    kl_warmup_epochs: int = 30,
    verbose: bool = True,
) -> dict[str, list]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    def _make_optimizer():
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    optimizer = _make_optimizer()
    # Phase 1: CosineAnnealing for recon warmup
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=lr * 0.1)

    best_auc = best_f1 = 0.0
    patience_counter = 0
    phase2_started = False
    history: dict[str, list] = {
        "train_loss": [], "val_auc": [], "val_f1": [], "val_balanced_acc": [],
        "beta": [], "cls_weight": [],
    }

    epoch_bar = tqdm(
        range(n_epochs),
        desc=f"{tag}",
        unit="ep",
        dynamic_ncols=True,
        leave=True,
    )

    t_start = time.time()

    for epoch in epoch_bar:
        beta = beta_schedule(epoch, warmup_epochs=kl_warmup_epochs)
        cw   = cls_weight_schedule(epoch, phase1_epochs=phase1_epochs, cls_weight_max=cls_weight_max)

        # At phase transition: reset optimizer + switch to ReduceLROnPlateau
        if epoch == phase1_epochs and not phase2_started:
            phase2_started = True
            optimizer = _make_optimizer()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
            )
            patience_counter = 0
            best_auc = 0.0

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, model_type,
            beta=beta, cls_weight=cw, epoch_bar=epoch_bar,
        )
        val_metrics = evaluate(model, val_loader, device, model_type)

        # Step scheduler
        if phase2_started:
            scheduler.step(val_metrics["auc"])
        else:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_metrics["auc"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_balanced_acc"].append(val_metrics["balanced_acc"])
        history["beta"].append(beta)
        history["cls_weight"].append(cw)

        phase = "recon" if cw == 0 else f"joint(cw={cw:.1f})"
        best_marker = " ★" if val_metrics["auc"] >= best_auc else ""
        lr_cur = optimizer.param_groups[0]["lr"]

        epoch_bar.set_description(f"{tag}")
        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            auc=f"{val_metrics['auc']:.4f}{best_marker}",
            f1=f"{val_metrics['f1']:.4f}",
            phase=phase,
            lr=f"{lr_cur:.0e}",
            patience=f"{patience_counter}/{patience}",
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), CKPT_DIR / f"{tag}_best_auc.pth")
            patience_counter = 0
        elif phase2_started:
            patience_counter += 1

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), CKPT_DIR / f"{tag}_best_f1.pth")

        if patience_counter >= patience:
            epoch_bar.set_description(f"{tag} [early stop]")
            epoch_bar.close()
            break

    tqdm.write(
        f"  ✓ {tag} | best AUC={best_auc:.4f}  best F1={best_f1:.4f}  "
        f"time={time.time()-t_start:.0f}s"
    )

    hist_path = CKPT_DIR / f"{tag}_history.json"
    with open(hist_path, "w") as f:
        json.dump({k: v for k, v in history.items() if k not in ("probs", "labels")}, f)

    return history
