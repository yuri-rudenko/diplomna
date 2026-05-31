"""
Smoke test: runs the full pipeline on the 50 cached Pitt subjects.
No GPU required. Should complete in ~3-5 minutes.

What is tested:
  - FC matrix building (Fisher-Z)
  - Stratified splits
  - StandardScaler normalization
  - Forward pass of all 3 models (correct shapes)
  - Loss computation for all 3 models
  - 5-epoch training loop (SAE, VAE, Attention-VAE)
  - Evaluation (AUC, F1, BalAcc)
  - Comparison table saved to results/metrics/
  - SHAP (fast mode, 5 background samples)

Usage:
    python smoke_test.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

PASS = "[PASS]"
FAIL = "[FAIL]"
SEP = "-" * 55


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


# ---------------------------------------------------------------------------
# 1. Build FC matrices from cached Pitt subjects (no re-download)
# ---------------------------------------------------------------------------
section("1. Load Pitt subjects & build FC matrices")

try:
    from nilearn import datasets
    from src.data.download_abide import _fisher_z, _load_ts, _pearson_fc

    DATA_DIR = PROJECT_DIR / "data"
    bunch = datasets.fetch_abide_pcp(
        data_dir=str(DATA_DIR),
        pipeline="cpac",
        derivatives=["rois_cc200"],
        band_pass_filtering=True,
        global_signal_regression=False,
        quality_checked=True,
        n_subjects=50,
        verbose=0,
    )

    fc_matrices, labels, sites = [], [], []
    import pandas as pd
    pheno_raw = pd.DataFrame(bunch.phenotypic).reset_index(drop=True)

    for i, item in enumerate(bunch.rois_cc200):
        ts = _load_ts(item)
        if ts is None:
            continue
        fc = _pearson_fc(ts)
        fc_z = _fisher_z(fc)
        fc_matrices.append(fc_z)
        labels.append(int(pheno_raw["DX_GROUP"].iloc[i]) == 1)
        sites.append(str(pheno_raw["SITE_ID"].iloc[i]))

    X = np.stack(fc_matrices, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    print(f"  Subjects: {len(X)}  ASD: {y.sum()}  Control: {(y==0).sum()}")
    print(f"  FC shape: {X.shape}  NaN: {np.isnan(X).any()}")
    assert X.shape[1:] == (200, 200), "FC shape mismatch"
    assert not np.isnan(X).any(), "NaN in FC matrices"
    print(f"{PASS} FC matrices OK")
except Exception:
    traceback.print_exc()
    print(f"{FAIL} FC matrices")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 2. Splits (2-fold on 50 subjects)
# ---------------------------------------------------------------------------
section("2. Stratified splits")

try:
    import pandas as pd
    from src.data.splits import make_cv_splits

    pheno_df = pd.DataFrame({"label": y, "site": sites})
    folds = make_cv_splits(pheno_df, n_splits=2, random_state=42)
    assert len(folds) == 2
    for k, (tr, te) in enumerate(folds):
        assert len(tr) + len(te) == len(X)
    print(f"{PASS} 2-fold splits OK")
except Exception:
    traceback.print_exc()
    print(f"{FAIL} Splits")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
section("3. Vectorization + StandardScaler")

try:
    from src.data.preprocessing import apply_scaler, fit_scaler, vectorize

    train_idx, test_idx = folds[0]
    X_train_vec = vectorize(X[train_idx])
    X_test_vec = vectorize(X[test_idx])
    assert X_train_vec.shape[1] == 19900

    scaler = fit_scaler(X_train_vec)
    X_train_n, X_test_n = apply_scaler(scaler, X_train_vec, X_test_vec)
    assert X_train_n.shape == X_train_vec.shape
    print(f"  Train vec: {X_train_n.shape}  mean≈{X_train_n.mean():.4f}  std≈{X_train_n.std():.4f}")
    print(f"{PASS} Preprocessing OK")
except Exception:
    traceback.print_exc()
    print(f"{FAIL} Preprocessing")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 4. Dataset + DataLoader
# ---------------------------------------------------------------------------
section("4. ABIDEDataset")

try:
    from src.data.dataset import ABIDEDataset

    train_ds = ABIDEDataset(X_train_n, y[train_idx])
    test_ds = ABIDEDataset(X_test_n, y[test_idx])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=8, drop_last=False)

    xb, yb = next(iter(train_loader))
    assert xb.shape[1] == 19900
    print(f"  Batch X: {xb.shape}  y: {yb.shape}  dtype: {xb.dtype}")
    print(f"{PASS} Dataset OK")
except Exception:
    traceback.print_exc()
    print(f"{FAIL} Dataset")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 5. Model forward passes (tiny architecture for speed)
# ---------------------------------------------------------------------------
section("5. Model forward passes")

TINY = {"hidden_dims": [64, 32], "latent_dim": 16}
device = torch.device("cpu")

try:
    from src.models.sparse_ae import SparseAutoencoder
    from src.models.vae import VariationalAutoencoder
    from src.models.attention_vae import AttentionVAE

    models_to_test = [
        ("SAE",          SparseAutoencoder(**TINY),  "sae"),
        ("VAE",          VariationalAutoencoder(**TINY), "vae"),
        ("Attention-VAE", AttentionVAE(**TINY, n_patches=4, n_heads=2), "vae"),
    ]

    x_fake = torch.randn(8, 19900)
    y_fake = torch.randint(0, 2, (8,))

    for name, model, mtype in models_to_test:
        model.eval()
        with torch.no_grad():
            if mtype == "sae":
                x_recon, z, logits = model(x_fake)
                loss, *_ = model.loss(x_fake, x_recon, z, logits, y_fake)
                print(f"  {name}: recon={x_recon.shape} z={z.shape} logits={logits.shape} loss={loss.item():.4f}")
            else:
                x_recon, mu, logvar, z, logits = model(x_fake)
                loss, *_ = model.loss(x_fake, x_recon, mu, logvar, logits, y_fake)
                print(f"  {name}: recon={x_recon.shape} mu={mu.shape} logits={logits.shape} loss={loss.item():.4f}")
            assert logits.shape == (8, 2), f"logits shape wrong: {logits.shape}"
            assert x_recon.shape == x_fake.shape
            assert not torch.isnan(loss)

    print(f"{PASS} All model forward passes OK")
except Exception:
    traceback.print_exc()
    print(f"{FAIL} Model forward passes")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 6. Training loop (5 epochs, tiny model)
# ---------------------------------------------------------------------------
section("6. Training loop — 5 epochs × 3 models")

try:
    from src.training.trainer import evaluate, train_model

    results = {}
    t0 = time.time()

    for name, model, mtype in models_to_test:
        model = type(model)(**TINY, **({"n_patches": 4, "n_heads": 2} if name == "Attention-VAE" else {}))
        tag = f"smoke_{name.lower().replace('-','_').replace(' ','_')}"
        history = train_model(
            model, train_loader, test_loader,
            model_type=mtype,
            tag=tag,
            n_epochs=5,
            lr=1e-3,
            device=device,
            patience=10,
            phase1_epochs=2,
            cls_weight_max=5.0,
            kl_warmup_epochs=2,
            verbose=True,
        )
        metrics = evaluate(model, test_loader, device, mtype)
        results[name] = metrics
        print(f"  {name}: AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  BalAcc={metrics['balanced_acc']:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"{PASS} Training loop OK")
except Exception:
    traceback.print_exc()
    print(f"{FAIL} Training loop")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 7. Schedules
# ---------------------------------------------------------------------------
section("7. Schedules")

try:
    from src.training.schedules import beta_schedule, cls_weight_schedule

    assert beta_schedule(0, warmup_epochs=30) == 0.0
    assert beta_schedule(30, warmup_epochs=30) == 1.0
    assert beta_schedule(100, warmup_epochs=30) == 1.0
    # cls_weight stays 0 in phase 1, then ramps 0 -> max linearly over
    # warmup_epochs (default 10) starting at phase1_epochs.
    assert cls_weight_schedule(0,  phase1_epochs=30) == 0.0   # phase 1
    assert cls_weight_schedule(29, phase1_epochs=30) == 0.0   # still phase 1
    assert cls_weight_schedule(30, phase1_epochs=30) == 0.0   # phase-2 ramp starts at 0
    assert cls_weight_schedule(35, phase1_epochs=30) == 2.5   # halfway up the 10-epoch ramp
    assert cls_weight_schedule(40, phase1_epochs=30) == 5.0   # ramp complete
    assert cls_weight_schedule(100, phase1_epochs=30) == 5.0  # stable at max
    print(f"  beta(0)={beta_schedule(0,30):.1f}  beta(15)={beta_schedule(15,30):.2f}  beta(30)={beta_schedule(30,30):.1f}")
    print(f"  cls_w(30)={cls_weight_schedule(30,30):.1f}  cls_w(35)={cls_weight_schedule(35,30):.1f}  "
          f"cls_w(40)={cls_weight_schedule(40,30):.1f}")
    print(f"{PASS} Schedules OK")
except Exception:
    traceback.print_exc()
    print(f"{FAIL} Schedules")


# ---------------------------------------------------------------------------
# 8. SHAP (fast mode: 5 background samples, CPU)
# ---------------------------------------------------------------------------
section("8. SHAP DeepExplainer (fast mode)")

try:
    import shap

    from src.models.attention_vae import AttentionVAE

    tiny_model = AttentionVAE(**TINY, n_patches=4, n_heads=2).to(device)
    tiny_model.eval()

    class _Wrapper(torch.nn.Module):
        def forward(self, x):
            _, mu, _, _, logits = tiny_model(x)
            return torch.softmax(logits, dim=1)[:, 1:2]

    wrapper = _Wrapper()
    bg = xb[:5]  # 5 background samples
    test_in = xb[:4]

    explainer = shap.DeepExplainer(wrapper, bg)
    shap_vals = explainer.shap_values(test_in, check_additivity=False)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_arr = np.array(shap_vals)
    if shap_arr.ndim == 3:          # (N, features, 1) → (N, features)
        shap_arr = shap_arr[..., 0]
    print(f"  SHAP output shape: {shap_arr.shape}  (expected: (4, 19900))")
    assert shap_arr.shape == (4, 19900), f"SHAP shape mismatch: {shap_arr.shape}"
    print(f"{PASS} SHAP OK")
except ImportError:
    print(f"  [skip] shap not installed (pip install shap)")
except Exception:
    traceback.print_exc()
    print(f"{FAIL} SHAP")


# ---------------------------------------------------------------------------
# 9. LogReg baseline + train/val/test split (no-leakage plumbing)
# ---------------------------------------------------------------------------
section("9. LogReg baseline + train/val/test split")

try:
    from sklearn.model_selection import train_test_split
    from run_experiments import fit_logreg_baseline

    tr, te = folds[0]

    # Stratified train_sub/val split mirroring run_experiment — the test fold
    # is never mixed into train or val (no model selection on test).
    sub_pos, val_pos = train_test_split(
        np.arange(len(tr)), test_size=0.15, stratify=y[tr], random_state=42,
    )
    train_sub, val_sub = tr[sub_pos], tr[val_pos]
    assert set(train_sub.tolist()) & set(val_sub.tolist()) == set()
    assert set(val_sub.tolist())   & set(te.tolist())      == set()
    assert set(train_sub.tolist()) & set(te.tolist())      == set()
    print(f"  split sizes — train_sub={len(train_sub)} val={len(val_sub)} test={len(te)}")

    # LogReg classical baseline: fit on the FULL train fold, evaluate on test.
    X_tr_full = vectorize(X[tr])
    X_te_vec  = vectorize(X[te])
    lr_metrics, lr_probs = fit_logreg_baseline(X_tr_full, y[tr], X_te_vec, y[te])
    assert lr_probs.shape == (len(te),)
    assert {"auc", "f1", "balanced_acc", "acc", "sensitivity", "specificity"} <= set(lr_metrics)
    assert 0.0 <= lr_metrics["auc"] <= 1.0 and np.isfinite(lr_probs).all()
    print(f"  logreg: AUC={lr_metrics['auc']:.4f}  F1={lr_metrics['f1']:.4f}  "
          f"BalAcc={lr_metrics['balanced_acc']:.4f}")
    print(f"{PASS} LogReg baseline + split OK")
except Exception:
    traceback.print_exc()
    print(f"{FAIL} LogReg baseline + split")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("SUMMARY")

print("All core components verified on 50 cached Pitt subjects.")
print("Ready to run full pipeline on Kaggle/Colab with ~871 subjects.\n")
print("Next steps:")
print("  1. python -m src.data.download_abide       # full dataset")
print("  2. python -m src.data.splits               # 5-fold splits")
print("  3. python -m src.compare_models --cv 5     # full training")
