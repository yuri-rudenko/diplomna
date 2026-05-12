# Comparative Analysis of Deep Learning Architectures for ASD Diagnosis Using fMRI Data

Diploma thesis project: classifying Autism Spectrum Disorder (ASD) from
resting-state fMRI using and comparing three autoencoder architectures.

## Dataset

- **ABIDE I** (Autism Brain Imaging Data Exchange), preprocessed release.
- Pipeline: **CPAC** (band-pass filtered, no global signal regression, quality-checked).
- Derivative: **rois_cc200** — mean BOLD time-series for 200 ROIs (Craddock-200 atlas).
- Accessed via `nilearn.datasets.fetch_abide_pcp` (~871 subjects).

## Models

| Model | Description |
|---|---|
| **Sparse AE (SAE)** | Sparse Autoencoder + MLP classifier head (ASD-SAENet style) |
| **VAE** | Variational Autoencoder + KL annealing + classifier on μ |
| **Attention-VAE** | VAE with patch-based Multi-Head Self-Attention in encoder |

All models: `input(19900) → [512 → 256] → latent(64) → [256 → 512] → output(19900)` + classifier.  
Input: upper-triangle of Fisher-Z transformed FC matrix (200×200 → 19900 features).

## Project structure

```
diploma/
├── data/
│   ├── ABIDE_pcp/           # nilearn cache (auto-downloaded)
│   └── processed/           # fc_matrices.npy, phenotype.csv, splits.npz
├── src/
│   ├── data/
│   │   ├── download_abide.py    # download + build FC matrices
│   │   ├── dataset.py           # ABIDEDataset (PyTorch)
│   │   ├── splits.py            # 5-fold stratified CV splits
│   │   ├── preprocessing.py     # StandardScaler wrapper
│   │   └── harmonize.py         # ComBat site harmonization
│   ├── models/
│   │   ├── sparse_ae.py         # Sparse Autoencoder
│   │   ├── vae.py               # Variational Autoencoder
│   │   └── attention_vae.py     # Attention-VAE (patch MHSA)
│   ├── training/
│   │   ├── trainer.py           # training loop (two-phase + KL annealing)
│   │   ├── schedules.py         # beta_schedule, cls_weight_schedule
│   │   ├── two_stage.py         # ablation: freeze encoder → train classifier
│   │   └── significance.py      # paired t-test between models
│   ├── xai/
│   │   ├── shap_explain.py      # DeepSHAP → connection importance
│   │   └── roi_brain.py         # ROI importance → brain visualization
│   ├── compare_models.py        # main entry point
│   └── visualize_sample.py      # EDA visualizations (existing)
├── results/
│   ├── checkpoints/             # .pth model weights
│   ├── metrics/                 # CSV result tables
│   └── figures/                 # PNG plots
└── requirements.txt
```

---

## Step-by-step guide

### 1. Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> **GPU recommended** for training. CPU works but will be slow (~4-8h for full CV).

---

### 2. Download and prepare data

```powershell
# Downloads ~871 subjects, builds FC matrices, saves to data/processed/
# Takes ~10-20 min on first run; subsequent runs use nilearn cache.
python -m src.data.download_abide
```

**Expected output:**
```
Done — 871 subjects  |  ASD: 403  |  Control: 468
Sites (17): ['Caltech', 'CMU', 'KKI', ...]
```

**Verify:**
```powershell
python -c "import numpy as np; X=np.load('data/processed/fc_matrices.npy'); print(X.shape, X.dtype, 'NaN:', np.isnan(X).any())"
# Expected: (871, 200, 200) float64 NaN: False
```

---

### 3. Exploratory analysis (EDA)

```powershell
python src/visualize_sample.py
```

Outputs to `visualizations/`: group counts, age/sex distributions,
mean ASD vs Control FC matrices, BOLD time-series examples, PCA/UMAP projections.
Use these figures in the **"Dataset Description"** section of the thesis.

---

### 4. Build 5-fold CV splits

```powershell
python -m src.data.splits
```

Creates `data/processed/splits.npz`. Splits are stratified by `(site, label)`
to prevent site-leakage. Check the printed balance report — ASD% should be
≈equal across folds.

---

### 5. Train all three models (full CV)

```powershell
# Full run: 5-fold CV, ComBat harmonization, 150 epochs per model
python -m src.compare_models --cv 5 --epochs 150

# Skip ComBat (faster, less accurate):
python -m src.compare_models --no-harmonize

# Quick smoke test (2 folds, 10 epochs):
python -m src.compare_models --cv 2 --epochs 10
```

**What happens inside each fold:**
1. ComBat fit on train → transform val/test (site-effect removal).
2. StandardScaler fit on train → transform val/test.
3. SAE, VAE, Attention-VAE each trained for up to 150 epochs with:
   - Phase 1 (epochs 0-30): reconstruction only
   - Phase 2 (epochs 30+): joint reconstruction + classification (`cls_weight=5.0`)
   - KL annealing: β grows 0 → 1 over 30 epochs (VAE/Attention-VAE)
   - Early stopping: patience=20 epochs on val AUC
4. Best checkpoint (by AUC) saved to `results/checkpoints/`.
5. Test metrics (AUC, F1, BalAcc, Sensitivity, Specificity) saved per fold.

**Outputs:**
- `results/metrics/cv_per_fold.csv` — metrics per (model, fold)
- `results/metrics/comparison_table.csv` — `mean ± std` summary
- `results/figures/roc_curves.png`, `metrics_bars.png`, `training_curves.png`

**Expected results (literature reference):**
| Model | AUC | F1 |
|---|---|---|
| SAE | ~0.70-0.72 | ~0.68 |
| VAE | ~0.72-0.75 | ~0.71 |
| Attention-VAE | ~0.74-0.78 | ~0.73 |

---

### 6. Ablation: two-stage VAE

```powershell
python -m src.training.two_stage --cv 5
```

Trains a pure VAE (no classifier, reconstruction + KL only), then freezes
the encoder and trains an MLP classifier on top of the frozen `μ`.
Compare with joint VAE from step 5 in the thesis ablation section.

Output: `results/metrics/two_stage_vae.csv`

---

### 7. Statistical significance

```powershell
python -m src.training.significance
```

Paired t-test on AUC values across 5 folds for each model pair.
**p < 0.05** means the difference is statistically significant.

Output: `results/metrics/significance.csv`

---

### 8. Regenerate figures only

```powershell
python -m src.compare_models --plot-only
```

---

### 9. XAI: SHAP explanation

```powershell
# Run SHAP on the best model (Attention-VAE by default)
python -m src.xai.shap_explain --model attention_vae --fold best
```

**What happens:**
- Loads the best-fold Attention-VAE checkpoint.
- Runs `shap.DeepExplainer` with 100 train subjects as background.
- Computes importance for all 19 900 FC connections.
- Aggregates to ROI importance: `mean |SHAP|` across all 199 connections of each ROI.

**Outputs:**
- `results/figures/shap_values_attention_vae_fold{k}.npy`
- `results/figures/roi_importance_attention_vae_fold{k}.npy`
- `results/figures/shap_top20_table.csv` — top-20 ROIs with importance scores

---

### 10. Brain visualization

```powershell
python -m src.xai.roi_brain --model attention_vae --fold 0
```

**Outputs:**
- `results/figures/brain_glassbrain_attention_vae_fold0.png` — glass-brain view
- `results/figures/brain_statmap_attention_vae_fold0.png` — stat-map on MNI
- `results/figures/attention_heatmap_attention_vae_fold0.png` — patch attention weights

**Interpretation:**
- Compare top-20 ROIs with known ASD-related networks:
  - Default Mode Network (precuneus, mPFC, posterior cingulate)
  - Salience Network (insula, anterior cingulate)
  - Fronto-parietal control network
- If top ROIs overlap with these networks → strong argument for the thesis discussion.

---

## What to write in the thesis

| Section | Source |
|---|---|
| Dataset description | `visualizations/*.png` + `phenotype.csv` stats |
| Methods: preprocessing | Fisher-Z, upper-triangle, ComBat, StandardScaler |
| Methods: architectures | `src/models/*.py` — layer sizes, activation functions, loss formulas |
| Methods: training | Two-phase scheme, KL annealing, early stopping |
| Results: comparison | `results/metrics/comparison_table.csv` + ROC curves + bar chart |
| Results: significance | `results/metrics/significance.csv` (paired t-test) |
| Results: ablation | `two_stage_vae.csv` vs joint VAE |
| Discussion: XAI | SHAP brain maps + top-20 ROI table + attention heatmap |
| Discussion: comparison with SOTA | ASD-SAENet (Almuqhim 2021), ASD-DiagNet (Eslami 2019) |

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| AUC < 0.60 on all models | Site leakage | Check ComBat + scaler fit only on train |
| KL → 0 (posterior collapse) | KL annealing too fast | Increase `kl_warmup_epochs` to 50 |
| Unstable metrics between folds | Overfitting | Increase `dropout` to 0.4, `weight_decay` to 1e-3 |
| SHAP OOM error | GPU memory | Reduce `--n-background 50`, use CPU |
| ComBat `singular matrix` error | Site with < 3 subjects in fold | Sites auto-merged to 'other_label' fallback |

---

## References

- Almuqhim & Saeed (2021). ASD-SAENet. *Front. Comput. Neurosci.*
- Eslami et al. (2019). ASD-DiagNet. *Front. Neurosci.*
- Di Martino et al. (2014). ABIDE. *Mol. Psychiatry.*
- Kingma & Welling (2013). Auto-Encoding Variational Bayes. *ICLR 2014.*
- Fortin et al. (2018). ComBat harmonization for fMRI. *NeuroImage.*

## Reproducibility

All randomness is seeded with `random_state=42`.
