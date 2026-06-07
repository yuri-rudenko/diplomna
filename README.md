# Comparative Analysis of Deep Learning Architectures for ASD Diagnosis Using fMRI

A diploma research project that classifies **Autism Spectrum Disorder (ASD)** vs.
typical controls from **resting-state fMRI** functional-connectivity (FC) matrices,
comparing three autoencoder architectures against a Logistic-Regression baseline
and a soft-voting ensemble. The full pipeline covers **10 ablation experiments,
5-fold cross-validation, statistical significance testing, and SHAP / UMAP / brain-map
explainability** — every step is a single reproducible command that runs unattended
on **local machines, Google Colab, or Kaggle**.

> **Headline finding:** the Logistic-Regression baseline (AUC ≈ 0.746) matches or
> beats every deep model; the best deep ensemble (`aug_full`) reaches the same
> ≈0.746, and a frozen-encoder two-stage VAE collapses to near-chance (≈0.60).
> See [Key results](#key-results).

Repository: <https://github.com/yuri-rudenko/diplomna>

---

## Table of contents

- [Dataset](#dataset)
- [Models](#models)
- [Experiments (ablations)](#experiments-ablations)
- [Quick start](#quick-start)
  - [Local](#local-windows--linux--macos)
  - [Google Colab](#google-colab)
  - [Kaggle](#kaggle)
- [Full pipeline (step by step)](#full-pipeline-step-by-step)
- [Smoke test](#smoke-test)
- [Project structure](#project-structure)
- [Output locations](#output-locations)
- [Key results](#key-results)
- [Troubleshooting](#troubleshooting)
- [Reproducibility](#reproducibility)
- [References](#references)

---

## Dataset

- **ABIDE I** (Autism Brain Imaging Data Exchange), preprocessed release.
- Pipeline: **CPAC** (band-pass filtered, no global-signal regression, quality-checked).
- Derivative: **rois_cc200** — mean BOLD time-series for 200 ROIs (Craddock CC200 atlas).
- Accessed via `nilearn.datasets.fetch_abide_pcp` and built into FC matrices automatically.
- **871 subjects** — **403 ASD / 468 control**, across **20 acquisition sites**.

Each subject is reduced to the upper triangle of its Fisher-Z transformed 200×200 FC
matrix (**19 900 features**). Splits are 5-fold stratified by `(site, label)` to prevent
site leakage.

---

## Models

| Model | Description |
|---|---|
| **Sparse AE (SAE)** | Sparse Autoencoder + MLP classifier head (ASD-SAENet style) |
| **VAE** | Variational Autoencoder + KL annealing + classifier on μ |
| **Attention-VAE** | VAE with patch-based Multi-Head Self-Attention in the encoder |
| **LogReg** | Logistic-Regression baseline on the raw FC features |
| **Ensemble** | Soft-voting average of the deep model + LogReg probabilities |

All autoencoders share the shape
`input(19900) → [512 → 256] → latent(64) → [256 → 512] → output(19900)` plus a
classifier head. Each training run produces, **per fold**, the deep model **plus**
a LogReg and an ensemble.

Approx. parameter counts: SAE ≈ 20.70 M · VAE ≈ 20.72 M · Attention-VAE ≈ 20.74 M.

---

## Experiments (ablations)

`run_experiments.py` drives 10 named configs (defined in [`src/experiments.py`](src/experiments.py)):

| Experiment | What it changes |
|---|---|
| `baseline` | Adam, no augmentation, no harmonization |
| `adamw` | AdamW optimizer |
| `aug_noise` | + Gaussian noise augmentation |
| `aug_mixup` | + Mixup augmentation |
| `aug_full` | Noise + Mixup + AdamW + label smoothing (**best combo**) |
| `combat` | ComBat site harmonization |
| `latent_32` | latent dim 32 |
| `latent_128` | latent dim 128 |
| `deep` | deeper network |
| `high_dropout` | higher dropout |

List them any time with `python run_experiments.py --list`.

---

## Quick start

**Requirements:** Python **3.10–3.12**, ~5 GB free disk, a CUDA GPU recommended for
training (CPU works but is slow). Key pinned deps: `nilearn 0.13.1`, `nibabel 5.4.2`,
`numpy 2.0.2`, `torch>=2.0`, `scikit-learn 1.6.1`, `shap 0.51.0`, `neuroCombat 0.2.12`,
`umap-learn 0.5.12`, `plotly 6.7.0`, `kaleido 1.3.0`. Seed is fixed at **42**.

### Local (Windows / Linux / macOS)

```powershell
# Windows PowerShell
git clone https://github.com/yuri-rudenko/diplomna.git
cd diplomna
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# Linux / macOS
git clone https://github.com/yuri-rudenko/diplomna.git
cd diplomna
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then run the [full pipeline](#full-pipeline-step-by-step). Verify the environment
first with the [smoke test](#smoke-test).

### Google Colab

Use a **GPU runtime** (Runtime → Change runtime type → GPU).

```python
!git clone https://github.com/yuri-rudenko/diplomna.git
%cd /content/diplomna
!pip install -r requirements.txt
```

Then run each `!python ...` command from the [full pipeline](#full-pipeline-step-by-step)
in its own cell, in order.

> Colab's disk is **ephemeral**. To keep `results/` and the export across runtime
> resets, mount Drive and copy (optional):
> ```python
> from google.colab import drive; drive.mount('/content/drive')
> import shutil
> shutil.copytree('results', '/content/drive/MyDrive/diplomna/results', dirs_exist_ok=True)
> shutil.copytree('diploma_export', '/content/drive/MyDrive/diplomna/diploma_export', dirs_exist_ok=True)
> ```

### Kaggle

Enable a GPU accelerator and Internet in the notebook settings, then use the same
clone + `pip install -r requirements.txt` + pipeline commands as Colab.

> `torch` is pinned as a **floor** (`>=2.0`), not an exact version, precisely so
> Colab/Kaggle's preinstalled CUDA-matched build is reused instead of triggering a
> multi-GB downgrade.

---

## Full pipeline (step by step)

All commands run **from the repository root**. Prefix with `!` in Colab/Kaggle cells.
Approximate timings: GPU steps ≈ 3–5 min/experiment on an A100; data download ≈ 30 min
first run (cached afterwards); all CPU steps are seconds to a few minutes.

```bash
# 1. Download ABIDE I, build FC matrices, auto-create 5-fold splits          [CPU]
python -m src.data.download_abide

# 2. Train all experiments — baseline + every ablation                       [GPU]
python run_experiments.py --exp baseline --cv 5
python run_experiments.py --exp aug_full latent_32 latent_128 aug_noise aug_mixup --cv 5
python run_experiments.py --exp adamw combat deep high_dropout --cv 5
#    (equivalently, run all ten at once:)
python run_experiments.py --cv 5

# 3. Build the cross-experiment comparison table                             [CPU]
python run_experiments.py --compare

# 4. Two-stage frozen-encoder VAE control experiment                         [GPU]
python -m src.training.two_stage --cv 5

# 5. Paired-t significance between models (within the baseline experiment)    [CPU]
python -m src.training.significance --exp baseline

# 6. Explainability — SHAP + brain glass/stat maps + attention heatmap,
#    all models at the best fold, one orchestrator command                   [CPU]
python -m src.xai.run_xai --exp aug_full

# 7. UMAP latent projection + all report charts                              [CPU]
python -m src.visualizations.umap_latent --exp aug_full --fold best
python -m src.visualizations.report_charts

# 8. Postprocess: PR curves, confusion matrices, cross-experiment tests       [CPU]
python run_experiments.py --postprocess --cv 5 --exp-a baseline --exp-b aug_full

# 9. Collect everything into diploma_export/ + FULL_RESULTS.md + zip          [CPU]
python export_results.py
```

**Useful `run_experiments.py` flags:** `--exp <names...>`, `--cv N` (default 5),
`--epochs N`, `--list`, `--compare`, `--plot-only`, `--postprocess`,
`--exp-a`/`--exp-b` (experiments to compare in postprocess).

**How paths resolve:** `--exp <name>` is the single source of truth. XAI, UMAP, and
the brain/attention plots read checkpoints and scalers from
`results/experiments/<name>/checkpoints/` automatically — no manual file copying.
`--fold best` picks the highest-AUC fold from that experiment's `cv_per_fold.csv`.
ComBat is auto-detected from the experiment config, so only the `combat` experiment
is explained with harmonization applied.

**Expected harmless warnings:** a `RuntimeWarning: invalid value encountered in divide`
during FC build; a TensorFlow oneDNN/CPU info banner and a UMAP `n_jobs` UserWarning;
a scipy Wilcoxon `invalid value` warning on the zero-delta LogReg row; and a possible
`fetch_atlas_craddock_2012` SSL/`403` failure during brain-map download (the SHAP CSV/NPY
and attention heatmap still save — only the anatomical PNG is skipped). The Craddock
atlas is on nitrc.org with an invalid TLS cert; the code retries that one request with
verification disabled — see [`src/utils/atlas_fetch.py`](src/utils/atlas_fetch.py).

---

## Smoke test

Before committing to the full run, verify the environment end-to-end on 50 cached
Pitt subjects (~3–5 min on CPU, downloads its own data):

```bash
python smoke_test.py
```

---

## Project structure

```
diplomna/
├── run_experiments.py          # main driver: train / compare / postprocess
├── export_results.py           # packages diploma_export/ + FULL_RESULTS.md + zip
├── smoke_test.py               # fast end-to-end environment check
├── requirements.txt
├── RUN.md                      # condensed copy-paste command list
├── data/                       # (gitignored) ABIDE cache + processed FC + splits
├── src/
│   ├── data/
│   │   ├── download_abide.py   # download + build FC matrices + auto splits
│   │   ├── dataset.py          # ABIDEDataset (PyTorch)
│   │   ├── splits.py           # 5-fold stratified CV splits
│   │   ├── preprocessing.py    # StandardScaler wrapper
│   │   └── harmonize.py        # ComBat site harmonization
│   ├── models/
│   │   ├── sparse_ae.py        # Sparse Autoencoder
│   │   ├── vae.py              # Variational Autoencoder
│   │   └── attention_vae.py    # Attention-VAE (patch MHSA)
│   ├── training/
│   │   ├── trainer.py          # training loop (two-phase + KL annealing)
│   │   ├── schedules.py        # beta / cls-weight schedules
│   │   ├── two_stage.py        # frozen-encoder control experiment
│   │   └── significance.py     # paired t-test between models
│   ├── xai/
│   │   ├── run_xai.py          # XAI orchestrator (SHAP + brain + attention)
│   │   ├── shap_explain.py     # DeepSHAP → connection/ROI importance
│   │   └── roi_brain.py        # ROI importance → brain visualization
│   ├── visualizations/
│   │   ├── umap_latent.py      # UMAP on latent mu
│   │   ├── report_charts.py    # training curves, ROC, ablation, SHAP charts
│   │   └── pr_curves.py        # precision-recall curves
│   ├── utils/
│   │   ├── atlas_fetch.py      # SSL-tolerant CC200 atlas download
│   │   └── checkpoints.py      # checkpoint path resolution
│   └── experiments.py          # the 10 ablation configs
└── results/                    # (gitignored) experiments/, metrics/, figures/
```

---

## Output locations

| Path | Contents |
|---|---|
| `results/experiments/<exp>/metrics/summary.csv` | per-experiment `mean ± std` metrics |
| `results/experiments/<exp>/metrics/cv_per_fold.csv` | per-(model, fold) metrics |
| `results/experiments/<exp>/checkpoints/` | trained model weights + scalers |
| `results/experiments/comparison_all.csv` | ranking across all 10 experiments |
| `results/metrics/` | two-stage, significance, cross-experiment CSVs |
| `results/figures/` | all PNGs (ROC, training curves, SHAP, UMAP, ablation, attention, brain) |
| `diploma_export/` + `diploma_export.zip` | packaged deliverable (`FULL_RESULTS.md` inside) |

---

## Key results

Ensemble AUC ranking across the 10 experiments (5-fold CV, `mean ± std`):

| Experiment | AUC | F1 | Balanced acc. |
|---|---|---|---|
| `aug_full` | 0.7461 ± 0.0239 | 0.6442 ± 0.0322 | 0.6759 ± 0.0269 |
| `aug_mixup` | 0.7445 ± 0.0350 | 0.6491 ± 0.0357 | 0.6795 ± 0.0245 |
| `deep` | 0.7444 ± 0.0212 | 0.6483 ± 0.0227 | 0.6807 ± 0.0066 |
| `high_dropout` | 0.7435 ± 0.0331 | 0.6437 ± 0.0505 | 0.6806 ± 0.0429 |
| `latent_32` | 0.7433 ± 0.0294 | 0.6478 ± 0.0410 | 0.6782 ± 0.0322 |
| `aug_noise` | 0.7421 ± 0.0393 | 0.6445 ± 0.0339 | 0.6715 ± 0.0279 |
| `adamw` | 0.7403 ± 0.0278 | 0.6611 ± 0.0222 | 0.6888 ± 0.0268 |
| `latent_128` | 0.7387 ± 0.0305 | 0.6455 ± 0.0348 | 0.6785 ± 0.0367 |
| `baseline` | 0.7280 ± 0.0277 | 0.6401 ± 0.0140 | 0.6667 ± 0.0188 |
| `combat` | 0.7273 ± 0.0380 | 0.6436 ± 0.0444 | 0.6719 ± 0.0291 |

**Takeaways:**

- **LogReg (AUC ≈ 0.746) matches or beats every deep model** — differences fall
  within fold noise. In paired t-tests, only `sae↔logreg`, `vae↔logreg`, and
  `vae↔ensemble` reach p<.05, and **none survive p<.01**.
- **The two-stage frozen-encoder VAE collapses to near-chance** (AUC ≈ 0.60) →
  joint reconstruction + classification supervision is necessary.
- **ComBat harmonization hurts** (`combat` AUC ≈ 0.727) due to a site–diagnosis confound.
- **Best deep ensemble:** `aug_full` (AUC ≈ 0.746).

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| AUC < 0.60 on all models | Site leakage | Ensure ComBat + scaler are fit on **train only** |
| Posterior collapse (KL → 0) | KL annealing too fast | Increase the KL warm-up in `src/training/schedules.py` |
| Unstable metrics between folds | Overfitting | Use `high_dropout`, or raise dropout / weight decay |
| SHAP out-of-memory | GPU memory | Reduce background sample size / run on CPU |
| `fetch_atlas_craddock_2012` SSL / 403 | nitrc.org invalid TLS cert | Non-fatal — only the anatomical brain PNG is skipped |
| ComBat `singular matrix` | Site with too few subjects in a fold | Sites auto-merged to a fallback group |
| Torch wants a multi-GB downgrade | Exact pin conflict | Already avoided — `torch` is a floor (`>=2.0`) |

---

## Reproducibility

All randomness is seeded with **`random_state=42`**. Dependencies are pinned in
[`requirements.txt`](requirements.txt) to the versions that produced these results
(the Colab A100 run). The full reproduction command list also lives in [RUN.md](RUN.md).

---

## References

- Almuqhim & Saeed (2021). *ASD-SAENet.* Front. Comput. Neurosci.
- Eslami et al. (2019). *ASD-DiagNet.* Front. Neurosci.
- Di Martino et al. (2014). *The ABIDE dataset.* Mol. Psychiatry.
- Kingma & Welling (2013). *Auto-Encoding Variational Bayes.* ICLR 2014.
- Fortin et al. (2018). *ComBat harmonization for fMRI.* NeuroImage.
- Craddock et al. (2012). *CC200 functional parcellation atlas.* Hum. Brain Mapp.
