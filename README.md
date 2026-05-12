# Comparative Analysis of Deep Learning Architectures for ASD Diagnosis Using fMRI Data

Diploma thesis project: classifying Autism Spectrum Disorder (ASD) from
resting-state fMRI using and comparing deep generative / autoencoder models.

## Dataset

- **ABIDE I** (Autism Brain Imaging Data Exchange), preprocessed release.
- Pipeline: **CPAC** (band-pass filtered, no global signal regression,
  quality-checked subjects only).
- Derivative: **rois_cc200** — mean BOLD time-series for 200 functional
  ROIs from the Craddock-200 atlas.
- Accessed via `nilearn.datasets.fetch_abide_pcp`.

## Goal

Train and **compare three autoencoder architectures** for binary ASD vs.
control classification, with a focus on **explainability** of the learned
latent representations:

1. **Sparse Autoencoder (Sparse AE)** — baseline.
2. **Variational Autoencoder (VAE)** — probabilistic latent space.
3. **Attention-VAE** — attention-augmented variant for highlighting the
   ROI / connectivity features that drive predictions.

Inputs are the 200×200 functional connectivity matrices computed from the
ROI time-series.

## Project structure

```
diploma/
├── data/                # ABIDE downloads (gitignored)
├── src/
│   └── visualize_sample.py
├── notebooks/           # exploratory notebooks
├── results/             # metrics, model checkpoints
├── visualizations/      # figures (.png + interactive .html)
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. install dependencies
python -m pip install -r requirements.txt

# 2. download a 50-subject sample and produce the exploratory figures
python src/visualize_sample.py
```

The first run downloads roughly tens of MB of ROI time-series files into
`data/ABIDE_pcp/` (cached by `nilearn` — re-runs are instant).

## Outputs

Running `visualize_sample.py` writes the following into `visualizations/`
(each as both a static `.png` and an interactive Plotly `.html`):

| File | Description |
| --- | --- |
| `subject_overview.*` | Group counts, age histogram, sex distribution |
| `connectivity_single.*` | 200×200 connectivity matrix for one subject |
| `connectivity_comparison.*` | Mean ASD vs. mean Control + difference matrix |
| `timeseries.*` | BOLD signal of 5 ROIs for one ASD + one Control subject |
| `embedding_2d.*` | PCA / UMAP scatter of subjects, colored by diagnosis |

A summary (subject count, ASD/Control split, time-series shape, output
path) is printed at the end of the run.

## Reproducibility

All randomness is seeded with `random_state=42`.
