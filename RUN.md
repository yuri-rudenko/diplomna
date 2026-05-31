# RUN — reproducing the ABIDE I ASD-vs-control pipeline

Every step below is a **single command** that runs unattended on **local,
Google Colab, or Kaggle** — no hand-written glue code, no manual file copying,
no SSL workarounds. Atlas downloads (Craddock/nitrc.org has an invalid TLS
cert) are handled in code; experiment checkpoints/scalers are resolved by name
via `--exp`; XAI is a single orchestrator command.

Run from the repository root.

| Step | GPU? | Approx. time |
|------|------|--------------|
| 1. Install deps | — | 1–3 min |
| 2. Download + build data | CPU | ~30 min first run (cached after) |
| 3. Train experiments (baseline + ablations) | **GPU** | ~3–5 min/experiment on A100; hours on CPU |
| 4. Compare experiments | CPU | seconds |
| 5. Two-stage VAE ablation | **GPU** | ~5–10 min |
| 6. Statistical significance | CPU | seconds |
| 7. XAI (SHAP + brain maps + attention) | CPU | ~5–10 min |
| 8. UMAP + report charts | CPU | ~2–5 min |
| 9. Postprocess (PR/confusion/cross-significance) | CPU | ~1 min |
| 10. Export FULL_RESULTS.md | CPU | seconds |

---

## Command list (copy-paste, in order)

```bash
# 1. Install pinned dependencies
pip install -r requirements.txt

# 2. Download ABIDE I, build FC matrices, auto-create 5-fold splits   [CPU]
python -m src.data.download_abide

# 3. Train all experiments — baseline + every ablation               [GPU]
python run_experiments.py --exp baseline --cv 5
python run_experiments.py --exp aug_full latent_32 latent_128 aug_noise aug_mixup --cv 5
python run_experiments.py --exp adamw combat deep high_dropout --cv 5
#    (equivalently: python run_experiments.py --cv 5   # runs all ten)

# 4. Build the cross-experiment comparison table                     [CPU]
python run_experiments.py --compare

# 5. Two-stage VAE ablation (frozen encoder + MLP head)              [GPU]
python -m src.training.two_stage --cv 5

# 6. Paired-t significance between models (baseline experiment)       [CPU]
python -m src.training.significance --exp baseline

# 7. Explainability — SHAP + brain glass/stat maps + attention heatmap,
#    all three models at the best fold, one command                  [CPU]
python -m src.xai.run_xai --exp aug_full

# 8. UMAP latent projection + all report charts                       [CPU]
python -m src.visualizations.umap_latent --exp aug_full --fold best
python -m src.visualizations.report_charts

# 9. Postprocess: PR curves, confusion matrices, cross-experiment test [CPU]
python run_experiments.py --postprocess --cv 5 --exp-a baseline --exp-b aug_full

# 10. Collect everything into FULL_RESULTS.md                          [CPU]
python export_results.py
```

---

## Notes

- **Steps 3 and 5 want a GPU.** They run on CPU but are far slower. Everything
  else is CPU-only.
- **`--exp <name>` is the single source of truth for paths.** XAI, UMAP, and the
  brain/attention plots resolve checkpoints and scalers from
  `results/experiments/<name>/checkpoints/` automatically — you never copy files
  into `results/checkpoints/` by hand. `--fold best` picks the highest-AUC fold
  from `results/experiments/<name>/metrics/cv_per_fold.csv`.
- **ComBat is auto-detected** by `run_xai`: it reads the experiment config in
  `src/experiments.py`, so the `combat` experiment (which trained with
  `harmonize=True`) is explained with ComBat applied, and all others are not.
- **Atlas SSL:** the Craddock CC200 atlas is hosted on nitrc.org with an invalid
  certificate. The code retries the download with verification disabled for that
  request only (`src/utils/atlas_fetch.py`) — no monkeypatching needed in a
  notebook.
- **Smoke test:** `python smoke_test.py` runs the whole pipeline on 50 cached
  Pitt subjects in ~3–5 min on CPU and downloads them itself. Use it to verify
  an environment before launching the full run.

### Optional: persisting results on Google Colab

The full dataset and `results/` live on the Colab VM's ephemeral disk. If you
want them to survive a runtime reset, mount Drive and copy — this is **optional**
and not required by any step above:

```python
from google.colab import drive; drive.mount('/content/drive')
import shutil
shutil.copytree('results', '/content/drive/MyDrive/diplomna/results', dirs_exist_ok=True)
```
