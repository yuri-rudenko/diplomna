"""
Diploma Results Export Script
Collects all results into diploma_export/ and builds FULL_RESULTS.md.

Run from the repository root in one command:
    python export_results.py

Reads the live results/ and data/processed/ trees of this repo. The classical
logreg baseline flows through the main results table, per-fold table and the
all-experiments summary; it is excluded from the training-dynamics section
(it has no training history).
"""

import os
import json
import shutil
import glob
from pathlib import Path
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed"
EXPORT = ROOT / "diploma_export"

EXPERIMENTS = [
    "adamw", "aug_full", "aug_mixup", "aug_noise", "baseline",
    "combat", "deep", "high_dropout", "latent_128", "latent_32"
]

# ─── Step 1: Create folder structure ──────────────────────────────────────────
print("=" * 60)
print("STEP 1: Creating export folder structure")
print("=" * 60)
for sub in ["figures", "data", "shap", "checkpoints_info"]:
    (EXPORT / sub).mkdir(parents=True, exist_ok=True)
    print(f"  Created: diploma_export/{sub}/")

# ─── Step 2: Copy figures ─────────────────────────────────────────────────────
print("\nSTEP 2: Copying figures")
png_count = 0
# 2a) Global figures (results/figures/*.png)
for png in sorted((RESULTS / "figures").glob("*.png")):
    dst = EXPORT / "figures" / png.name
    shutil.copy2(png, dst)
    print(f"  Copied: {png.name}")
    png_count += 1

# 2b) Per-experiment figures often missed in export (confusion/pr/roc/etc.)
exp_figure_names = [
    "confusion_matrices.png",
    "pr_curves.png",
    "roc_curves.png",
    "metrics_bars.png",
    "training_curves.png",
]
for exp in EXPERIMENTS:
    exp_fig_dir = RESULTS / "experiments" / exp / "figures"
    if not exp_fig_dir.exists():
        continue
    for fname in exp_figure_names:
        src = exp_fig_dir / fname
        if not src.exists():
            continue
        dst_name = f"{exp}_{fname}"
        dst = EXPORT / "figures" / dst_name
        shutil.copy2(src, dst)
        print(f"  Copied: {dst_name}")
        png_count += 1
print(f"  Total PNG files copied: {png_count}")

# ─── Step 3: Copy CSV/JSON tables ─────────────────────────────────────────────
print("\nSTEP 3: Copying CSV/JSON tables")

def copy_if_exists(src, dst, label=""):
    src = Path(src)
    dst = Path(dst)
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Copied: {label or dst.name} ({src.stat().st_size} B)")
        return True
    else:
        print(f"  NOT FOUND: {src}")
        return False

copy_if_exists(RESULTS/"experiments"/"comparison_all.csv",
               EXPORT/"data"/"comparison_all.csv")
copy_if_exists(RESULTS/"metrics"/"cv_per_fold.csv",
               EXPORT/"data"/"global_cv_per_fold.csv")
copy_if_exists(RESULTS/"metrics"/"cv_per_fold_all_experiments.csv",
               EXPORT/"data"/"cv_per_fold_all_experiments.csv")
copy_if_exists(RESULTS/"metrics"/"two_stage_vae.csv",
               EXPORT/"data"/"two_stage_vae.csv")
copy_if_exists(RESULTS/"metrics"/"significance_baseline.csv",
               EXPORT/"data"/"significance_baseline.csv")
for sig_vs in sorted((RESULTS / "metrics").glob("significance_*_vs_*_by_model.csv")):
    copy_if_exists(sig_vs, EXPORT / "data" / sig_vs.name)
copy_if_exists(DATA/"phenotype.csv",
               EXPORT/"data"/"phenotype.csv")
copy_if_exists(DATA/"dataset_stats.json",
               EXPORT/"data"/"dataset_stats.json")

for exp in EXPERIMENTS:
    exp_dir = RESULTS / "experiments" / exp / "metrics"
    copy_if_exists(exp_dir/"summary.csv",        EXPORT/"data"/f"{exp}_summary.csv")
    copy_if_exists(exp_dir/"cv_per_fold.csv",    EXPORT/"data"/f"{exp}_cv_per_fold.csv")
    copy_if_exists(exp_dir/"config.json",        EXPORT/"data"/f"{exp}_config.json")

# ─── Step 4: Copy SHAP files ──────────────────────────────────────────────────
print("\nSTEP 4: Copying SHAP files")
for pattern in ["shap_top20_*.csv", "roi_importance_*.npy", "shap_values_*.npy"]:
    for f in sorted((RESULTS / "figures").glob(pattern)):
        shutil.copy2(f, EXPORT / "shap" / f.name)
        print(f"  Copied: {f.name}")

# ─── Step 5: Copy history JSONs ───────────────────────────────────────────────
print("\nSTEP 5: Copying training history JSONs")
hist_count = 0
for exp in EXPERIMENTS:
    ckpt_dir = RESULTS / "experiments" / exp / "checkpoints"
    for hf in sorted(ckpt_dir.glob("*_history.json")):
        shutil.copy2(hf, EXPORT / "checkpoints_info" / hf.name)
        hist_count += 1
print(f"  Total history JSONs copied: {hist_count}")

# ─── Step 6: Build FULL_RESULTS.md ────────────────────────────────────────────
print("\nSTEP 6: Building FULL_RESULTS.md")

lines = []

def h(text): lines.append(text)
def nl(): lines.append("")
def rule(): lines.append("---"); nl()

# ── Header ────────────────────────────────────────────────────────────────────
h("# DIPLOMA RESULTS — Full Export")
h(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
h("Project: ASD fMRI Autoencoder Comparison (ABIDE I, n=871, 5-fold CV)")
nl(); rule()

# ── Section 1: Dataset Statistics ─────────────────────────────────────────────
h("## 1. DATASET STATISTICS")
nl()
ds_path = EXPORT / "data" / "dataset_stats.json"
if ds_path.exists():
    ds = json.loads(ds_path.read_text(encoding="utf-8"))
    h(f"| Metric | Value |")
    h(f"|--------|-------|")
    h(f"| Total N | {ds['n_total']} |")
    h(f"| ASD | {ds['n_asd']} ({ds['n_asd']/ds['n_total']*100:.1f}%) |")
    h(f"| Control | {ds['n_control']} ({ds['n_control']/ds['n_total']*100:.1f}%) |")
    h(f"| Sites | {ds['n_sites']} |")
    h(f"| Features (FC upper triangle) | {ds['n_features']} |")
    nl()
    h("### Age Statistics")
    nl()
    h("| Group | Mean ± SD | Min | Max | N |")
    h("|-------|-----------|-----|-----|---|")
    for grp, key in [("ALL", "age_all"), ("ASD", "age_asd"), ("Control", "age_control")]:
        a = ds[key]
        h(f"| {grp} | {a['mean']:.2f} ± {a['std']:.2f} | {a['min']:.2f} | {a['max']:.2f} | {a['n']} |")
    nl()
    h("### Sex Distribution")
    nl()
    h("| Group | Male | Female |")
    h("|-------|------|--------|")
    h(f"| ASD | {ds['sex_asd_male']} | {ds['sex_asd_female']} |")
    h(f"| Control | {ds['sex_control_male']} | {ds['sex_control_female']} |")
    h(f"| Total | {ds['sex_total_male']} | {ds['sex_total_female']} |")
    nl()
    h("### Site Distribution")
    nl()
    h("| Site | Total | ASD | Control |")
    h("|------|-------|-----|---------|")
    for site, vals in ds["site_table"].items():
        h(f"| {site} | {vals['total']} | {vals['asd']} | {vals['control']} |")
else:
    h("**NOT FOUND**: dataset_stats.json")
nl(); rule()

# ── Section 2: Model Parameters ───────────────────────────────────────────────
h("## 2. MODEL PARAMETERS")
nl()
cfg_path = EXPORT / "data" / "baseline_config.json"
params = {"sae": 20701374, "vae": 20717822, "attention_vae": 20735486}
cfg_data = None
if cfg_path.exists():
    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "model_params" in cfg_data:
        params = cfg_data["model_params"]
    cfg = cfg_data.get("experiment", {})
    input_dim = 19900
    h("| Parameter | Value |")
    h("|-----------|-------|")
    h(f"| input_dim | {input_dim} |")
    h(f"| hidden_dims | {cfg.get('hidden_dims', [512, 256])} |")
    h(f"| latent_dim | {cfg.get('latent_dim', 64)} |")
    h(f"| dropout | {cfg.get('dropout', 0.3)} |")
else:
    h("| Parameter | Value |")
    h("|-----------|-------|")
    h("| input_dim | 19900 |")
    h("| hidden_dims | [512, 256] |")
    h("| latent_dim | 64 |")
    h("| dropout | 0.3 |")
nl()
h("### Trainable Parameters")
nl()
h("| Model | Parameters |")
h("|-------|-----------|")
for m, p in params.items():
    h(f"| {m} | {p:,} |")
nl(); rule()

# ── Section 3: Experiment Configurations ──────────────────────────────────────
h("## 3. EXPERIMENT CONFIGURATIONS")
nl()
h("| experiment | optimizer | augment | noise_std | use_mixup | label_smoothing | harmonize | hidden_dims | latent_dim | dropout | n_epochs | lr | cls_weight_max |")
h("|------------|-----------|---------|-----------|-----------|-----------------|-----------|-------------|------------|---------|----------|----|----------------|")
for exp in EXPERIMENTS:
    cp = EXPORT / "data" / f"{exp}_config.json"
    if cp.exists():
        cd = json.loads(cp.read_text(encoding="utf-8"))
        c = cd.get("experiment", {})
        h(f"| {exp} | {c.get('optimizer','adam')} | {c.get('augment',False)} | "
          f"{c.get('noise_std',0.02)} | {c.get('use_mixup',False)} | "
          f"{c.get('label_smoothing',0.0)} | {c.get('harmonize',False)} | "
          f"{c.get('hidden_dims',[512,256])} | {c.get('latent_dim',64)} | "
          f"{c.get('dropout',0.3)} | {c.get('n_epochs',150)} | "
          f"{c.get('lr',0.001)} | {c.get('cls_weight_max',5.0)} |")
    else:
        h(f"| {exp} | NOT FOUND | | | | | | | | | | | |")
nl(); rule()

# ── Section 4: Main Results Table (baseline) ──────────────────────────────────
h("## 4. MAIN RESULTS TABLE (baseline experiment)")
nl()
bp = EXPORT / "data" / "baseline_summary.csv"
if bp.exists():
    rows = bp.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    h("| " + " | ".join(header) + " |")
    h("|" + "|".join(["---"]*len(header)) + "|")
    for row in rows[1:]:
        h("| " + " | ".join(row.split(",")) + " |")
else:
    h("**NOT FOUND**: baseline_summary.csv")
nl(); rule()

# ── Section 5: Per-fold Results (baseline) ────────────────────────────────────
h("## 5. PER-FOLD RESULTS (baseline)")
nl()
fp = EXPORT / "data" / "global_cv_per_fold.csv"
if fp.exists():
    rows = fp.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    h("| " + " | ".join(header) + " |")
    h("|" + "|".join(["---"]*len(header)) + "|")
    for row in rows[1:]:
        cols = row.split(",")
        h("| " + " | ".join(f"{float(c):.4f}" if i < len(cols)-3 else c
                             for i, c in enumerate(cols)) + " |")
else:
    h("**NOT FOUND**: global_cv_per_fold.csv")
nl(); rule()

# ── Section 6: All Experiments Summary ────────────────────────────────────────
h("## 6. ALL EXPERIMENTS SUMMARY")
nl()
cp_all = EXPORT / "data" / "comparison_all.csv"
if cp_all.exists():
    import csv
    with open(cp_all, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    def auc_val(row):
        return float(row["auc"].replace("±", "").split()[0].strip())

    ensemble_rows = [r for r in all_rows if r["model"] == "ensemble"]
    ensemble_rows_sorted = sorted(ensemble_rows, key=lambda r: auc_val(r), reverse=True)

    h("### Full Table (sorted by ensemble AUC descending)")
    nl()
    h("| experiment | model | auc | f1 | balanced_acc | sensitivity | specificity |")
    h("|------------|-------|-----|-----|-------------|------------|------------|")
    for exp in [r["experiment"] for r in ensemble_rows_sorted]:
        for row in [r for r in all_rows if r["experiment"] == exp]:
            h(f"| {row['experiment']} | {row['model']} | {row['auc']} | "
              f"{row['f1']} | {row['balanced_acc']} | {row['sensitivity']} | {row['specificity']} |")
    nl()

    h("### Key Comparisons")
    nl()
    def ens_auc(exp_name):
        rows = [r for r in all_rows if r["experiment"] == exp_name and r["model"] == "ensemble"]
        return auc_val(rows[0]) if rows else None

    def model_auc(exp_name, model):
        rows = [r for r in all_rows if r["experiment"] == exp_name and r["model"] == model]
        return auc_val(rows[0]) if rows else None

    best_ens = max(ensemble_rows, key=lambda r: auc_val(r))
    h(f"- **Best experiment by ensemble AUC**: {best_ens['experiment']} = {auc_val(best_ens):.4f}")

    for model in ["sae", "vae", "attention_vae", "logreg"]:
        model_rows = [r for r in all_rows if r["model"] == model]
        if not model_rows:
            continue
        best_m = max(model_rows, key=lambda r: auc_val(r))
        h(f"- **Best experiment by {model} AUC**: {best_m['experiment']} = {auc_val(best_m):.4f}")

    baseline_ens = ens_auc("baseline")
    for exp in ["aug_full", "aug_noise", "combat"]:
        val = ens_auc(exp)
        if val and baseline_ens:
            delta = val - baseline_ens
            sign = "+" if delta >= 0 else ""
            h(f"- **{exp} vs baseline ensemble delta**: {sign}{delta:+.4f} ({val:.4f} vs {baseline_ens:.4f})")

    nl()
    h("### Latent Dimension Comparison")
    nl()
    for ld in ["latent_32", "baseline", "latent_128"]:
        val = ens_auc(ld)
        label = ld if ld != "baseline" else "latent_64 (baseline)"
        h(f"- {label}: ensemble AUC = {val:.4f}" if val else f"- {label}: NOT FOUND")
else:
    h("**NOT FOUND**: comparison_all.csv")
nl(); rule()

# ── Section 7: Ablation Findings ──────────────────────────────────────────────
h("## 7. ABLATION FINDINGS")
nl()
if cp_all.exists():
    def fmt(exp, model="ensemble"):
        v = model_auc(exp, model) if model != "ensemble" else ens_auc(exp)
        return f"{v:.4f}" if v else "N/A"

    bl = ens_auc("baseline") or 0
    an = ens_auc("aug_noise") or 0
    af = ens_auc("aug_full") or 0
    am = ens_auc("aug_mixup") or 0
    cb = ens_auc("combat") or 0
    l32 = ens_auc("latent_32") or 0
    l64 = bl
    l128 = ens_auc("latent_128") or 0
    aw = ens_auc("adamw") or 0

    h(f"- **Augmentation (noise)**: aug_noise ensemble AUC {an:.4f} vs baseline {bl:.4f} "
      f"(delta = {an-bl:+.4f})")
    h(f"- **Augmentation (full)**: aug_full ensemble AUC {af:.4f} vs baseline {bl:.4f} "
      f"(delta = {af-bl:+.4f})")
    h(f"- **Augmentation (mixup)**: aug_mixup ensemble AUC {am:.4f} vs baseline {bl:.4f} "
      f"(delta = {am-bl:+.4f})")
    h(f"- **ComBat harmonization**: combat ensemble AUC {cb:.4f} vs baseline {bl:.4f} "
      f"(delta = {cb-bl:+.4f})")
    h(f"- **Latent dimension**: latent_32={l32:.4f}, latent_64={l64:.4f}, latent_128={l128:.4f}")
    h(f"- **Optimizer (AdamW vs Adam)**: adamw={aw:.4f}, baseline(adam)={bl:.4f} "
      f"(delta = {aw-bl:+.4f})")

    # Architecture spread within baseline
    sae_bl = model_auc("baseline", "sae") or 0
    vae_bl = model_auc("baseline", "vae") or 0
    att_bl = model_auc("baseline", "attention_vae") or 0
    spread = max(sae_bl, vae_bl, att_bl) - min(sae_bl, vae_bl, att_bl)
    h(f"- **Architecture spread (baseline)**: SAE={sae_bl:.4f}, VAE={vae_bl:.4f}, "
      f"AttVAE={att_bl:.4f} — max spread = {spread:.4f}")
else:
    h("**NOT FOUND**: comparison_all.csv")
nl(); rule()

# ── Section 8: Statistical Significance ───────────────────────────────────────
h("## 8. STATISTICAL SIGNIFICANCE")
nl()
sig_path = EXPORT / "data" / "significance_baseline.csv"
sig_vs_paths = sorted((EXPORT / "data").glob("significance_*_vs_*_by_model.csv"))

has_any_sig = False
if sig_path.exists():
    has_any_sig = True
    h("### Model vs model (baseline / pooled mode)")
    nl()
    rows = sig_path.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    h("| " + " | ".join(header) + " |")
    h("|" + "|".join(["---"]*len(header)) + "|")
    for row in rows[1:]:
        h("| " + " | ".join(row.split(",")) + " |")
    nl()

for sig_vs in sig_vs_paths:
    has_any_sig = True
    h(f"### Experiment vs experiment ({sig_vs.name})")
    nl()
    rows = sig_vs.read_text(encoding="utf-8").strip().splitlines()
    if rows:
        header = rows[0].split(",")
        h("| " + " | ".join(header) + " |")
        h("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows[1:]:
            h("| " + " | ".join(row.split(",")) + " |")
    nl()

if not has_any_sig:
    h("**NOT FOUND** — significance files do not exist.")
    h("To compute model-vs-model: `python -m src.training.significance --exp baseline`")
    h("To compute exp-vs-exp: `python -m src.training.significance --exp-a baseline --exp-b aug_full`")
nl(); rule()

# ── Section 9: Two-Stage Ablation ─────────────────────────────────────────────
h("## 9. TWO-STAGE ABLATION")
nl()
ts_path = EXPORT / "data" / "two_stage_vae.csv"
if ts_path.exists():
    import csv as csv_mod
    with open(ts_path, encoding="utf-8") as f:
        ts_rows = list(csv_mod.DictReader(f))

    h("### Two-Stage VAE per-fold results")
    nl()
    ts_header = list(ts_rows[0].keys())
    h("| " + " | ".join(ts_header) + " |")
    h("|" + "|".join(["---"]*len(ts_header)) + "|")
    aucs = []
    for row in ts_rows:
        h("| " + " | ".join(row[k] for k in ts_header) + " |")
        try:
            aucs.append(float(row["auc"]))
        except Exception:
            pass
    nl()
    if aucs:
        import statistics
        ts_mean = statistics.mean(aucs)
        ts_std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        h(f"- **Two-stage VAE AUC**: {ts_mean:.4f} ± {ts_std:.4f}")
        joint_auc = model_auc("baseline", "vae") if cp_all.exists() else None
        if joint_auc:
            h(f"- **Joint VAE AUC (baseline)**: {joint_auc:.4f}")
            h(f"- **Delta (joint - two-stage)**: {joint_auc - ts_mean:+.4f}")
else:
    h("**NOT FOUND**: two_stage_vae.csv")
nl(); rule()

# ── Section 10: Training Dynamics ─────────────────────────────────────────────
h("## 10. TRAINING DYNAMICS")
nl()
h("| experiment | model | fold | best_val_auc | best_epoch | total_epochs | final_train_loss |")
h("|------------|-------|------|-------------|------------|-------------|-----------------|")

dynamics = {}
for hf in sorted((EXPORT / "checkpoints_info").glob("*_history.json")):
    name = hf.stem  # e.g. baseline_sae_fold0_history -> we split differently
    # Pattern: {exp}_{model}_{fold}_history  — but exp and model can have underscores
    # Model names: sae, vae, attention_vae
    # Try to extract fold first
    stem = hf.stem  # e.g. "baseline_sae_fold0_history"
    # Remove trailing "_history"
    if stem.endswith("_history"):
        stem = stem[:-8]
    # Extract fold
    fold_idx = stem.rfind("_fold")
    if fold_idx == -1:
        continue
    fold = stem[fold_idx+5:]
    prefix = stem[:fold_idx]  # e.g. "baseline_sae"
    # Extract model (last part of prefix matching known models)
    model = None
    for m in ["attention_vae", "vae", "sae"]:
        if prefix.endswith("_" + m):
            model = m
            exp = prefix[:-(len(m)+1)]
            break
    if model is None:
        continue

    try:
        data = json.loads(hf.read_text(encoding="utf-8"))
        best_auc = data.get("best_val_auc", max(data["val_auc"]))
        val_auc_list = data["val_auc"]
        best_epoch = val_auc_list.index(max(val_auc_list))
        total_epochs = len(data["train_loss"])
        final_loss = data["train_loss"][-1]
        h(f"| {exp} | {model} | {fold} | {best_auc:.4f} | {best_epoch} | "
          f"{total_epochs} | {final_loss:.4f} |")
        if model not in dynamics:
            dynamics[model] = []
        dynamics[model].append(best_epoch)
    except Exception as e:
        h(f"| {exp} | {model} | {fold} | ERROR: {e} | | | |")

nl()
h("### Convergence Summary")
nl()
for model, epochs in dynamics.items():
    if epochs:
        avg = sum(epochs) / len(epochs)
        h(f"- **{model}**: average best epoch = {avg:.1f} (across {len(epochs)} runs)")

if dynamics:
    fastest = min(dynamics.items(), key=lambda kv: sum(kv[1])/len(kv[1]))
    h(f"\n- **Fastest converging model**: {fastest[0]} "
      f"(avg best epoch = {sum(fastest[1])/len(fastest[1]):.1f})")
nl(); rule()

# ── Section 11: SHAP / XAI Results ───────────────────────────────────────────
h("## 11. SHAP / XAI RESULTS")
nl()
network_counts = {}
top1_per_model = {}
top5_sets = {}

for shap_csv in sorted((EXPORT / "shap").glob("shap_top20_*.csv")):
    h(f"### {shap_csv.stem}")
    nl()
    rows_shap = shap_csv.read_text(encoding="utf-8").strip().splitlines()
    header_s = rows_shap[0].split(",")
    h("| " + " | ".join(header_s) + " |")
    h("|" + "|".join(["---"]*len(header_s)) + "|")

    # Determine model from filename
    fn = shap_csv.stem  # shap_top20_sae_fold3
    parts = fn.split("_")
    # Find "fold" marker
    fold_i = next((i for i, p in enumerate(parts) if p.startswith("fold")), None)
    if fold_i:
        model_parts = parts[2:fold_i]
        model_key = "_".join(model_parts)
    else:
        model_key = fn

    top5_rois = []
    for i, row in enumerate(rows_shap[1:], 1):
        cols = row.split(",")
        h("| " + " | ".join(cols) + " |")
        if len(cols) >= 3:
            net = cols[2].strip()
            roi = cols[1].strip()
            network_counts[net] = network_counts.get(net, 0) + 1
            if i == 1:
                top1_per_model[model_key] = (roi, net)
            if i <= 5:
                top5_rois.append(roi)

    top5_sets[model_key] = set(top5_rois)
    nl()

h("### Network Distribution (top-20 across all models)")
nl()
if network_counts:
    total_n = sum(network_counts.values())
    h("| Yeo-7 Network | Count | % |")
    h("|---------------|-------|---|")
    for net, cnt in sorted(network_counts.items(), key=lambda x: -x[1]):
        h(f"| {net} | {cnt} | {cnt/total_n*100:.1f}% |")
    most_common = max(network_counts.items(), key=lambda x: x[1])
    nl()
    h(f"- **Most common network**: {most_common[0]} ({most_common[1]} / {total_n} top-20 ROIs)")

nl()
h("### Top-1 ROI per model")
nl()
for model_key, (roi, net) in top1_per_model.items():
    h(f"- **{model_key}**: ROI #{roi} ({net})")

nl()
h("### Top-5 ROI intersection across all models")
nl()
if len(top5_sets) >= 2:
    intersection = set.intersection(*top5_sets.values()) if top5_sets else set()
    if intersection:
        h(f"ROIs in top-5 for ALL models: {sorted(intersection)}")
    else:
        h("No ROI appears in top-5 for ALL three models.")
nl(); rule()

# ── Section 12: Figure List ───────────────────────────────────────────────────
h("## 12. LIST OF ALL GENERATED FIGURES")
nl()
fig_descriptions = {
    "training_curves_detailed": "Training loss + val AUC curves",
    "roc_combined": "ROC curves for all 3 models",
    "ablation_auc_heatmap": "Heatmap: all experiments × models AUC",
    "ablation_delta_bar": "Delta AUC vs baseline per experiment",
    "shap_top20_bar": "SHAP top-20 ROI bar chart",
    "shap_network_pie": "Yeo-7 network distribution of top-20 SHAP ROIs",
    "dataset_stats": "Dataset demographics overview",
    "umap": "UMAP latent space visualization",
    "confusion": "Confusion matrix",
    "pr_curve": "Precision-recall curve",
}

h("| Filename | Description |")
h("|----------|-------------|")
for png in sorted((EXPORT / "figures").glob("*.png")):
    desc = "Visualization"
    for key, val in fig_descriptions.items():
        if key in png.stem:
            desc = val
            break
    h(f"| {png.name} | {desc} |")
nl(); rule()

# ── Section 13: Key Numbers Cheat Sheet ──────────────────────────────────────
h("## 13. KEY NUMBERS FOR REPORT TEXT")
nl()
h("```")
h(f"N total subjects:         {ds['n_total'] if ds_path.exists() else 'N/A'}")
h(f"N ASD:                    {ds['n_asd'] if ds_path.exists() else 'N/A'}")
h(f"N Control:                {ds['n_control'] if ds_path.exists() else 'N/A'}")
asd_pct = f"{ds['n_asd']/ds['n_total']*100:.1f}%" if ds_path.exists() else "N/A"
h(f"ASD %:                    {asd_pct}")
h(f"N sites:                  {ds['n_sites'] if ds_path.exists() else 20}")
h(f"CV strategy:              Stratified 5-fold by (site, label)")
h(f"Input features:           19,900 (upper triangle of 200×200 FC matrix)")
h(f"SAE parameters:           20,701,374")
h(f"VAE parameters:           20,717,822")
h(f"AttVAE parameters:        20,735,486")
h("")
if cp_all.exists():
    best_model_bl = max(
        [r for r in all_rows if r["experiment"] == "baseline" and r["model"] != "ensemble"],
        key=lambda r: auc_val(r)
    )
    h("BASELINE RESULTS:")
    h(f"  Best single model (AUC): {best_model_bl['model']} = {auc_val(best_model_bl):.4f}")
    ens_bl_row = next(r for r in all_rows if r["experiment"] == "baseline" and r["model"] == "ensemble")
    h(f"  Ensemble AUC:            {ens_bl_row['auc']}")
    single_aucs = [auc_val(r) for r in all_rows if r["experiment"] == "baseline" and r["model"] != "ensemble"]
    h(f"  AUC range (all models):  {min(single_aucs):.4f} – {max(single_aucs):.4f}")
    h(f"  Max inter-model spread:  {max(single_aucs)-min(single_aucs):.4f}")
    h("")
    h("BEST OVERALL CONFIG:")
    h(f"  Experiment:              {best_ens['experiment']}")
    h(f"  Ensemble AUC:            {auc_val(best_ens):.4f}")
    h(f"  vs baseline:             {auc_val(best_ens)-bl:+.4f}")
h("")
if network_counts:
    mc = max(network_counts.items(), key=lambda x: x[1])
    h("SHAP:")
    h(f"  Most important network:  {mc[0]} ({mc[1]} / {sum(network_counts.values())} top-20 ROIs)")
    if "attention_vae" in top1_per_model:
        roi_av, net_av = top1_per_model["attention_vae"]
        h(f"  Top-1 ROI (AttVAE):      ROI #{roi_av} ({net_av})")
h("```")
nl()

# ── Write file ─────────────────────────────────────────────────────────────────
out_path = EXPORT / "FULL_RESULTS.md"
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  Written: FULL_RESULTS.md ({len(lines)} lines)")

# ─── Step 7: Create ZIP ───────────────────────────────────────────────────────
print("\nSTEP 7: Creating ZIP archive")
zip_path = ROOT / "diploma_export"
shutil.make_archive(str(zip_path), "zip", str(ROOT), "diploma_export")
zip_file = ROOT / "diploma_export.zip"
size_mb = zip_file.stat().st_size / (1024 * 1024)
print(f"  Created: diploma_export.zip ({size_mb:.1f} MB)")

# ─── Step 8: Verify completeness ──────────────────────────────────────────────
print("\nSTEP 8: Verification")
print(f"  FULL_RESULTS.md lines:        {len(lines)}")
print(f"  figures/ files:               {len(list((EXPORT/'figures').glob('*')))}")
print(f"  data/ files:                  {len(list((EXPORT/'data').glob('*')))}")
print(f"  shap/ files:                  {len(list((EXPORT/'shap').glob('*')))}")
print(f"  checkpoints_info/ files:      {len(list((EXPORT/'checkpoints_info').glob('*')))}")
print(f"  ZIP size:                     {size_mb:.1f} MB")
print("\nExport complete!")
