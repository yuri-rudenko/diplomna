"""
Generate all charts needed for the diploma report that are not produced
by run_experiments.py --plot-only.

Reads only from saved .npy / .csv / .json files — no model loading, no retraining.

Charts produced:
  results/figures/
    ablation_auc_heatmap.png
    ablation_grouped_bar.png
    ablation_delta_bar.png
    shap_top20_bar_{model}_fold{k}.png     (for each saved shap_top20 csv)
    shap_network_pie.png
    training_curves_detailed_{model}.png   (per-fold lines + mean + phase boundary)
    roc_combined.png                       (all folds concatenated per model, one figure)
    dataset_stats.png                      (bar chart of subjects per site)

Usage:
    python -m src.visualizations.report_charts
    python -m src.visualizations.report_charts --exp baseline aug_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_DIR / "results" / "figures"
EXPERIMENTS_DIR = PROJECT_DIR / "results" / "experiments"
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES  = ["sae", "vae", "attention_vae"]
MODEL_LABELS = {"sae": "SAE", "vae": "VAE", "attention_vae": "Attention-VAE"}
MODEL_COLORS = {
    "sae":           "#e15759",
    "vae":           "#4e79a7",
    "attention_vae": "#59a14f",
    "ensemble":      "#f28e2b",
}
YEO7_COLORS = {
    "Visual":         "#781286",
    "Somatomotor":    "#4682B4",
    "DorsAttn":       "#00760E",
    "VentAttn":       "#C43AFA",
    "Limbic":         "#DCF8A4",
    "Frontoparietal": "#E69422",
    "Default":        "#CD3E4E",
    "Unassigned":     "#7F7F7F",
    "Unknown":        "#AAAAAA",
}


# ---------------------------------------------------------------------------
# 1. Ablation heatmap
# ---------------------------------------------------------------------------

def plot_ablation_heatmap(comparison_csv: Path | None = None) -> None:
    path = comparison_csv or EXPERIMENTS_DIR / "comparison_all.csv"
    if not path.exists():
        print(f"  [skip] ablation heatmap — {path} not found. Run: python run_experiments.py --compare")
        return

    df = pd.read_csv(path)

    def _mean(s):
        try:
            return float(str(s).split("±")[0].strip().split(" ")[0])
        except Exception:
            return float("nan")

    df["auc_mean"] = df["auc"].apply(_mean)

    pivot = df.pivot_table(index="experiment", columns="model",
                           values="auc_mean", aggfunc="first")
    # Order models
    col_order = [m for m in MODEL_NAMES + ["ensemble"] if m in pivot.columns]
    pivot = pivot[col_order]

    # Sort experiments by ensemble AUC descending
    if "ensemble" in pivot.columns:
        pivot = pivot.sort_values("ensemble", ascending=False)

    fig, ax = plt.subplots(figsize=(len(col_order) * 1.6 + 1, len(pivot) * 0.55 + 1.5))
    vmin = max(0.5, float(pivot.min().min()) - 0.02)
    vmax = float(pivot.max().max()) + 0.01
    im = ax.imshow(pivot.values, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([MODEL_LABELS.get(c, c) for c in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("AUC (mean over folds) — Ablation Study", fontsize=12, pad=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if val < (vmin + vmax) / 2 else "black",
                        fontweight="bold" if val == pivot.values[:, j].max() else "normal")

    plt.colorbar(im, ax=ax, label="AUC", fraction=0.03, pad=0.02)
    fig.tight_layout()
    out = FIGURES_DIR / "ablation_auc_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# 2. Ablation delta bar (vs baseline)
# ---------------------------------------------------------------------------

def plot_ablation_delta(comparison_csv: Path | None = None) -> None:
    path = comparison_csv or EXPERIMENTS_DIR / "comparison_all.csv"
    if not path.exists():
        print(f"  [skip] ablation delta — {path} not found")
        return

    df = pd.read_csv(path)

    def _mean(s):
        try:
            return float(str(s).split("±")[0].strip().split(" ")[0])
        except Exception:
            return float("nan")

    df["auc_mean"] = df["auc"].apply(_mean)

    if "baseline" not in df["experiment"].values:
        print("  [skip] ablation delta — 'baseline' experiment not found in comparison_all.csv")
        return

    baseline = df[df["experiment"] == "baseline"].set_index("model")["auc_mean"]
    experiments = [e for e in df["experiment"].unique() if e != "baseline"]
    models_to_plot = [m for m in MODEL_NAMES + ["ensemble"] if m in baseline.index]

    fig, axes = plt.subplots(1, len(models_to_plot),
                              figsize=(4 * len(models_to_plot), 5), sharey=False)
    if len(models_to_plot) == 1:
        axes = [axes]

    for ax, mname in zip(axes, models_to_plot):
        base_val = baseline.get(mname, float("nan"))
        deltas = []
        exp_labels = []
        for exp in experiments:
            row = df[(df["experiment"] == exp) & (df["model"] == mname)]
            if row.empty:
                continue
            val = _mean(row["auc"].values[0])
            deltas.append(val - base_val)
            exp_labels.append(exp)

        colors = ["#59a14f" if d >= 0 else "#e15759" for d in deltas]
        bars = ax.barh(exp_labels, deltas, color=colors, alpha=0.85)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("ΔAUC vs baseline")
        ax.set_title(MODEL_LABELS.get(mname, mname), fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        for bar, d in zip(bars, deltas):
            ax.text(d + (0.001 if d >= 0 else -0.001), bar.get_y() + bar.get_height() / 2,
                    f"{d:+.3f}", va="center", ha="left" if d >= 0 else "right", fontsize=7)

    fig.suptitle("AUC improvement vs baseline", fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / "ablation_delta_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# 3. SHAP top-20 bar chart (colored by Yeo-7 network)
# ---------------------------------------------------------------------------

def plot_shap_top20_bar() -> None:
    """One horizontal bar chart per saved shap_top20_*.csv file."""
    csv_files = sorted(FIGURES_DIR.glob("shap_top20_*.csv"))
    if not csv_files:
        print("  [skip] shap top-20 bar — no shap_top20_*.csv files found")
        return

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        tag = csv_path.stem.replace("shap_top20_", "")
        networks = df["yeo7_network"].values
        shap_vals = df["mean_abs_shap"].values
        roi_idx = df["roi_index"].values

        bar_colors = [YEO7_COLORS.get(n, "#AAAAAA") for n in networks]
        labels = [f"ROI {idx}  ({net})" for idx, net in zip(roi_idx, networks)]

        fig, ax = plt.subplots(figsize=(9, 7))
        y_pos = np.arange(len(df))
        ax.barh(y_pos, shap_vals, color=bar_colors, alpha=0.88)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"Top-20 ROIs by SHAP importance\n{tag}", fontsize=11)
        ax.grid(axis="x", alpha=0.3)

        # Legend for Yeo-7 networks present
        unique_nets = list(dict.fromkeys(networks))
        patches = [mpatches.Patch(color=YEO7_COLORS.get(n, "#AAAAAA"), label=n)
                   for n in unique_nets]
        ax.legend(handles=patches, loc="lower right", fontsize=8,
                  title="Yeo-7 network", title_fontsize=8)

        fig.tight_layout()
        out = FIGURES_DIR / f"shap_top20_bar_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# 4. SHAP network distribution pie (one per model)
# ---------------------------------------------------------------------------

def plot_shap_network_pie() -> None:
    """Pie charts showing Yeo-7 network distribution in top-20 ROIs per model."""
    model_csvs: dict[str, list[pd.DataFrame]] = {}
    for csv_path in sorted(FIGURES_DIR.glob("shap_top20_*.csv")):
        for mname in MODEL_NAMES:
            if f"_{mname}_" in csv_path.name or csv_path.name.endswith(f"_{mname}.csv"):
                model_csvs.setdefault(mname, []).append(pd.read_csv(csv_path))

    if not model_csvs:
        print("  [skip] shap network pie — no shap_top20 files found")
        return

    n = len(model_csvs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (mname, dfs) in zip(axes, model_csvs.items()):
        combined = pd.concat(dfs, ignore_index=True)
        counts = combined["yeo7_network"].value_counts()
        pie_colors = [YEO7_COLORS.get(n, "#AAAAAA") for n in counts.index]
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            colors=pie_colors,
            autopct="%1.0f%%",
            startangle=90,
            pctdistance=0.75,
        )
        for t in texts:
            t.set_fontsize(8)
        for at in autotexts:
            at.set_fontsize(7)
        ax.set_title(f"{MODEL_LABELS.get(mname, mname)}\nTop-20 ROI network distribution",
                     fontsize=10)

    fig.suptitle("Yeo-7 network distribution in SHAP top-20 ROIs", fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / "shap_network_pie.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# 5. Enhanced training curves (per-fold + mean + phase boundary)
# ---------------------------------------------------------------------------

def plot_training_curves_detailed(
    exp_name: str = "baseline",
    n_cv: int = 5,
) -> None:
    """Per-fold training curves with bold mean line and red phase-2 boundary."""
    ckpt_dir = EXPERIMENTS_DIR / exp_name / "checkpoints"
    if not ckpt_dir.exists():
        print(f"  [skip] training curves — {ckpt_dir} not found")
        return

    for mname in MODEL_NAMES:
        histories = []
        for k in range(n_cv):
            hist_path = ckpt_dir / f"{exp_name}_{mname}_fold{k}_history.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    histories.append(json.load(f))

        if not histories:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        max_epochs = max(len(h["train_loss"]) for h in histories)

        for h in histories:
            ax1.plot(h["train_loss"], alpha=0.3,
                     color=MODEL_COLORS.get(mname, "#888"), lw=1)
            ax2.plot(h["val_auc"],   alpha=0.3,
                     color=MODEL_COLORS.get(mname, "#888"), lw=1)

        # Mean lines
        min_len = min(len(h["train_loss"]) for h in histories)
        mean_loss = np.mean([h["train_loss"][:min_len] for h in histories], axis=0)
        mean_auc  = np.mean([h["val_auc"][:min_len]   for h in histories], axis=0)
        ax1.plot(mean_loss, color=MODEL_COLORS.get(mname, "#888"), lw=2.5, label="mean")
        ax2.plot(mean_auc,  color=MODEL_COLORS.get(mname, "#888"), lw=2.5, label="mean")

        # Phase-2 boundary
        phase_ep = histories[0].get("cls_weight", [])
        if phase_ep:
            boundary = next((i for i, cw in enumerate(phase_ep) if cw > 0), None)
            if boundary:
                ax1.axvline(boundary, color="red", ls="--", lw=1.2, label="joint phase")
                ax2.axvline(boundary, color="red", ls="--", lw=1.2, label="joint phase")

        ax1.set_title(f"{MODEL_LABELS.get(mname, mname)} — Train Loss")
        ax1.set_xlabel("Epoch"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        ax2.set_title(f"{MODEL_LABELS.get(mname, mname)} — Val AUC")
        ax2.set_xlabel("Epoch"); ax2.set_ylim(0.4, 1.0)
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        fig.suptitle(f"{exp_name} — {MODEL_LABELS.get(mname, mname)} "
                     f"({len(histories)} folds)", fontsize=11)
        fig.tight_layout()
        out = FIGURES_DIR / f"training_curves_detailed_{exp_name}_{mname}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# 6. Combined ROC (all folds concatenated per model — one figure per experiment)
# ---------------------------------------------------------------------------

def plot_roc_combined(exp_name: str = "baseline", n_cv: int = 5) -> None:
    """ROC using all fold predictions concatenated — single clean curve per model."""
    from sklearn.metrics import roc_curve, roc_auc_score

    figs_dir = EXPERIMENTS_DIR / exp_name / "figures"
    if not figs_dir.exists():
        print(f"  [skip] combined ROC — {figs_dir} not found")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC=0.50)")

    for mname in MODEL_NAMES + ["ensemble"]:
        all_probs, all_labels = [], []
        for k in range(n_cv):
            tag = f"{exp_name}_{mname}_fold{k}"
            pp = figs_dir / f"probs_{tag}.npy"
            ll = figs_dir / f"labels_{tag}.npy"
            if pp.exists() and ll.exists():
                all_probs.extend(np.load(pp))
                all_labels.extend(np.load(ll))
        if not all_probs:
            continue
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc_val = roc_auc_score(all_labels, all_probs)
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(mname, "#888"), lw=2,
                label=f"{MODEL_LABELS.get(mname, mname)} (AUC={auc_val:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC — {exp_name} (all folds combined)", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / f"roc_combined_{exp_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# 7. Dataset stats bar chart (subjects per site, colored by ASD/Control)
# ---------------------------------------------------------------------------

def plot_dataset_stats() -> None:
    stats_path = PROCESSED_DIR / "dataset_stats.json"
    if not stats_path.exists():
        print(f"  [skip] dataset stats chart — {stats_path} not found")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    site_table = stats.get("site_table", {})
    if not site_table:
        return

    sites = sorted(site_table.keys())
    asd_counts  = [site_table[s]["asd"]     for s in sites]
    ctrl_counts = [site_table[s]["control"] for s in sites]
    x = np.arange(len(sites))

    fig, ax = plt.subplots(figsize=(max(10, len(sites) * 0.65), 5))
    ax.bar(x, ctrl_counts, label="Control", color="#4e79a7", alpha=0.9)
    ax.bar(x, asd_counts,  bottom=ctrl_counts, label="ASD", color="#e15759", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of subjects")
    ax.set_title(f"ABIDE I — Subjects per site  "
                 f"(N={stats['n_total']}, ASD={stats['n_asd']}, "
                 f"Control={stats['n_control']})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "dataset_stats.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def generate_all(exp_names: list[str] | None = None, n_cv: int = 5) -> None:
    """Generate every report chart from saved data."""
    print("\n" + "=" * 60)
    print("GENERATING REPORT CHARTS")
    print("=" * 60)

    # Charts that depend on experiment data
    exps = exp_names or []
    if not exps and EXPERIMENTS_DIR.exists():
        exps = [d.name for d in sorted(EXPERIMENTS_DIR.iterdir()) if d.is_dir()]

    for exp in exps:
        print(f"\n── {exp} ──")
        plot_training_curves_detailed(exp, n_cv)
        plot_roc_combined(exp, n_cv)

    # Charts from global data
    print("\n── Ablation charts ──")
    plot_ablation_heatmap()
    plot_ablation_delta()

    print("\n── SHAP charts ──")
    plot_shap_top20_bar()
    plot_shap_network_pie()

    print("\n── Dataset stats ──")
    plot_dataset_stats()

    print(f"\nAll report charts saved → {FIGURES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all diploma report charts from saved .npy/.csv/.json"
    )
    parser.add_argument("--exp", nargs="+", default=None,
                        help="Experiment name(s) for training curve + ROC charts "
                             "(default: all found in results/experiments/)")
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()
    generate_all(exp_names=args.exp, n_cv=args.cv)
