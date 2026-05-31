"""
Generate Precision-Recall curves from saved fold predictions.

Works only with saved files:
  results/experiments/{exp}/figures/probs_{exp}_{model}_fold{k}.npy
  results/experiments/{exp}/figures/labels_{exp}_{model}_fold{k}.npy

No training or checkpoint loading is performed.

Usage:
    python -m src.visualizations.pr_curves --exp baseline --cv 5
    python -m src.visualizations.pr_curves --cv 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_DIR / "results" / "experiments"

MODEL_NAMES = ["sae", "vae", "attention_vae", "logreg", "ensemble"]
MODEL_COLORS = {
    "sae": "#1f77b4",
    "vae": "#ff7f0e",
    "attention_vae": "#2ca02c",
    "logreg": "#9467bd",
    "ensemble": "#d62728",
}


def _iter_experiments(selected: list[str] | None = None) -> list[str]:
    if selected:
        return selected
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(
        d.name for d in EXPERIMENTS_DIR.iterdir()
        if d.is_dir() and (d / "figures").exists()
    )


def plot_pr_experiment(exp_name: str, n_cv: int = 5) -> bool:
    """Create PR plot for one experiment. Returns True if saved."""
    figs_dir = EXPERIMENTS_DIR / exp_name / "figures"
    if not figs_dir.exists():
        print(f"  [skip] {exp_name}: figures dir not found ({figs_dir})")
        return False

    recall_grid = np.linspace(0.0, 1.0, 200)
    fig, ax = plt.subplots(figsize=(7, 6))
    has_any = False

    for model_name in MODEL_NAMES:
        curves = []
        aps = []

        for k in range(n_cv):
            tag = f"{exp_name}_{model_name}_fold{k}"
            probs_path = figs_dir / f"probs_{tag}.npy"
            labels_path = figs_dir / f"labels_{tag}.npy"
            if not (probs_path.exists() and labels_path.exists()):
                continue

            probs = np.load(probs_path)
            labels = np.load(labels_path)
            precision, recall, _ = precision_recall_curve(labels, probs)

            order = np.argsort(recall)
            recall_sorted = recall[order]
            precision_sorted = precision[order]

            curves.append(np.interp(recall_grid, recall_sorted, precision_sorted))
            aps.append(average_precision_score(labels, probs))

        if not curves:
            continue

        has_any = True
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        color = MODEL_COLORS.get(model_name, "#777777")

        ax.plot(
            recall_grid,
            mean_curve,
            color=color,
            lw=2,
            label=f"{model_name} (AP={np.mean(aps):.3f}±{np.std(aps):.3f})",
        )
        ax.fill_between(
            recall_grid,
            np.clip(mean_curve - std_curve, 0, 1),
            np.clip(mean_curve + std_curve, 0, 1),
            color=color,
            alpha=0.15,
        )

    if not has_any:
        plt.close(fig)
        print(f"  [skip] {exp_name}: no probs/labels fold files found")
        return False

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Precision-Recall — {exp_name} ({n_cv}-fold CV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()

    out = figs_dir / "pr_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PR curves from saved fold predictions."
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        default=None,
        help="Experiment names (default: all experiments found).",
    )
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    exp_names = _iter_experiments(args.exp)
    if not exp_names:
        print("No experiment directories found in results/experiments/")
        return

    print(f"Generating PR curves for: {exp_names}")
    saved_count = 0
    for exp_name in exp_names:
        print(f"\n── {exp_name} ──")
        if plot_pr_experiment(exp_name, n_cv=args.cv):
            saved_count += 1

    print(f"\nDone. PR curves saved for {saved_count}/{len(exp_names)} experiments.")


if __name__ == "__main__":
    main()
