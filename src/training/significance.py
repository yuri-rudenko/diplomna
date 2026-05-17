"""
Paired t-test for statistical significance between model pairs.

Reads cv_per_fold.csv from any experiment or from the global metrics dir.

Usage:
    # After compare_models.py:
    python -m src.training.significance

    # After run_experiments.py (pick a specific experiment):
    python -m src.training.significance --exp baseline

    # Compare best models across all experiments:
    python -m src.training.significance --exp baseline aug_full
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
EXPERIMENTS_DIR = PROJECT_DIR / "results" / "experiments"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _load_cv_data(exp_names: list[str] | None = None) -> pd.DataFrame:
    """
    Load cv_per_fold.csv from one or more experiments (or global metrics dir).

    Priority:
      1. Specific experiments named via exp_names
      2. Global results/metrics/cv_per_fold.csv (compare_models.py output)
    """
    if exp_names:
        dfs = []
        for name in exp_names:
            p = EXPERIMENTS_DIR / name / "metrics" / "cv_per_fold.csv"
            if p.exists():
                df = pd.read_csv(p)
                df["experiment"] = name
                dfs.append(df)
            else:
                print(f"  [warn] {p} not found — skipping {name}")
        if dfs:
            return pd.concat(dfs, ignore_index=True)

    # Fallback: global metrics dir (compare_models.py)
    global_path = METRICS_DIR / "cv_per_fold.csv"
    if global_path.exists():
        return pd.read_csv(global_path)

    raise FileNotFoundError(
        "No cv_per_fold.csv found. Run one of:\n"
        "  python -m src.compare_models\n"
        "  python run_experiments.py --exp baseline\n"
        "  python -m src.training.significance --exp baseline"
    )


def run_significance(
    metric: str = "auc",
    exp_names: list[str] | None = None,
    tag: str = "",
) -> pd.DataFrame:
    """
    Compute paired t-tests between all model pairs for a given metric.

    Parameters
    ----------
    metric    : column name in cv_per_fold.csv to test
    exp_names : list of experiment names from run_experiments.py, or None for global
    tag       : suffix for the output filename (e.g. experiment name)
    """
    df = _load_cv_data(exp_names)
    models = df["model"].unique().tolist()
    rows = []

    for m1, m2 in combinations(models, 2):
        vals1 = df[df["model"] == m1].sort_values("fold")[metric].values
        vals2 = df[df["model"] == m2].sort_values("fold")[metric].values

        # Ensure equal length (some models may have different folds if partial run)
        min_len = min(len(vals1), len(vals2))
        if min_len < 2:
            print(f"  [warn] {m1} vs {m2}: not enough folds ({min_len}) — skipping")
            continue

        t_stat, p_val = stats.ttest_rel(vals1[:min_len], vals2[:min_len])
        rows.append({
            "model_a":          m1,
            "model_b":          m2,
            "metric":           metric,
            f"mean_{m1}":       float(vals1.mean()),
            f"mean_{m2}":       float(vals2.mean()),
            "delta":            float(vals1.mean() - vals2.mean()),
            "t_stat":           float(t_stat),
            "p_value":          float(p_val),
            "significant_p05":  bool(p_val < 0.05),
            "significant_p01":  bool(p_val < 0.01),
        })

    result = pd.DataFrame(rows)

    suffix = f"_{tag}" if tag else ""
    out = METRICS_DIR / f"significance{suffix}.csv"
    result.to_csv(out, index=False)

    print(f"\nPaired t-test results ({metric}):")
    print(result[["model_a", "model_b", "t_stat", "p_value",
                  "significant_p05", "significant_p01"]].to_string(index=False))
    print(f"\nSaved → {out}")
    return result


def run_experiment_vs_experiment(
    exp_a: str,
    exp_b: str,
    metric: str = "auc",
) -> pd.DataFrame:
    """
    Compare two experiments for each model with paired fold-wise tests.

    This is different from run_significance(), which compares model-vs-model.
    """
    df_a = _load_cv_data([exp_a])
    df_b = _load_cv_data([exp_b])

    rows = []
    model_names = sorted(set(df_a["model"].unique()) | set(df_b["model"].unique()))

    for model_name in model_names:
        a_vals = df_a[df_a["model"] == model_name].sort_values("fold")[metric].values
        b_vals = df_b[df_b["model"] == model_name].sort_values("fold")[metric].values

        min_len = min(len(a_vals), len(b_vals))
        if min_len < 2:
            print(
                f"  [warn] {model_name}: not enough paired folds for "
                f"{exp_a} vs {exp_b} ({min_len}) — skipping"
            )
            continue

        a_vals = a_vals[:min_len]
        b_vals = b_vals[:min_len]

        t_stat, p_val = stats.ttest_rel(a_vals, b_vals)

        try:
            w_stat, w_p = stats.wilcoxon(a_vals, b_vals)
            w_stat = float(w_stat)
            w_p = float(w_p)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")

        delta = float(b_vals.mean() - a_vals.mean())  # exp_b - exp_a
        rows.append({
            "model": model_name,
            "metric": metric,
            "experiment_a": exp_a,
            "experiment_b": exp_b,
            "n_folds": int(min_len),
            f"mean_{exp_a}": float(a_vals.mean()),
            f"mean_{exp_b}": float(b_vals.mean()),
            "delta_b_minus_a": delta,
            "t_stat": float(t_stat),
            "p_value_ttest": float(p_val),
            "wilcoxon_stat": w_stat,
            "p_value_wilcoxon": w_p,
            "significant_ttest_p05": bool(p_val < 0.05),
            "significant_ttest_p01": bool(p_val < 0.01),
        })

    result = pd.DataFrame(rows)
    out = METRICS_DIR / f"significance_{exp_a}_vs_{exp_b}_by_model.csv"
    result.to_csv(out, index=False)

    print(f"\nExperiment-vs-experiment paired tests ({metric}): {exp_a} vs {exp_b}")
    if not result.empty:
        print(
            result[
                ["model", "n_folds", "delta_b_minus_a", "p_value_ttest", "p_value_wilcoxon"]
            ].to_string(index=False)
        )
    else:
        print("  [warn] No comparable model/fold pairs found.")
    print(f"\nSaved → {out}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="auc",
                        help="Metric column to test (default: auc)")
    parser.add_argument("--exp", nargs="+", default=None,
                        help="Experiment name(s) from run_experiments.py")
    parser.add_argument("--exp-a", default=None,
                        help="Experiment A for experiment-vs-experiment test")
    parser.add_argument("--exp-b", default=None,
                        help="Experiment B for experiment-vs-experiment test")
    args = parser.parse_args()

    if args.exp_a or args.exp_b:
        if not (args.exp_a and args.exp_b):
            raise ValueError("Use both --exp-a and --exp-b together.")
        run_experiment_vs_experiment(
            exp_a=args.exp_a,
            exp_b=args.exp_b,
            metric=args.metric,
        )
    else:
        tag = "_".join(args.exp) if args.exp else ""
        run_significance(metric=args.metric, exp_names=args.exp, tag=tag)
