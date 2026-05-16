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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="auc",
                        help="Metric column to test (default: auc)")
    parser.add_argument("--exp", nargs="+", default=None,
                        help="Experiment name(s) from run_experiments.py")
    args = parser.parse_args()

    tag = "_".join(args.exp) if args.exp else ""
    run_significance(metric=args.metric, exp_names=args.exp, tag=tag)
