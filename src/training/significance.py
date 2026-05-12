"""
Paired t-test for statistical significance between model pairs.

Reads results/metrics/cv_per_fold.csv and computes pairwise p-values.

Usage:
    python -m src.training.significance
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS_DIR = Path(__file__).resolve().parents[2] / "results" / "metrics"


def run_significance(metric: str = "auc") -> pd.DataFrame:
    cv_path = METRICS_DIR / "cv_per_fold.csv"
    if not cv_path.exists():
        raise FileNotFoundError(f"Run compare_models first: {cv_path} not found")

    df = pd.read_csv(cv_path)
    models = df["model"].unique().tolist()
    rows = []

    for m1, m2 in combinations(models, 2):
        vals1 = df[df["model"] == m1][metric].values
        vals2 = df[df["model"] == m2][metric].values
        t_stat, p_val = stats.ttest_rel(vals1, vals2)
        rows.append({
            "model_a": m1,
            "model_b": m2,
            "metric": metric,
            f"mean_{m1}": vals1.mean(),
            f"mean_{m2}": vals2.mean(),
            "t_stat": t_stat,
            "p_value": p_val,
            "significant_p05": p_val < 0.05,
        })

    result = pd.DataFrame(rows)
    out = METRICS_DIR / "significance.csv"
    result.to_csv(out, index=False)

    print(f"\nPaired t-test results ({metric}):")
    print(result[["model_a", "model_b", f"mean_{models[0]}", f"mean_{models[1]}", "p_value", "significant_p05"]].to_string(index=False))
    print(f"\nSaved → {out}")
    return result


if __name__ == "__main__":
    run_significance(metric="auc")
