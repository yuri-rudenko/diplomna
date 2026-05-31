"""
One-command XAI pipeline for a trained experiment.

Runs, in a single process, everything the diploma's explainability section
needs — with no hand-written glue:

  1. DeepSHAP for all three models at fold='best'
     (harmonize is auto-set to match the experiment config, e.g. the `combat`
      experiment used ComBat → harmonize=True).
  2. Brain glass-brain + stat-map projection of ROI importance for each model.
  3. Attention-weight heatmap for attention_vae.

All atlas downloads go through the SSL-robust fetch wrapper (nitrc.org ships an
invalid certificate), so this works unattended on local / Colab / Kaggle.

Usage:
    python -m src.xai.run_xai --exp aug_full
    python -m src.xai.run_xai --exp combat            # harmonize auto-enabled
    python -m src.xai.run_xai --exp aug_full --models sae vae attention_vae
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

ALL_MODELS = ["sae", "vae", "attention_vae"]


def _harmonize_for(exp_name: str) -> bool:
    """Look up the experiment's harmonize flag (source of truth = experiments.py)."""
    from src.experiments import EXPERIMENTS

    cfg = EXPERIMENTS.get(exp_name)
    if cfg is None:
        print(f"  [warn] experiment '{exp_name}' not found in src/experiments.py "
              f"— assuming harmonize=False")
        return False
    return bool(cfg.harmonize)


def run_xai(
    exp_name: str,
    models: list[str] | None = None,
    n_background: int = 100,
) -> None:
    from src.xai.shap_explain import explain_model
    from src.xai.roi_brain import plot_attention_heatmap, plot_brain_importance
    from src.utils.checkpoints import best_fold

    models = models or ALL_MODELS
    harmonize = _harmonize_for(exp_name)

    print("\n" + "=" * 60)
    print(f"  XAI PIPELINE — experiment: {exp_name}")
    print(f"  models: {models}  |  harmonize: {harmonize}  |  fold: best")
    print("=" * 60)

    for model_type in models:
        print(f"\n── SHAP: {model_type} ──")
        result = explain_model(
            model_type=model_type,
            fold="best",
            n_background=n_background,
            exp_name=exp_name,
            harmonize=harmonize,
        )

        fold_idx = best_fold(model_type, exp_name)
        roi_importance = result["roi_importance"]

        print(f"\n── Brain maps: {model_type} (fold {fold_idx}) ──")
        try:
            plot_brain_importance(roi_importance, model_type=model_type, fold_idx=fold_idx)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] brain map failed for {model_type}: "
                  f"{type(exc).__name__}: {exc}")

    if "attention_vae" in models:
        att_fold = best_fold("attention_vae", exp_name)
        print(f"\n── Attention heatmap: attention_vae (fold {att_fold}) ──")
        try:
            plot_attention_heatmap("attention_vae", att_fold, exp_name=exp_name)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] attention heatmap failed: {type(exc).__name__}: {exc}")

    print("\nXAI pipeline complete. Artifacts → results/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-command SHAP + brain maps + attention heatmap for an experiment."
    )
    parser.add_argument("--exp", required=True,
                        help="Experiment name from run_experiments.py (e.g. aug_full)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=ALL_MODELS,
                        help="Subset of models (default: all three)")
    parser.add_argument("--n-background", type=int, default=100)
    args = parser.parse_args()

    run_xai(args.exp, models=args.models, n_background=args.n_background)
