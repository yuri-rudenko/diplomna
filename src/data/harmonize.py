"""
ComBat site harmonization wrapper using neuroCombat 0.2.x.

Runs ComBat on train+test combined within each fold so that site effects
are removed consistently. StandardScaler is still fitted on train only,
preserving the no-leakage guarantee for feature normalization.

This approach (ComBat on full fold data) is standard in ABIDE literature —
ComBat does not use diagnosis labels, so it cannot leak class information.

Usage:
    X_train_h, X_test_h = fit_and_apply_combat(
        X_train, X_test, sites_train, sites_test, age_train, age_test, sex_train, sex_test
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_and_apply_combat(
    X_train: np.ndarray,
    X_test: np.ndarray,
    sites_train: np.ndarray,
    sites_test: np.ndarray,
    age_train: np.ndarray | None = None,
    age_test: np.ndarray | None = None,
    sex_train: np.ndarray | None = None,
    sex_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply ComBat harmonization to train+test combined.

    Returns
    -------
    X_train_harmonized : (N_train, 19900)
    X_test_harmonized  : (N_test,  19900)
    """
    from neuroCombat import neuroCombat

    n_train = len(X_train)
    X_all    = np.vstack([X_train, X_test]).astype(np.float64)   # (N, 19900)
    sites_all = np.concatenate([sites_train, sites_test])

    batch = pd.Series(sites_all).astype(str).values

    covars: dict = {"batch": batch}
    continuous_cols: list[str] = []
    categorical_cols: list[str] = []

    if age_train is not None and age_test is not None:
        age_all = np.concatenate([age_train, age_test]).astype(np.float64)
        # Fill NaN ages with median to avoid ComBat crash
        median_age = float(np.nanmedian(age_all))
        age_all = np.where(np.isnan(age_all), median_age, age_all)
        covars["age"] = age_all
        continuous_cols.append("age")

    if sex_train is not None and sex_test is not None:
        sex_all = np.concatenate([sex_train, sex_test]).astype(int)
        covars["sex"] = sex_all
        categorical_cols.append("sex")

    covars_df = pd.DataFrame(covars)

    # Validate: each site must have ≥ 2 subjects
    site_counts = pd.Series(batch).value_counts()
    small_sites = site_counts[site_counts < 2].index.tolist()
    if small_sites:
        raise ValueError(
            f"ComBat requires ≥2 subjects per site. "
            f"Sites with <2 subjects: {small_sites}"
        )

    result = neuroCombat(
        dat=X_all.T,                                    # (19900, N)
        covars=covars_df,
        batch_col="batch",
        continuous_cols=continuous_cols if continuous_cols else None,
        categorical_cols=categorical_cols if categorical_cols else None,
    )

    X_harm = result["data"].T.astype(np.float32)        # (N, 19900)
    return X_harm[:n_train], X_harm[n_train:]


# ---------------------------------------------------------------------------
# Legacy aliases — kept so shap_explain.py still imports without error
# ---------------------------------------------------------------------------

def fit_combat(X_train, sites_train, age_train=None, sex_train=None):
    raise RuntimeError(
        "fit_combat() is deprecated. Use fit_and_apply_combat() instead."
    )


def apply_combat(combat_params, X_new, sites_new, age_new=None, sex_new=None):
    raise RuntimeError(
        "apply_combat() is deprecated. Use fit_and_apply_combat() instead."
    )
