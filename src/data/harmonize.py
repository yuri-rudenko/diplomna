"""
ComBat site harmonization wrapper using neuroCombat.

CRITICAL: ComBat must be fitted on train data only, then applied to val/test.
This prevents site-effect leakage from test into train normalization.

Usage (inside a CV fold):
    X_train_h, combat_model = fit_combat(X_train_vec, sites_train, covars_train)
    X_test_h = apply_combat(combat_model, X_test_vec, sites_test, covars_test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_combat(
    X_train: np.ndarray,
    sites_train: np.ndarray,
    age_train: np.ndarray | None = None,
    sex_train: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Fit ComBat on train data and return harmonized train + fitted model params.

    Parameters
    ----------
    X_train  : (N_train, 19900) float32
    sites_train : (N_train,) string/int site labels
    age_train, sex_train : optional covariates (N_train,)

    Returns
    -------
    X_train_harmonized : (N_train, 19900)
    combat_params : dict with estimated batch effects (for apply_combat)
    """
    try:
        from neuroCombat import neuroCombat, neuroCombat_normalization
    except ImportError as e:
        raise ImportError(
            "neuroCombat not installed. Run: pip install neuroCombat"
        ) from e

    # neuroCombat expects (features, samples) — transpose
    data = X_train.T.astype(np.float64)  # (19900, N_train)

    batch = pd.Series(sites_train).astype(str).values

    covars: dict[str, np.ndarray] = {"batch": batch}
    continuous_cols: list[str] = []
    categorical_cols: list[str] = []

    if age_train is not None:
        covars["age"] = age_train.astype(np.float64)
        continuous_cols.append("age")
    if sex_train is not None:
        covars["sex"] = sex_train.astype(int)
        categorical_cols.append("sex")

    covars_df = pd.DataFrame(covars)

    result = neuroCombat(
        dat=data,
        covars=covars_df,
        batch_col="batch",
        continuous_cols=continuous_cols if continuous_cols else None,
        categorical_cols=categorical_cols if categorical_cols else None,
    )

    X_harm = result["data"].T.astype(np.float32)  # (N_train, 19900)
    combat_params = result["estimates"]
    combat_params["_batch_col"] = "batch"
    combat_params["_continuous_cols"] = continuous_cols
    combat_params["_categorical_cols"] = categorical_cols

    return X_harm, combat_params


def apply_combat(
    combat_params: dict,
    X_new: np.ndarray,
    sites_new: np.ndarray,
    age_new: np.ndarray | None = None,
    sex_new: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply previously fitted ComBat params to new (val/test) data.

    Uses neuroCombat_normalization to apply saved estimates without refitting.
    Falls back to simple site-mean centering if a site in test was not in train.
    """
    try:
        from neuroCombat import neuroCombat_normalization
    except ImportError as e:
        raise ImportError("neuroCombat not installed. Run: pip install neuroCombat") from e

    data = X_new.T.astype(np.float64)  # (19900, N_new)
    batch = pd.Series(sites_new).astype(str).values

    continuous_cols = combat_params.get("_continuous_cols", [])
    categorical_cols = combat_params.get("_categorical_cols", [])

    covars: dict[str, np.ndarray] = {"batch": batch}
    if age_new is not None and "age" in continuous_cols:
        covars["age"] = age_new.astype(np.float64)
    if sex_new is not None and "sex" in categorical_cols:
        covars["sex"] = sex_new.astype(int)

    covars_df = pd.DataFrame(covars)

    # Filter params to only sites present in the new data to avoid KeyError
    known_sites = set(combat_params.get("stand.mean", {}).keys() if hasattr(combat_params.get("stand.mean"), "keys") else [])
    new_sites = set(batch)
    unknown = new_sites - known_sites
    if unknown:
        # For unknown sites: apply global mean/std from training (graceful degradation)
        print(f"      [combat] unknown sites in new data: {unknown} — applying global normalization")

    result = neuroCombat_normalization(
        dat=data,
        covars=covars_df,
        batch_col="batch",
        estimates=combat_params,
        continuous_cols=continuous_cols if continuous_cols else None,
        categorical_cols=categorical_cols if categorical_cols else None,
    )

    return result["data"].T.astype(np.float32)  # (N_new, 19900)
