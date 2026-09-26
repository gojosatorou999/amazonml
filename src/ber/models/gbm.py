"""LightGBM with entity-grouped cross-fitting, plus cross-fitted isotonic calibration.

Folds are grouped by Source-1 entity (never by pair): splitting pairs would put
the same entity on both sides and inflate every number downstream.  The
out-of-fold predictions are what every later stage -- pruning threshold,
context features, calibration, decision layer -- is tuned on, so nothing is
ever evaluated on data a model saw.

Test-time prediction averages the fold models (a 5-model bag), which is also
slightly better calibrated than any single fold model.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

DEFAULT_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=0,
    seed=0,
    deterministic=True,
    force_row_wise=True,
)


def fit_oof(X, y, fold_of: np.ndarray, rounds: int = 500, params: dict | None = None):
    """Return (oof predictions, fold models).  `fold_of[k]` is row k's fold id."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    oof = np.zeros(len(y), dtype=np.float64)
    models = []
    for f in np.unique(fold_of):
        tr, va = fold_of != f, fold_of == f
        ds = lgb.Dataset(X[tr], label=y[tr], free_raw_data=True)
        m = lgb.train(params, ds, num_boost_round=rounds)
        oof[va] = m.predict(X[va])
        models.append(m)
    return oof, models


def predict_bag(models, X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


def crossfit_isotonic(p: np.ndarray, y: np.ndarray, fold_of: np.ndarray):
    """Cross-fitted calibrated OOF probabilities + a final calibrator on all rows."""
    out = np.zeros_like(p)
    for f in np.unique(fold_of):
        tr, va = fold_of != f, fold_of == f
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p[tr], y[tr])
        out[va] = iso.predict(p[va])
    final = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
    return out, final


def entity_folds(entity_index: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Deterministic fold id per Source-1 row position."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(entity_index)
    perm = rng.permutation(len(uniq))
    fold_of_entity = dict(zip(uniq[perm], np.arange(len(uniq)) % n_folds))
    return np.array([fold_of_entity[e] for e in entity_index])
