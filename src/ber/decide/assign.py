"""Turn calibrated pair probabilities into per-entity output sets.

Two ingredients, both switchable so their value is measured, not assumed:

exclusive(p)
    Source 1 is the deduplicated reference, so a real S2/S3 record describes
    one business and has at most one true Source-1 owner (verified on the
    training ground truth by `ber.cli eda`).  If k Source-1 entities claim the
    same record with independent probabilities p_1..p_k, the mutually exclusive
    posterior over {nobody, s_1..s_k} is

        q_i = o_i / (1 + sum_j o_j),   o = p / (1 - p)

    Uncontested records are unchanged (q = p); contested ones lose mass in
    proportion to their competitors' confidence.  Pure precision gain, and
    F0.5 weights precision 2x.

decide(...)
    Per Source-1 entity, emit the prefix maximising expected F0.5
    (decide.expected_f) -- or, as the ablation baseline, a global threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .expected_f import choose_set


def exclusive(j: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    o = p / (1.0 - p)
    tot = pd.Series(o).groupby(j).transform("sum").values
    return o / (1.0 + tot)


def decide(
    s1_ids: np.ndarray,
    r_ids: np.ndarray,
    p: np.ndarray,
    mode: str = "expected_f",
    threshold: float = 0.5,
    max_k: int = 12,
) -> dict[str, list[str]]:
    """Return {s1_id: [chosen record ids]} for every s1 id that has candidates."""
    s1_ids = np.asarray(s1_ids)
    r_ids = np.asarray(r_ids)
    p = np.asarray(p, dtype=float)
    order = np.lexsort((-p, s1_ids))            # by entity, then p descending
    s_sorted, r_sorted, p_sorted = s1_ids[order], r_ids[order], p[order]
    cuts = np.flatnonzero(s_sorted[1:] != s_sorted[:-1]) + 1
    out: dict[str, list[str]] = {}
    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(order)]):
        s = s_sorted[lo]
        if mode == "threshold":
            k = int((p_sorted[lo:hi] >= threshold).sum())   # sorted desc => a prefix
            out[s] = r_sorted[lo : lo + k].tolist()
        else:
            ids, _ = choose_set(r_sorted[lo:hi].tolist(), p_sorted[lo:hi], max_k=max_k)
            out[s] = ids
    return out


def best_threshold(s1_ids, r_ids, p, truth, all_s1, grid=None):
    """Grid-search the global-threshold baseline on OOF data (for the ablation)."""
    from ..eval.metric import score

    grid = grid if grid is not None else np.round(np.arange(0.2, 0.96, 0.025), 3)
    best = (-1.0, 0.5)
    for t in grid:
        pred = decide(s1_ids, r_ids, p, mode="threshold", threshold=t)
        f = score({s: pred.get(s, []) for s in all_s1}, {s: truth.get(s, set()) for s in all_s1}).macro_f
        if f > best[0]:
            best = (f, float(t))
    return best
