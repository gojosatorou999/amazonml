"""Exact reimplementation of the challenge metric: macro-averaged F_0.5.

F_0.5 is computed per Source-1 entity and then averaged over every Source-1
entity in the evaluation set, singletons included.  An entity with no true
matches scores 1.0 for a correctly predicted empty list and 0.0 otherwise.

Algebraic note used throughout the codebase:

    F_beta = (1 + b^2) * TP / (|pred| + b^2 * |truth|)          with b^2 = 0.25

which is what `decide.expected_f` takes expectations over.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

BETA2 = 0.25  # beta = 0.5


def f_beta_set(pred: set[str], truth: set[str], beta2: float = BETA2) -> float:
    """F_beta for a single Source-1 entity."""
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    return (1.0 + beta2) * tp / (len(pred) + beta2 * len(truth))


@dataclass
class ScoreBreakdown:
    """Macro F_0.5 plus the sub-populations we actually tune against."""

    macro_f: float
    n: int
    singleton_f: float          # entities whose truth set is empty
    n_singletons: int
    matched_f: float            # entities with >= 1 true match
    n_matched: int
    precision_micro: float
    recall_micro: float

    def __str__(self) -> str:
        return (
            f"macro_F0.5={self.macro_f:.5f}  (n={self.n})\n"
            f"  singletons  F={self.singleton_f:.5f}  n={self.n_singletons}\n"
            f"  has-match   F={self.matched_f:.5f}  n={self.n_matched}\n"
            f"  micro P={self.precision_micro:.5f}  R={self.recall_micro:.5f}"
        )


def score(
    pred: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
) -> ScoreBreakdown:
    """Score predictions against ground truth.

    `truth` defines the evaluation universe: every key in it is scored, and a
    Source-1 entity missing from `pred` is treated as an empty prediction (the
    real leaderboard rejects the submission instead, but for local validation
    scoring it as an empty list is the informative behaviour).
    """
    per_entity: list[tuple[float, bool]] = []
    tp_total = fp_total = fn_total = 0

    for s1, raw_truth in truth.items():
        t = set(raw_truth)
        p = set(pred.get(s1, ()))
        per_entity.append((f_beta_set(p, t), bool(t)))
        tp_total += len(p & t)
        fp_total += len(p - t)
        fn_total += len(t - p)

    n = len(per_entity)
    singles = [f for f, has in per_entity if not has]
    matched = [f for f, has in per_entity if has]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731

    return ScoreBreakdown(
        macro_f=mean([f for f, _ in per_entity]),
        n=n,
        singleton_f=mean(singles),
        n_singletons=len(singles),
        matched_f=mean(matched),
        n_matched=len(matched),
        precision_micro=tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0,
        recall_micro=tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0,
    )
