"""Per-entity decision layer: pick the output set that maximises *expected* F_0.5.

Why this exists
---------------
Almost every entity-resolution pipeline ends with a single global threshold on
a pairwise match probability.  That is the wrong objective here.  The metric is
macro-averaged over Source-1 entities and uses

    F_0.5 = 1.25 * TP / (|S| + 0.25 * |T|)

where S is the set we emit and T the (unknown) true match set.  Because the
denominator contains |S|, the value of adding one more candidate depends on how
many we already emitted *for that entity* -- so the optimal cut is per-entity,
not global.  It also makes the empty set a genuine competitor: emitting nothing
scores 1.0 when T is empty, which is how singletons are won.

Given calibrated per-candidate probabilities p_i, treating matches as
independent Bernoullis, the optimal S is always a prefix of the candidates
sorted by p descending (adding a lower-probability item before a higher one can
never help).  So we only evaluate n+1 prefixes and take the argmax of

    E[F(k)] = sum_{x,y} P(X=x) P(Y=y) * 1.25x / (k + 0.25(x + y))

with X ~ PoissonBinomial(p_1..p_k)   (true matches we emitted)
     Y ~ PoissonBinomial(p_{k+1}..p_n) (true matches we withheld)

and the special case F = 1.0 when k = 0 and x + y = 0.

Both PMFs are built incrementally, so the whole sweep is O(n^2) per entity.
"""
from __future__ import annotations

import numpy as np

BETA2 = 0.25


def _pb_forward(probs: np.ndarray) -> list[np.ndarray]:
    """PMFs of PoissonBinomial(p_1..p_k) for k = 0..n (prefixes)."""
    pmfs = [np.array([1.0])]
    cur = np.array([1.0])
    for p in probs:
        nxt = np.zeros(cur.size + 1)
        nxt[:-1] += cur * (1.0 - p)
        nxt[1:] += cur * p
        cur = nxt
        pmfs.append(cur)
    return pmfs


def _pb_backward(probs: np.ndarray) -> list[np.ndarray]:
    """PMFs of PoissonBinomial(p_{k+1}..p_n) for k = 0..n (suffixes)."""
    n = probs.size
    pmfs: list[np.ndarray | None] = [None] * (n + 1)
    cur = np.array([1.0])
    pmfs[n] = cur
    for k in range(n - 1, -1, -1):
        p = probs[k]
        nxt = np.zeros(cur.size + 1)
        nxt[:-1] += cur * (1.0 - p)
        nxt[1:] += cur * p
        cur = nxt
        pmfs[k] = cur
    return pmfs  # type: ignore[return-value]


def expected_f_curve(probs: np.ndarray, beta2: float = BETA2) -> np.ndarray:
    """E[F_beta] for every prefix size k = 0..n. `probs` must be sorted desc."""
    n = probs.size
    fwd = _pb_forward(probs)
    bwd = _pb_backward(probs)
    out = np.empty(n + 1)

    for k in range(n + 1):
        px, py = fwd[k], bwd[k]
        # outer product -> joint over (x, y); small arrays, this is cheap
        joint = np.outer(px, py)
        x = np.arange(px.size)[:, None]
        y = np.arange(py.size)[None, :]
        if k == 0:
            # only the (x=0, y=0) cell scores, and it scores exactly 1.0
            out[k] = joint[0, 0]
            continue
        denom = k + beta2 * (x + y)
        f = (1.0 + beta2) * x / denom
        out[k] = float((joint * f).sum())
    return out


def choose_set(
    candidate_ids: list[str],
    probs: np.ndarray,
    *,
    max_k: int | None = None,
    min_prob: float = 0.0,
    beta2: float = BETA2,
) -> tuple[list[str], float]:
    """Return (chosen ids, expected F) maximising expected F_beta.

    `max_k` caps the sweep for speed; `min_prob` drops hopeless tail candidates
    before the sweep (they only ever lower E[F] once emitted, but they do still
    contribute to E[|T|], so prune conservatively).
    """
    if len(candidate_ids) == 0:
        return [], 1.0

    order = np.argsort(-probs, kind="stable")
    ids = [candidate_ids[i] for i in order]
    p = np.clip(probs[order].astype(float), 1e-9, 1 - 1e-9)

    if min_prob > 0.0:
        keep = p >= min_prob
        if keep.any():
            ids = [i for i, k in zip(ids, keep) if k]
            p = p[keep]
        else:
            return [], float(np.prod(1.0 - p))

    if max_k is not None:
        p = p[: max(max_k, 1) + 8]  # keep a little tail so E[|T|] stays honest
        ids = ids[: p.size]

    curve = expected_f_curve(p, beta2=beta2)
    if max_k is not None:
        curve = curve[: max_k + 1]
    k = int(np.argmax(curve))
    return ids[:k], float(curve[k])


def evpi(probs: np.ndarray, beta2: float = BETA2) -> tuple[float, int]:
    """Expected value of perfect information: how much E[F] rises if a human
    labels one candidate.  Returns (best gain, index of that candidate).

    For candidate c:  EVPI_c = p_c * max E[F | c=1] + (1-p_c) * max E[F | c=0] - max E[F].
    Ranking entities by this is active learning with the competition metric as
    the acquisition function: reviewers see the pairs where a decision moves
    the score most, not merely the ones nearest 0.5.
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-9, 1 - 1e-9)
    if p.size == 0:
        return 0.0, -1

    def best(q):
        q = np.sort(q)[::-1]
        return float(expected_f_curve(q, beta2).max())

    base = best(p)
    gains = []
    for c in range(p.size):
        hi, lo = p.copy(), p.copy()
        hi[c], lo[c] = 1 - 1e-9, 1e-9
        gains.append(p[c] * best(hi) + (1 - p[c]) * best(lo) - base)
    c = int(np.argmax(gains))
    return float(gains[c]), c
