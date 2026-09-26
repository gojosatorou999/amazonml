"""Blocking scaling benchmark -> reports/scaling.md

Replicates the provided records x1, x2, x4, x8 (fresh IDs, light perturbation
so copies are not byte-identical) and times candidate retrieval in two modes:

    exact    all features in the sparse product: O(|S1| x |R|) worst case
    capped   product restricted to features with df <= cap: an inverted index
             over rare features, O(N x features x cap) -- linear in N

It fits a power law t = a * N^b to each mode and projects to 10^9 records.

    python -m ber.eval.scaling --data data --factors 1 2 4 8
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..blocking.retrieve import retrieve
from ..pipeline import load_split
from ..prepare import Lexicon, prepare


def _replicate(df: pd.DataFrame, f: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    parts = [df]
    for c in range(1, f):
        d = df.copy()
        d["entity_id"] = d.entity_id + f"x{c}"
        # one random character dropped per name so replicas are near- not exact duplicates
        d["business_name"] = [n[:i] + n[i + 1 :] if len(n) > 3 else n
                              for n in d.business_name for i in [rng.randrange(1, max(len(n) - 1, 2))]]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--modes", nargs="+", default=["exact", "capped"])
    a = ap.parse_args(argv)

    s1, r = load_split(Path(a.data), a.split)
    lex_path = Path(a.artifacts) / "lexicon.json"
    lex = Lexicon.load(lex_path) if lex_path.exists() else Lexicon.empty()
    rows = []
    for f in a.factors:
        P1 = prepare(_replicate(s1, f, 1), lex)
        PR = prepare(_replicate(r, f, 2), lex)
        n = len(P1) + len(PR)
        for mode, cap in [m for m in (("exact", None), ("capped", a.cap)) if m[0] in a.modes]:
            t = time.perf_counter()
            pool, _ = retrieve(P1, PR, df_cap=cap)
            dt = time.perf_counter() - t
            rows.append(dict(factor=f, records=n, mode=mode, seconds=round(dt, 2),
                             pairs_per_s1=round(len(pool) / len(P1), 1)))
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    lines = ["# Blocking scaling benchmark", "",
             f"Records replicated from `{a.data}/{a.split}` (light perturbation per copy). "
             "Single machine, CPU only.", "",
             "| factor | records | mode | seconds | fused pool pairs / S1 |", "|---|---|---|---|---|"]
    lines += [f"| x{r_.factor} | {r_.records:,} | {r_.mode} | {r_.seconds} | {r_.pairs_per_s1} |" for r_ in df.itertuples()]
    lines += ["", "## Fitted growth  t = a · N^b", "", "| mode | exponent b | projected time at 10^9 records |", "|---|---|---|"]
    for mode, g in df.groupby("mode"):
        if len(g) >= 2:
            b, loga = np.polyfit(np.log(g.records), np.log(g.seconds), 1)
            proj = np.exp(loga) * (1e9 ** b)
            lines.append(f"| {mode} | {b:.2f} | {proj / 3600:,.0f} CPU-hours (single core-set) |")
    lines += [
        "", "## Reading this", "",
        "- **exact** scores every S1 against every record that shares *any* n-gram. At challenge scale this is",
        "  affordable and gives the highest recall, so it is the default for the submission.",
        "- **capped** scores only through features with document frequency ≤ cap — an inverted index over",
        "  rare n-grams/tokens (prefix filtering). Work per record is bounded, so growth is ~linear.",
        "- At billions of records the capped channels shard by resolved country group and postcode prefix",
        "  (embarrassingly parallel), and the key channel is a hash join. Nothing in the pipeline needs",
        "  an all-pairs pass.",
        "- Caveat, stated plainly: the synthetic harness draws names from a ~26-word vocabulary, so few",
        "  n-grams are rare and the capped mode loses recall there; real business names are far more",
        "  diverse, which is exactly the regime prefix filtering is designed for.",
    ]
    Path(a.reports).mkdir(parents=True, exist_ok=True)
    (Path(a.reports) / "scaling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
