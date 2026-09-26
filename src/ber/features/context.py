"""Context features: a pair's score *relative to its competitors*.

Pairwise features judge (s1, r) in isolation.  Identity is comparative: a 0.8
name similarity is decisive when it is the only candidate above 0.3 and
meaningless when forty chain branches all score 0.8.  Given stage-1 scores p
over the whole blocking pool we add, per pair:

  forward competition   rank of p among s1's candidates, gap to s1's best,
                        margin over s1's runner-up, mass of s1's other candidates
  reverse competition   the same from the record's side: how many *other*
                        Source-1 entities want this record, and how badly.
                        Source 1 is deduplicated, so a record has at most one
                        true owner -- being a record's second choice is strong
                        evidence against.
  sibling support       max over s1's other confident candidates r' of
                        p(s1, r') * sim(r, r').  Cross-source records of one
                        business corroborate each other: if S2-a confidently
                        matches s1 and S3-b is a near-copy of S2-a, S3-b gains
                        evidence.  One round of propagation over the candidate
                        graph -- collective resolution rather than independent
                        pair classification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


def build_context_features(pool: pd.DataFrame, p: np.ndarray, r: pd.DataFrame, support_floor: float = 0.3) -> pd.DataFrame:
    df = pd.DataFrame({"i": pool.i.values, "j": pool.j.values, "p": p.astype(np.float32)})
    gi = df.groupby("i")["p"]
    gj = df.groupby("j")["p"]
    out = pd.DataFrame(index=df.index)
    out["ctx_p"] = df.p
    out["ctx_rank_i"] = gi.rank(ascending=False, method="min").astype(np.float32)
    out["ctx_gap_i"] = (gi.transform("max") - df.p).astype(np.float32)
    out["ctx_sum_i"] = (gi.transform("sum") - df.p).astype(np.float32)
    out["ctx_nhi_i"] = df.assign(h=df.p >= 0.5).groupby("i")["h"].transform("sum").astype(np.float32)
    # margin over the best *other* candidate of the same s1
    top2 = df.sort_values(["i", "p"], ascending=[True, False]).groupby("i")["p"].apply(lambda s: s.values[:2])
    first = top2.map(lambda a: a[0]); second = top2.map(lambda a: a[1] if len(a) > 1 else 0.0)
    best_other_i = np.where(df.p.values >= df.i.map(first).values, df.i.map(second).values, df.i.map(first).values)
    out["ctx_margin_i"] = (df.p.values - best_other_i).astype(np.float32)

    out["ctx_rank_j"] = gj.rank(ascending=False, method="min").astype(np.float32)
    out["ctx_gap_j"] = (gj.transform("max") - df.p).astype(np.float32)
    out["ctx_sum_j"] = (gj.transform("sum") - df.p).astype(np.float32)
    out["ctx_cnt_j"] = gj.transform("size").astype(np.float32)
    odds = df.p / (1.0 - df.p.clip(upper=0.999))
    out["ctx_excl_j"] = (odds / (1.0 + df.assign(o=odds).groupby("j")["o"].transform("sum"))).astype(np.float32)

    # sibling support among confident candidates of the same s1
    conf = df[df.p >= support_floor]
    sup = np.zeros(len(df), np.float32)
    nsib = np.zeros(len(df), np.float32)
    if len(conf):
        sib = df[["i", "j"]].reset_index().merge(conf[["i", "j", "p"]].rename(columns={"j": "j2", "p": "p2"}), on="i")
        sib = sib[sib.j != sib.j2]
        if len(sib):
            name = r.core_str.values
            addr = r.addr_str.values
            sn = process.cpdist(name[sib.j.values], name[sib.j2.values], scorer=fuzz.token_set_ratio, workers=-1)
            sa = process.cpdist(addr[sib.j.values], addr[sib.j2.values], scorer=fuzz.token_set_ratio, workers=-1)
            sib["s"] = sib.p2.values * (0.5 * (sn + sa) / 100.0)
            agg = sib.groupby("index").agg(s=("s", "max"), c=("s", "size"))
            sup[agg.index.values] = agg.s.values
            nsib[agg.index.values] = agg.c.values
    out["ctx_support"] = sup
    out["ctx_nsib"] = nsib
    return out.reset_index(drop=True)
