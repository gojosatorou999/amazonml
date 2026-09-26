"""Recall-oriented candidate retrieval: several channels, both directions, fused.

Channels (each covers a failure mode of the others):

    name_char   char 2-4-gram TF-IDF on the core name      typos, spacing, 'MaplecrestSunstone'
    full_char   char 3-gram TF-IDF on core name + address  records whose name alone is generic
    word        word TF-IDF on name + address + numbers    rare-token agreement (IDF does the work)
    keys        exact structural keys, inverted index      postcode, house+street, name+house,
                                                           order-free name, acronym

Every TF-IDF channel runs **bidirectionally**: S1 -> R finds each S1 entity's
nearest records, and R -> S1 finds each record's nearest S1 entities.  The
reverse direction matters because a record's best S1 entity is not always in
that entity's own top-k (a generic S1 name has many near neighbours that crowd
the forward list); it costs one extra sparse product and rescues those pairs.

Fusion is Reciprocal Rank Fusion (Cormack et al., SIGIR 2009): score-scale free,
so channels with incomparable similarity scales combine without tuning.

Scaling: every product is a chunked sparse matmul with a per-row top-k, so
memory is O(chunk x |R|) and time is linear in |S1| for a fixed corpus.  At
billions of records the same channels shard naturally by country group and
postcode prefix (see reports/scaling.md).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

RRF_K = 60


def topk_sparse(Q: sparse.csr_matrix, D: sparse.csr_matrix, k: int, budget: float = 4e7):
    """Row-wise top-k cosine of L2-normalised sparse rows. Returns (idx, sim)."""
    nq, nd = Q.shape[0], D.shape[0]
    k = min(k, nd)
    DT = D.T.tocsc().astype(np.float32)
    chunk = max(1, int(budget // max(nd, 1)))
    # unused slots keep sim = 0 and are dropped downstream (`_channel_rows` keeps sim > 0)
    idx = np.zeros((nq, k), dtype=np.int32)
    sim = np.zeros((nq, k), dtype=np.float32)
    for s in range(0, nq, chunk):
        C = (Q[s : s + chunk].astype(np.float32) @ DT).tocsr()
        if C.nnz < 0.05 * C.shape[0] * nd:
            # sparse path (df-capped retrieval): top-k straight from each row's
            # nonzeros -- cost O(nnz), never O(chunk x |R|)
            for row in range(C.shape[0]):
                lo, hi = C.indptr[row], C.indptr[row + 1]
                if hi == lo:
                    continue
                d, v = C.indices[lo:hi], C.data[lo:hi]
                kk = min(k, hi - lo)
                top = np.argpartition(-v, kk - 1)[:kk] if hi - lo > kk else np.arange(hi - lo)
                top = top[np.argsort(-v[top], kind="stable")]
                idx[s + row, :kk] = d[top]
                sim[s + row, :kk] = v[top]
            continue
        block = C.toarray()
        part = np.argpartition(-block, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-vals, axis=1, kind="stable")
        idx[s : s + chunk] = np.take_along_axis(part, order, axis=1)
        sim[s : s + chunk] = np.take_along_axis(vals, order, axis=1)
    return idx, sim


def _tfidf(texts_q, texts_d, df_cap: int | None = None, **kw):
    """TF-IDF vectors, optionally restricted to features with df <= df_cap.

    The cap is what makes retrieval sub-quadratic.  The sparse product costs
    sum_f df_q(f) * df_d(f); frequent features ('ing', 'road') dominate that sum
    while carrying almost no identity signal.  Dropping them from the *product*
    (norms still come from the full vector, so scores remain partial cosines on
    a common scale) bounds the work per record by (#features x df_cap): linear
    in corpus size -- an inverted index over rare features, i.e. the classic
    prefix-filtering idea from set-similarity joins.
    """
    vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **kw)
    vec.fit(list(texts_q) + list(texts_d))
    return _cap(vec.transform(texts_q), vec.transform(texts_d), df_cap)


def _channel_rows(name, idx, sim, reverse: bool) -> pd.DataFrame:
    nq, k = idx.shape
    q = np.repeat(np.arange(nq, dtype=np.int32), k)
    d = idx.ravel()
    rank = np.tile(np.arange(1, k + 1, dtype=np.int16), nq)
    s = sim.ravel()
    keep = s > 0
    if reverse:  # rows were R -> S1; swap back to (s1, r)
        q, d = d, q
    return pd.DataFrame(
        {"i": q[keep], "j": d[keep], "ch": name + ("_rev" if reverse else ""), "rank": rank[keep], "sim": s[keep]}
    )


def record_keys(row) -> list[str]:
    """Deterministic blocking keys for one prepared record."""
    out = []
    if row.postcode:
        out.append("pc:" + row.postcode)
    if row.house and row.street:
        out.append("hs:" + row.house + "|" + row.street[0])
    if row.core:
        out.append("nm:" + " ".join(sorted(set(row.core))))
        out.append("cc:" + row.core_cat)
        if row.house:
            out.append("nh:" + row.core[0] + "|" + row.house)
    if len(row.acr) >= 2:
        out.append("ac:" + row.acr)
    return out


def key_index(r: pd.DataFrame) -> dict[str, list[int]]:
    inv: dict[str, list[int]] = defaultdict(list)
    for j, row in enumerate(r.itertuples(index=False)):
        for key in record_keys(row):
            inv[key].append(j)
    return inv


def key_channel(s1: pd.DataFrame, inv: dict[str, list[int]], k: int, max_block: int = 60, i_offset: int = 0) -> pd.DataFrame:
    """Deterministic keys through an inverted index; oversized blocks are skipped.

    `max_block` is the classic blocking guard: a key shared by hundreds of
    records ('postcode missing', a very common name) carries no information and
    would explode the candidate count, so it is dropped rather than truncated.
    """
    rows_i, rows_j, rows_s = [], [], []
    for i, row in enumerate(s1.itertuples(index=False)):
        score: dict[int, float] = defaultdict(float)
        for key in record_keys(row):
            block = inv.get(key)
            if not block or len(block) > max_block:
                continue
            w = 1.0 / math.log(2 + len(block))
            for j in block:
                score[j] += w
        if score:
            best = sorted(score.items(), key=lambda kv: -kv[1])[:k]
            for j, sc in best:
                rows_i.append(i + i_offset)
                rows_j.append(j)
                rows_s.append(sc)
    df = pd.DataFrame({"i": np.array(rows_i, dtype=np.int64), "j": np.array(rows_j, dtype=np.int64),
                       "sim": np.array(rows_s, dtype=np.float32)})
    df["rank"] = (df.groupby("i")["sim"].rank(ascending=False, method="first").astype(np.int16)
                  if len(df) else pd.Series([], dtype=np.int16))
    df["ch"] = "keys"
    return df


def _texts(df: pd.DataFrame) -> dict[str, list[str]]:
    nums = df.numbers.map(lambda s: " ".join(sorted(s)))
    return {
        "name_char": df.core_str.where(df.core_str != "", df.name_str).tolist(),
        "full_char": (df.core_str + " " + df.addr_str).tolist(),
        "word": (df.name_str + " " + df.addr_str + " " + nums).tolist(),
    }


CHANNEL_KW = {
    "name_char": dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2),
    "full_char": dict(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.5),
    "word": dict(analyzer="word", token_pattern=r"\S+", min_df=1),
}


def _cap(Q, D, df_cap):
    if not df_cap:
        return Q, D
    df = np.asarray((Q > 0).sum(axis=0)).ravel() + np.asarray((D > 0).sum(axis=0)).ravel()
    mask = sparse.diags((df <= df_cap).astype(np.float32))
    Q, D = (Q @ mask).tocsr(), (D @ mask).tocsr()
    Q.eliminate_zeros(); D.eliminate_zeros()
    return Q, D


class RetrievalIndex:
    """All retrieval state for one (S1, R) corpus, reusable for new queries.

    `long` holds every channel's (i, j, rank, sim) rows for the batch; `query`
    scores *new* Source-1 records against the same fitted vectorisers and R
    matrices.  Reverse-channel ranks for a query are exact: each record keeps
    its top-`kr` S1 similarities, and the query enters that list iff it beats
    the stored `kr`-th value.
    """

    def __init__(self, s1: pd.DataFrame, r: pd.DataFrame, k: int = 20, df_cap: int | None = None):
        self.k, self.kr, self.df_cap = k, max(3, k // 4), df_cap
        self.n1 = len(s1)
        tq, td = _texts(s1), _texts(r)
        self.vec, self.D, self.rev_sims, self.masks = {}, {}, {}, {}
        self.diag: dict = {}
        frames = []
        for name, kw in CHANNEL_KW.items():
            vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **kw).fit(tq[name] + td[name])
            Q0, D0 = vec.transform(tq[name]), vec.transform(td[name])
            Q, D = _cap(Q0, D0, df_cap)
            fi, fs = topk_sparse(Q, D, k)
            ri, rs = topk_sparse(D, Q, self.kr)
            frames.append(_channel_rows(name, fi, fs, reverse=False))
            frames.append(_channel_rows(name, ri, rs, reverse=True))
            self.vec[name], self.D[name], self.rev_sims[name] = vec, D.T.tocsc().astype(np.float32), rs
            if df_cap:
                df = np.asarray((Q0 > 0).sum(axis=0)).ravel() + np.asarray((D0 > 0).sum(axis=0)).ravel()
                self.masks[name] = sparse.diags((df <= df_cap).astype(np.float32))
            if name == "full_char":
                self.diag["best_fwd"] = (fi[:, 0], fs[:, 0])
        self.inv = key_index(r)
        frames.append(key_channel(s1, self.inv, k))
        self.long = pd.concat(frames, ignore_index=True)

    def query(self, s1_new: pd.DataFrame, pool: int = 30) -> pd.DataFrame:
        """Fused pool rows for new S1 records; their `i` values start at the batch size."""
        tq = _texts(s1_new)
        frames = []
        for name, vec in self.vec.items():
            Q = vec.transform(tq[name])
            if name in self.masks:
                Q = (Q @ self.masks[name]).tocsr()
            S = (Q.astype(np.float32) @ self.D[name]).toarray()             # (n_new, |R|)
            k = min(self.k, S.shape[1])
            fi = np.argsort(-S, axis=1, kind="stable")[:, :k].astype(np.int32)
            fs = np.take_along_axis(S, fi, axis=1)
            fr = _channel_rows(name, fi, fs, reverse=False)
            fr["i"] = fr["i"] + self.n1
            frames.append(fr)
            rs = self.rev_sims[name]                                          # (|R|, kr), sorted desc
            for q in range(S.shape[0]):
                sims = S[q]
                enter = np.flatnonzero((sims > rs[:, -1]) & (sims > 0))
                if len(enter):
                    rank = 1 + (rs[enter] > sims[enter, None]).sum(axis=1)
                    frames.append(pd.DataFrame({"i": self.n1 + q, "j": enter, "ch": name + "_rev",
                                                "rank": rank.astype(np.int16),
                                                "sim": sims[enter].astype(np.float32)}))
        frames.append(key_channel(s1_new, self.inv, self.k, i_offset=self.n1))
        return fuse(pd.concat(frames, ignore_index=True), pool)


def fuse(long: pd.DataFrame, pool: int) -> pd.DataFrame:
    """Reciprocal Rank Fusion of channel rows -> the per-S1 pool with wide rank/sim columns."""
    long = long.copy()
    long["rr"] = 1.0 / (RRF_K + long["rank"].astype(np.float32))
    fused = long.groupby(["i", "j"], sort=False).agg(rrf=("rr", "sum"), n_channels=("ch", "nunique")).reset_index()
    fused["pool_rank"] = fused.groupby("i")["rrf"].rank(ascending=False, method="first")
    # keep the fused top-`pool`, plus anything a reverse channel ranked 1-2:
    # "this record's best S1 is you" is too strong a signal to truncate away
    rev_strong = long[(long.ch.str.endswith("_rev")) & (long["rank"] <= 2)][["i", "j"]].drop_duplicates()
    rev_strong["rev_keep"] = True
    fused = fused.merge(rev_strong, on=["i", "j"], how="left")
    fused = fused[(fused.pool_rank <= pool) | fused.rev_keep.eq(True)].drop(columns="rev_keep")

    wide_rank = long.pivot_table(index=["i", "j"], columns="ch", values="rank", aggfunc="min")
    wide_sim = long.pivot_table(index=["i", "j"], columns="ch", values="sim", aggfunc="max")
    wide_rank.columns = [f"{c}_rank" for c in wide_rank.columns]
    wide_sim.columns = [f"{c}_sim" for c in wide_sim.columns]
    out = fused.merge(wide_rank.reset_index(), on=["i", "j"], how="left").merge(
        wide_sim.reset_index(), on=["i", "j"], how="left"
    )
    for c in out.columns:
        if c.endswith("_rank"):
            out[c] = out[c].fillna(0).astype(np.int16)  # 0 = channel did not retrieve it
        elif c.endswith("_sim"):
            out[c] = out[c].fillna(0).astype(np.float32)
    return out.sort_values(["i", "pool_rank"]).reset_index(drop=True)


def retrieve(
    s1: pd.DataFrame, r: pd.DataFrame, k: int = 20, pool: int = 30, df_cap: int | None = None,
    return_index: bool = False,
):
    """Return the fused candidate pool with per-channel rank/sim columns.

    Output columns: i, j (row positions into s1 / r), rrf, n_channels, and
    `<channel>_rank`, `<channel>_sim` for every channel (missing -> rank 0, sim 0).
    """
    idx = RetrievalIndex(s1, r, k=k, df_cap=df_cap)
    out = fuse(idx.long, pool)
    return (out, idx.diag, idx) if return_index else (out, idx.diag)
