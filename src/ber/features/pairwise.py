"""Pairwise comparison features for (Source-1 entity, S2/S3 record) pairs.

Families
--------
name      edit / token / char similarity on the full expanded name and on the
          core name (legal tail stripped); Soft-TFIDF; rare-token disagreement;
          acronym and concatenation matches; legal-tail agreement
address   slot-wise: postcode, house number, all-numbers; Soft-TFIDF over
          address tokens, street slot and locality tail; landmark flags
country   same resolved country group / same raw label
density   how common the core name is in each source (chains, franchises)
retrieval per-channel ranks and similarities from blocking (RRF inputs)

Soft-TFIDF (Cohen, Ravikumar & Fienberg, 2003) is the workhorse: tokens are
IDF-weighted, so 'Pvt Ltd Restaurant' agreement is near-worthless while a rare
token agreement is near-proof, and tokens are matched *softly* -- equal,
abbreviation (prefix / ordered subsequence: 'bd' ~ 'boulevard'), or
Jaro-Winkler >= 0.88 (typos).  The abbreviation test needs no dictionary,
which is what carries the unseen-country path.

Vectorised string metrics use rapidfuzz.process.cpdist (C++, multi-threaded);
Soft-TFIDF uses a cached token-pair similarity so its Python loop stays cheap.
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from ..normalize.abbrev import is_abbrev

CHANNELS = ["name_char", "full_char", "word", "name_char_rev", "full_char_rev", "word_rev", "keys"]
SOFT_THR = 0.88


@lru_cache(maxsize=2_000_000)
def tok_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if a.isdigit() or b.isdigit():
        return 0.0
    if (len(a) >= 2 and is_abbrev(a, b)) or (len(b) >= 2 and is_abbrev(b, a)):
        return 0.9
    return JaroWinkler.similarity(a, b)


def idf_table(docs) -> dict[str, float]:
    df = Counter()
    n = 0
    for toks in docs:
        df.update(set(toks))
        n += 1
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def soft_tfidf(a, b, idf, default_idf):
    """Returns (score, max IDF of an a-token with no soft match, same for b)."""
    if not a or not b:
        return 0.0, 0.0, 0.0
    sa, sb = set(a), set(b)
    wa = {t: idf.get(t, default_idf) for t in sa}
    wb = {t: idf.get(t, default_idf) for t in sb}
    na = math.sqrt(sum(w * w for w in wa.values()))
    nb = math.sqrt(sum(w * w for w in wb.values()))
    s = 0.0
    miss_a = 0.0
    matched_b = set()
    for t, w in wa.items():
        if t in wb:
            s += w * wb[t]
            matched_b.add(t)
            continue
        best, bu = 0.0, None
        for u in sb:
            v = tok_sim(t, u)
            if v > best:
                best, bu = v, u
        if best >= SOFT_THR:
            s += w * wb[bu] * best
            matched_b.add(bu)
        else:
            miss_a = max(miss_a, w)
    miss_b = max((w for u, w in wb.items() if u not in matched_b), default=0.0)
    return s / (na * nb), miss_a, miss_b


def legal_match(a: tuple, b: tuple) -> float:
    """Legal-form agreement, abbreviation-aware without a dictionary.

    -1 if either side has no legal tail; 1 for equal sets; 0.9 when one side is
    a single token abbreviating the other's concatenation ('llc' ~ 'limited
    liability company', 'sas' ~ 'societe par actions simplifiee'); else Jaccard.
    """
    if not a or not b:
        return -1.0
    x, y = set(a), set(b)
    if x == y:
        return 1.0
    for short, long in ((a, b), (b, a)):
        if len(short) == 1 and len(long) > 1 and is_abbrev(short[0], "".join(long)):
            return 0.9
    return len(x & y) / len(x | y)


def _state(a: str, b: str) -> int:
    """0 both missing, 1 one missing, 2 equal, 3 conflict."""
    if not a and not b:
        return 0
    if not a or not b:
        return 1
    return 2 if a == b else 3


def build_pair_features(pool: pd.DataFrame, s1: pd.DataFrame, r: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """`pool` has i, j (row positions) plus retrieval columns; returns feature frame."""
    A = s1.iloc[pool.i.values].reset_index(drop=True)
    B = r.iloc[pool.j.values].reset_index(drop=True)
    n = len(pool)
    F: dict[str, np.ndarray] = {}

    def cp(scorer, xa, xb):
        return process.cpdist(xa, xb, scorer=scorer, workers=-1).astype(np.float32) / (
            100.0 if scorer is not JaroWinkler.similarity else 1.0
        )

    na, nb = A.name_str.tolist(), B.name_str.tolist()
    ca, cb = A.core_str.tolist(), B.core_str.tolist()
    aa, ab = A.addr_str.tolist(), B.addr_str.tolist()
    F["n_ratio"] = cp(fuzz.ratio, na, nb)
    F["n_tsort"] = cp(fuzz.token_sort_ratio, na, nb)
    F["n_tset"] = cp(fuzz.token_set_ratio, na, nb)
    F["n_partial"] = cp(fuzz.partial_ratio, na, nb)
    F["c_ratio"] = cp(fuzz.ratio, ca, cb)
    F["c_tset"] = cp(fuzz.token_set_ratio, ca, cb)
    F["c_jw"] = cp(JaroWinkler.similarity, ca, cb)
    F["c_cat_ratio"] = cp(fuzz.ratio, A.core_cat.tolist(), B.core_cat.tolist())
    F["id_ratio"] = cp(fuzz.ratio, A.ident_str.tolist(), B.ident_str.tolist())
    F["id_tset"] = cp(fuzz.token_set_ratio, A.ident_str.tolist(), B.ident_str.tolist())
    F["a_ratio"] = cp(fuzz.ratio, aa, ab)
    F["a_tset"] = cp(fuzz.token_set_ratio, aa, ab)
    F["a_tsort"] = cp(fuzz.token_sort_ratio, aa, ab)

    # IDF over this dataset's records (transductive; provided data only)
    name_idf = idf_table(list(s1.name_toks) + list(r.name_toks))
    addr_idf = idf_table(list(s1.addr_toks) + list(r.addr_toks))
    dn = max(name_idf.values(), default=1.0)
    da = max(addr_idf.values(), default=1.0)

    n_soft = np.zeros(n, np.float32); n_ma = np.zeros(n, np.float32); n_mb = np.zeros(n, np.float32)
    c_soft = np.zeros(n, np.float32)
    id_soft = np.zeros(n, np.float32); id_ma = np.zeros(n, np.float32); id_mb = np.zeros(n, np.float32)
    a_soft = np.zeros(n, np.float32); a_ma = np.zeros(n, np.float32); a_mb = np.zeros(n, np.float32)
    st_soft = np.zeros(n, np.float32); tl_soft = np.zeros(n, np.float32)
    first_eq = np.zeros(n, np.int8); acr_hit = np.zeros(n, np.int8); tail_j = np.full(n, -1, np.float32)
    pc = np.zeros(n, np.int8); pc3 = np.zeros(n, np.int8); hs = np.zeros(n, np.int8); hs_part = np.zeros(n, np.int8)
    num_j = np.zeros(n, np.float32); num_conf = np.zeros(n, np.int8)
    cg = np.zeros(n, np.int8); cl = np.zeros(n, np.int8)

    cols = ["name_toks", "core", "ident", "legal_tail", "acr", "addr_toks", "street", "addr_tail",
            "postcode", "house", "numbers", "country"]
    for k, (ra, rb) in enumerate(zip(A[cols].itertuples(index=False), B[cols].itertuples(index=False))):
        n_soft[k], n_ma[k], n_mb[k] = soft_tfidf(ra.name_toks, rb.name_toks, name_idf, dn)
        c_soft[k] = soft_tfidf(ra.core, rb.core, name_idf, dn)[0]
        id_soft[k], id_ma[k], id_mb[k] = soft_tfidf(ra.ident, rb.ident, name_idf, dn)
        a_soft[k], a_ma[k], a_mb[k] = soft_tfidf(ra.addr_toks, rb.addr_toks, addr_idf, da)
        st_soft[k] = soft_tfidf(ra.street, rb.street, addr_idf, da)[0]
        tl_soft[k] = soft_tfidf(ra.addr_tail, rb.addr_tail, addr_idf, da)[0]
        if ra.core and rb.core:
            first_eq[k] = tok_sim(ra.core[0], rb.core[0]) >= SOFT_THR
            acr_hit[k] = (len(ra.acr) >= 2 and ra.acr in rb.core) or (len(rb.acr) >= 2 and rb.acr in ra.core)
        tail_j[k] = legal_match(ra.legal_tail, rb.legal_tail)
        pc[k] = _state(ra.postcode, rb.postcode)
        pc3[k] = bool(ra.postcode) and bool(rb.postcode) and ra.postcode[:3] == rb.postcode[:3]
        hs[k] = _state(ra.house, rb.house)
        if hs[k] == 3:
            hs_part[k] = bool(set(ra.house.replace("-", "/").split("/")) & set(rb.house.replace("-", "/").split("/")))
        if ra.numbers and rb.numbers:
            inter = len(ra.numbers & rb.numbers)
            num_j[k] = inter / len(ra.numbers | rb.numbers)
            num_conf[k] = inter == 0
        if ra.country and rb.country:
            cl[k] = ra.country == rb.country
            cg[k] = 1 + (groups.get(ra.country, ra.country) == groups.get(rb.country, rb.country))
        # cg: 0 unknown, 1 different group, 2 same group

    F.update(
        n_soft=n_soft, n_miss_a=n_ma / dn, n_miss_b=n_mb / dn, c_soft=c_soft,
        id_soft=id_soft, id_miss_a=id_ma / dn, id_miss_b=id_mb / dn,
        a_soft=a_soft, a_miss_a=a_ma / da, a_miss_b=a_mb / da, st_soft=st_soft, tl_soft=tl_soft,
        first_eq=first_eq, acr_hit=acr_hit, tail_jacc=tail_j,
        pc_state=pc, pc3=pc3, house_state=hs, house_part=hs_part, num_jacc=num_j, num_conflict=num_conf,
        country_grp=cg, country_label=cl,
        c_equal=(A.core_str.values == B.core_str.values).astype(np.int8),
        c_len_a=A.core.map(len).values.astype(np.int8), c_len_b=B.core.map(len).values.astype(np.int8),
        a_len_a=A.addr_toks.map(len).values.astype(np.int8), a_len_b=B.addr_toks.map(len).values.astype(np.int8),
        lm_a=A.landmark.values.astype(np.int8), lm_b=B.landmark.values.astype(np.int8),
        src=B.src.values.astype(np.int8),
    )
    F["name_x_addr"] = F["c_soft"] * F["a_soft"]
    F["ident_x_addr"] = F["id_soft"] * F["a_soft"]

    # density: a core name shared by many records is weak evidence (chains)
    f1 = s1.core_str.map(s1.core_str.value_counts())
    fr = r.core_str.map(r.core_str.value_counts())
    F["dens_s1"] = np.log1p(f1.values[pool.i.values]).astype(np.float32)
    F["dens_r"] = np.log1p(fr.values[pool.j.values]).astype(np.float32)

    for ch in CHANNELS:
        F[f"{ch}_rank"] = pool.get(f"{ch}_rank", pd.Series(0, index=pool.index)).values.astype(np.int16)
        F[f"{ch}_sim"] = pool.get(f"{ch}_sim", pd.Series(0.0, index=pool.index)).values.astype(np.float32)
    F["rrf"] = pool.rrf.values.astype(np.float32)
    F["n_channels"] = pool.n_channels.values.astype(np.int8)
    F["pool_rank"] = pool.pool_rank.values.astype(np.int16)
    return pd.DataFrame(F)
