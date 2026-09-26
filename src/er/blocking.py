"""Stage 2: key-based blocking that scales to millions of records.

Every record emits hashed keys from several families; an S1 record and an S2/S3 record become a
candidate pair when they share a key whose block is small enough to be informative. A pair's
retrieval score is the IDF-weighted count of shared keys, summed over families. We keep the
top-K records per S1 entity *and* the top-k S1 entities per record (reverse direction: a record
whose S1 owner has a crowded, generic name is still retrieved from its own side).

Families
  nt  identity name token                     typo-free rare word ("buildwell")
  np  unordered pair of identity name tokens  generic words that are rare together ("family practice")
  ns  squashed name spans                     domains / concatenations ("simonartificial.com")
  sk  pair of phonetic name skeletons         Indic-script names vs their romanised S1 spelling
  ah  house number x address word             "7900" x "princess"
  aw  pair of address words                   numberless addresses ("siddharth nagar" x "goregaon")
"""
from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import polars as pl

WORK = Path("work")

# (max S1 records, max S2/S3 records) sharing one key, and a family weight multiplier.
CAPS = {
    "nt": (40, 200, 1.0),
    "np": (40, 200, 1.0),
    "ns": (40, 200, 1.5),
    "sk": (40, 200, 0.8),
    "ah": (40, 200, 1.0),
    "aw": (40, 200, 0.7),
}


def _h(fam: str, *parts: pl.Expr) -> pl.Expr:
    return pl.concat_str([pl.lit(fam), pl.col("country"), *parts], separator="\x1f").hash(seed=7)


def family_keys(rec: pl.LazyFrame, fam: str) -> pl.LazyFrame:
    """-> (key: u64, idx: u32, s1: bool), one row per (record, key), deduplicated."""
    base = rec.select("idx", "country", (pl.col("src") == 1).alias("s1"), "nt", "nsq", "nsk", "at", "an", "n_indic")
    if fam == "nt":
        out = (base.select("idx", "country", "s1", pl.col("nt").list.unique().alias("t")).explode("t")
               .filter(pl.col("t").str.len_chars() >= 2).select(_h("nt", pl.col("t")).alias("key"), "idx", "s1"))
    elif fam in ("np", "sk", "aw"):
        col, cap_len = {"np": ("nt", 8), "sk": ("nsk", 8), "aw": ("at", 8)}[fam]
        src = base
        if fam == "sk":
            # S1 names are romanised; only S2/S3 records written in an Indic script need the phonetic bridge,
            # but S1 must emit the keys so the two sides can meet.
            src = base.filter(pl.col("s1") | pl.col("n_indic"))
        t = (src.select("idx", "country", "s1", pl.col(col).list.head(cap_len).list.unique().alias("t"))
             .filter(pl.col("t").list.len() >= 2))
        a = t.explode("t").rename({"t": "a"})
        b = t.select("idx", pl.col("t").alias("b")).explode("b")
        out = (a.join(b, on="idx").filter(pl.col("a") < pl.col("b"))
               .select(_h(fam, pl.col("a"), pl.col("b")).alias("key"), "idx", "s1"))
    elif fam == "ns":
        spans = pl.concat_list(
            pl.col("nsq"),
            pl.col("nt").list.head(2).list.join(""),
            pl.col("nt").list.head(3).list.join(""),
            pl.col("nt").list.tail(2).list.join(""),
        )
        out = (base.select("idx", "country", "s1", spans.list.unique().alias("t")).explode("t")
               .filter(pl.col("t").str.len_chars() >= 6).select(_h("ns", pl.col("t")).alias("key"), "idx", "s1"))
    elif fam == "ah":
        t = base.select("idx", "country", "s1", pl.col("an").list.head(4).list.unique().alias("n"),
                        pl.col("at").list.head(8).list.unique().alias("w"))
        out = (t.explode("n").explode("w").filter(pl.col("n").is_not_null() & pl.col("w").is_not_null())
               .select(_h("ah", pl.col("n"), pl.col("w")).alias("key"), "idx", "s1"))
    else:
        raise ValueError(fam)
    return out.unique()


def block(split: str, families=tuple(CAPS), topk: int = 30, rev_k: int = 3, n_chunks: int = 8) -> pl.DataFrame:
    rec = pl.scan_parquet(WORK / f"{split}_rec.parquet")
    n_r = rec.filter(pl.col("src") != 1).select(pl.len()).collect().item()
    pair_files = []
    for fam in families:
        t = time.time()
        c1, c2, fw = CAPS[fam]
        k = family_keys(rec, fam).collect()
        df = k.group_by("key").agg(pl.col("s1").sum().alias("n1"), (~pl.col("s1")).sum().alias("n2"))
        df = df.filter((pl.col("n1") >= 1) & (pl.col("n2") >= 1) & (pl.col("n1") <= c1) & (pl.col("n2") <= c2))
        df = df.with_columns((fw * (math.log(n_r) - pl.col("n2").cast(pl.Float64).log())).cast(pl.Float32).alias("w"))
        k = k.join(df.select("key", "w"), on="key")
        s1k = k.filter(pl.col("s1")).select("key", pl.col("idx").alias("i"))
        rk = k.filter(~pl.col("s1")).select("key", pl.col("idx").alias("j"), "w")
        del k, df
        pairs = s1k.join(rk, on="key").select("i", "j", "w")
        pairs = pairs.group_by("i", "j").agg(pl.col("w").max())  # one key's worth per family per pair
        f = WORK / f"blk_{split}_{fam}.parquet"
        pairs.write_parquet(f)
        pair_files.append(f)
        print(f"  {fam}: pairs={pairs.height:,} ({time.time() - t:.0f}s)", flush=True)
        del s1k, rk, pairs
        gc.collect()

    # Fuse families: score = sum of per-family weights; per-family indicator bits kept as features.
    t = time.time()
    parts = []
    for c in range(n_chunks):
        lf = pl.concat([
            pl.scan_parquet(f).filter(pl.col("i") % n_chunks == c).with_columns(pl.lit(fam).alias("fam"))
            for fam, f in zip(families, pair_files)
        ])
        g = lf.group_by("i", "j").agg(
            pl.col("w").sum().alias("score"),
            *[pl.col("w").filter(pl.col("fam") == fam).max().fill_null(0).alias(f"b_{fam}") for fam in families],
        ).collect()
        parts.append(g)
    pool = pl.concat(parts)
    del parts
    gc.collect()
    pool = pool.with_columns(
        pl.col("score").rank("ordinal", descending=True).over("i").alias("rank_i"),
        pl.col("score").rank("ordinal", descending=True).over("j").alias("rank_j"),
    )
    cand = pool.filter((pl.col("rank_i") <= topk) | (pl.col("rank_j") <= rev_k))
    print(f"  fuse: pool={pool.height:,} cand={cand.height:,} ({time.time() - t:.0f}s)", flush=True)
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--revk", type=int, default=3)
    a = ap.parse_args()
    cand = block(a.split, topk=a.topk, rev_k=a.revk)
    cand.write_parquet(WORK / f"{a.split}_cand.parquet")


if __name__ == "__main__":
    main()
