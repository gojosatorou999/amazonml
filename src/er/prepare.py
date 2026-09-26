"""Stage 1: raw TSV/parquet -> one normalised record table per split (all three sources)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from .norm import ADDR_STOP, LEGAL, STREET, fold, tokens
from .translit import has_indic, skeleton, translit

RAW = Path("student_resource/dataset")
WORK = Path("work")


def load_raw(split: str, src: int) -> pl.LazyFrame:
    pq = WORK / f"{split}_s{src}.parquet"
    if pq.exists():
        return pl.scan_parquet(pq)
    return pl.scan_csv(RAW / split / f"{split}_source{src}.tsv", separator="\t", quote_char=None,
                       schema_overrides={c: pl.Utf8 for c in
                                         ["entity_id", "business_name", "business_address", "country"]})


def prepare(split: str) -> pl.DataFrame:
    frames = [load_raw(split, s).with_columns(pl.lit(s, pl.UInt8).alias("src")) for s in (1, 2, 3)]
    lf = pl.concat(frames).with_columns(
        pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""),
        pl.col("country").fill_null(""),
    )
    legal = list(LEGAL)
    lf = lf.with_columns(
        has_indic(pl.col("business_name")).alias("n_indic"),
        has_indic(pl.col("business_address")).alias("a_indic"),
        fold(pl.when(has_indic(pl.col("business_name"))).then(translit(pl.col("business_name")))
             .otherwise(pl.col("business_name"))).alias("nf"),
        fold(pl.when(has_indic(pl.col("business_address"))).then(translit(pl.col("business_address")))
             .otherwise(pl.col("business_address"))).alias("af0"),
    ).with_columns(
        tokens(pl.col("nf")).alias("ntall"),
        # street-type spellings canonicalised token by token
        tokens(pl.col("af0")).list.eval(pl.element().replace(STREET)).alias("atall"),
    ).with_columns(
        pl.col("ntall").list.eval(pl.element().filter(~pl.element().is_in(legal))).alias("nt"),
        pl.col("atall").list.join(" ").alias("af"),
        pl.col("atall").list.eval(pl.element().filter(
            ~pl.element().str.contains(r"\d") & ~pl.element().is_in(list(ADDR_STOP)) & (pl.element().str.len_chars() > 1)
        )).alias("at"),
        # house / plot numbers: digit runs with leading zeros stripped ("004013" -> "4013", "34th" -> "34")
        pl.col("af0").str.extract_all(r"\d+").list.eval(
            pl.element().str.strip_chars_start("0").replace("", "0")).alias("an"),
    ).with_columns(
        pl.col("nt").list.join("").alias("nsq"),
        skeleton(pl.col("nt").list.join(" ")).alias("nsk_s"),
    ).with_columns(
        tokens(pl.col("nsk_s")).alias("nsk"),
    ).drop("af0", "nsk_s")
    df = lf.collect()
    df = df.with_row_index("idx")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    t = time.time()
    df = prepare(a.split)
    WORK.mkdir(exist_ok=True)
    df.write_parquet(WORK / f"{a.split}_rec.parquet")
    print(a.split, df.shape, f"{time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
