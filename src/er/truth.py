"""Ground truth as (i, j) record-index pairs for the train split."""
from __future__ import annotations

from pathlib import Path

import polars as pl

WORK = Path("work")


def truth_pairs(split: str = "train") -> pl.DataFrame:
    ids = pl.scan_parquet(WORK / f"{split}_rec.parquet").select("entity_id", "idx")
    gt = (pl.scan_parquet(WORK / f"{split}_gt.parquet")
          .filter(pl.col("matched_entity_ids") != "")
          .select(pl.col("source1_entity_id").alias("a"), pl.col("matched_entity_ids").str.split(",").alias("b"))
          .explode("b"))
    return (gt.join(ids.rename({"entity_id": "a", "idx": "i"}), on="a")
            .join(ids.rename({"entity_id": "b", "idx": "j"}), on="b")
            .select("i", "j").collect())
