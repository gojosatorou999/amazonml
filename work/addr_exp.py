"""Rare-token address keys: each record keys on pairs among its m rarest address words (df>=2),
and on its house numbers x those rare words. Measures recall per family against train truth."""
import sys, time, math, gc
import polars as pl
M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
C1, C2 = int(sys.argv[2]) if len(sys.argv) > 2 else 40, int(sys.argv[3]) if len(sys.argv) > 3 else 200
t0 = time.time()
rec = pl.scan_parquet("work/train_rec.parquet").select(
    "idx", (pl.col("src") == 1).alias("s1"), pl.col("country").hash(seed=1).alias("c"),
    pl.col("at").list.unique().list.eval(pl.element().hash(seed=3)).alias("w"),
    pl.col("an").list.head(3).list.unique().list.eval(pl.element().hash(seed=5)).alias("n"),
).collect()
print("load", rec.height, f"{rec.estimated_size()/1e9:.2f}GB", f"{time.time()-t0:.0f}s", flush=True)
tok = rec.select("idx", "c", "w").explode("w").drop_nulls("w")
df = tok.group_by("c", "w").len("df")
tok = tok.join(df, on=["c", "w"]).filter(pl.col("df") >= 2)
# m rarest per record (ties broken by token hash so both sides pick consistently)
rare = (tok.sort(["idx", "df", "w"]).group_by("idx", maintain_order=True).head(M)
        .group_by("idx").agg(pl.col("w"), pl.col("c").first()))
del tok, df; gc.collect()
print("rare", rare.height, f"{time.time()-t0:.0f}s", flush=True)
rare = rare.join(rec.select("idx", "s1", "n"), on="idx")
a = rare.select("idx", "s1", "c", pl.col("w").alias("a")).explode("a")
b = rare.select("idx", pl.col("w").alias("b")).explode("b")
keys = {
    "ar": a.join(b, on="idx").filter(pl.col("a") < pl.col("b"))
           .select(pl.struct(pl.lit(1), "c", "a", "b").hash().alias("key"), "idx", "s1"),
    "hr": a.join(rare.select("idx", pl.col("n")).explode("n").drop_nulls("n"), on="idx")
           .select(pl.struct(pl.lit(2), "c", "a", "n").hash().alias("key"), "idx", "s1"),
}
del a, b; gc.collect()
truth = pl.read_parquet("work/train_truth.parquet")
n_r = rec.filter(~pl.col("s1")).height
for fam, k in keys.items():
    t = time.time()
    k = k.unique()
    g = k.group_by("key").agg(pl.col("s1").sum().alias("n1"), (~pl.col("s1")).sum().alias("n2"))
    g = g.filter((pl.col("n1") >= 1) & (pl.col("n2") >= 1) & (pl.col("n1") <= C1) & (pl.col("n2") <= C2))
    g = g.with_columns((math.log(n_r) - pl.col("n2").cast(pl.Float64).log()).cast(pl.Float32).alias("w"))
    k = k.join(g.select("key", "w"), on="key")
    pairs = (k.filter(pl.col("s1")).select("key", pl.col("idx").alias("i"))
             .join(k.filter(~pl.col("s1")).select("key", pl.col("idx").alias("j"), "w"), on="key")
             .group_by("i", "j").agg(pl.col("w").max()))
    pairs.write_parquet(f"work/blk_train_{fam}.parquet")
    r = truth.join(pairs, on=["i", "j"], how="semi").height / truth.height
    print(f"{fam}: keys={k.height:,} pairs={pairs.height:,} per_s1={pairs.height/2206821:.1f} recall={r:.4f} ({time.time()-t:.0f}s)", flush=True)
    del k, g, pairs; gc.collect()
