import sys, time, gc, math
sys.path.insert(0, "src")
import polars as pl
from er.blocking import family_keys, CAPS, WORK
fams = sys.argv[1].split(",")
rec = pl.scan_parquet("work/train_rec.parquet")
truth = pl.read_parquet("work/train_truth.parquet").with_columns(pl.lit(True).alias("y"))
n_r = rec.filter(pl.col("src") != 1).select(pl.len()).collect().item()
for fam in fams:
    t = time.time()
    c1, c2, fw = CAPS[fam]
    k = family_keys(rec, fam).collect()
    print(fam, "keys", f"{k.height:,}", f"{time.time()-t:.0f}s", flush=True)
    df = k.group_by("key").agg(pl.col("s1").sum().alias("n1"), (~pl.col("s1")).sum().alias("n2"))
    for cc1, cc2 in [(c1, c2)]:
        d = df.filter((pl.col("n1") >= 1) & (pl.col("n2") >= 1) & (pl.col("n1") <= cc1) & (pl.col("n2") <= cc2))
        print("   est pairs", f"{(d['n1'].cast(pl.Int64)*d['n2']).sum():,}")
    d = d.with_columns((fw * (math.log(n_r) - pl.col("n2").cast(pl.Float64).log())).cast(pl.Float32).alias("w"))
    k = k.join(d.select("key", "w"), on="key")
    s1k = k.filter(pl.col("s1")).select("key", pl.col("idx").alias("i"))
    rk = k.filter(~pl.col("s1")).select("key", pl.col("idx").alias("j"), "w")
    del k, df, d; gc.collect()
    pairs = s1k.join(rk, on="key").group_by("i", "j").agg(pl.col("w").max())
    pairs.write_parquet(WORK / f"blk_train_{fam}.parquet")
    rec_ = truth.join(pairs, on=["i","j"], how="semi").height / truth.height
    print(f"   pairs={pairs.height:,} recall={rec_:.4f} per_s1={pairs.height/2206821:.1f} ({time.time()-t:.0f}s)", flush=True)
    del pairs, s1k, rk; gc.collect()
