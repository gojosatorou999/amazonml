import sys, time, polars as pl
fams = sys.argv[1].split(","); NC = 8
truth = pl.read_parquet("work/train_truth.parquet").with_columns(pl.lit(1, pl.UInt8).alias("y"))
Ks = [5, 10, 20, 30, 50, 100, 10**9]
tot = {k: 0 for k in Ks}; npairs = {k: 0 for k in Ks}
t = time.time()
for c in range(NC):
    lf = pl.concat([pl.scan_parquet(f"work/blk_train_{f}.parquet").filter(pl.col("i") % NC == c).select("i", "j", "w")
                    for f in fams])
    g = lf.group_by("i", "j").agg(pl.col("w").sum().alias("s"), pl.len().alias("nf")).collect()
    g = g.with_columns(pl.col("s").rank("ordinal", descending=True).over("i").alias("r"))
    g = g.join(truth.filter(pl.col("i") % NC == c), on=["i", "j"], how="left")
    for k in Ks:
        tot[k] += g.filter((pl.col("r") <= k) & pl.col("y").is_not_null()).height
        npairs[k] += g.filter(pl.col("r") <= k).height
    print(c, f"{time.time()-t:.0f}s", flush=True)
for k in Ks:
    print(f"top{k}: recall={tot[k]/truth.height:.4f} per_s1={npairs[k]/2206821:.1f}")
