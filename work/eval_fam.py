import sys, time, polars as pl
truth = pl.read_parquet("work/train_truth.parquet")
print("truth", truth.height, truth.columns, flush=True)
n1 = 2206821
got = None
for fam in sys.argv[1].split(","):
    t = time.time()
    p = pl.scan_parquet(f"work/blk_train_{fam}.parquet")
    n = p.select(pl.len()).collect().item()
    hit = truth.lazy().join(p.select("i", "j"), on=["i", "j"], how="semi").collect().with_columns(pl.lit(fam).alias("f"))
    print(f"{fam}: pairs={n:,} per_s1={n/n1:.1f} recall={hit.height/truth.height:.4f} ({time.time()-t:.0f}s)", flush=True)
    got = hit if got is None else pl.concat([got, hit])
u = got.select("i", "j").unique()
print("union recall", u.height / truth.height)
