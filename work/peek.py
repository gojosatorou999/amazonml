import polars as pl
s1 = pl.read_parquet("work/train_s1.parquet")
r = pl.concat([pl.read_parquet("work/train_s2.parquet"), pl.read_parquet("work/train_s3.parquet")])
gt = pl.read_parquet("work/train_gt.parquet")
ex = gt.filter(pl.col("matched_entity_ids")!="").sample(12, seed=3)
pl.Config.set_fmt_str_lengths(200); pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)
for sid, m in ex.iter_rows():
    a = s1.filter(pl.col("entity_id")==sid).row(0)
    print("\n###", a)
    for mid in m.split(","):
        print("   ", r.filter(pl.col("entity_id")==mid).row(0))
print(s1["country"].value_counts(), r["country"].value_counts())
