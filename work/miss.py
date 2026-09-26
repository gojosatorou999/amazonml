import polars as pl
pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(260)
truth = pl.scan_parquet("work/train_truth.parquet")
hit = pl.concat([pl.scan_parquet(f"work/blk_train_{f}.parquet").select("i","j") for f in ["nt","np","ns","ar","hr"]]).unique()
miss = truth.join(hit, on=["i","j"], how="anti").collect()
print("missed", miss.height)
rec = pl.scan_parquet("work/train_rec.parquet").select("idx","business_name","business_address","country","nt","at","an")
m = miss.sample(100000, seed=1).lazy()
a = rec.rename({c: c+"1" for c in ["idx","business_name","business_address","country","nt","at","an"]})
b = rec.rename({c: c+"2" for c in ["idx","business_name","business_address","country","nt","at","an"]})
p = m.join(a, left_on="i", right_on="idx1").join(b, left_on="j", right_on="idx2").collect()
p = p.with_columns(
    pl.col("nt1").list.set_intersection("nt2").list.len().alias("nn"),
    pl.col("at1").list.set_intersection("at2").list.len().alias("na"),
    pl.col("an1").list.set_intersection("an2").list.len().alias("nnum"),
    pl.col("nt1").list.len().alias("l1"),
)
print(p.group_by("country1").len())
print("nn", p["nn"].value_counts().sort("nn").head(6))
print("na", p["na"].value_counts().sort("na").head(8))
print("nnum", p["nnum"].value_counts().sort("nnum").head(5))
print("l1", p["l1"].value_counts().sort("l1").head(8))
print(p.sample(30, seed=2).select("business_name1","business_name2","business_address1","business_address2"))
