import polars as pl, sys
sys.path.insert(0, "src")
from er.norm import fold, tokens, LEGAL, ADDR_STOP
def prep(path):
    return (pl.scan_parquet(path).select(
        pl.col("entity_id").alias("id"), "business_name", "business_address", "country",
        tokens(fold(pl.col("business_name"))).list.eval(pl.element().filter(~pl.element().is_in(list(LEGAL)))).alias("nt"),
        tokens(fold(pl.col("business_address"))).list.eval(pl.element().filter(~pl.element().is_in(list(ADDR_STOP)))).alias("at"),
    ))
s1 = prep("work/train_s1.parquet").collect()
r = pl.concat([prep("work/train_s2.parquet"), prep("work/train_s3.parquet")]).collect()
gt = pl.read_parquet("work/train_gt.parquet").select(pl.col("matched_entity_ids").str.split(",").alias("b")).explode("b")
d = r.join(gt, left_on="id", right_on="b", how="anti")
print("distractors", d.height, "of", r.height)
print("true pairs w/ country mismatch check:")
pl.Config.set_fmt_str_lengths(100); pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
for row in d.filter(pl.col("country")=="US").sample(8, seed=5).iter_rows(named=True):
    sc = s1.with_columns((pl.col("nt").list.set_intersection(pl.lit(row["nt"])).list.len()*2 + pl.col("at").list.set_intersection(pl.lit(row["at"])).list.len()).alias("sc")).top_k(3, by="sc")
    print("\nDISTRACTOR:", row["business_name"], "|", row["business_address"])
    for x in sc.iter_rows(named=True): print("   S1:", x["sc"], x["business_name"], "|", x["business_address"])
for row in d.filter(pl.col("country")=="India").sample(5, seed=5).iter_rows(named=True):
    sc = s1.with_columns((pl.col("nt").list.set_intersection(pl.lit(row["nt"])).list.len()*2 + pl.col("at").list.set_intersection(pl.lit(row["at"])).list.len()).alias("sc")).top_k(3, by="sc")
    print("\nDISTRACTOR:", row["business_name"], "|", row["business_address"])
    for x in sc.iter_rows(named=True): print("   S1:", x["sc"], x["business_name"], "|", x["business_address"])
