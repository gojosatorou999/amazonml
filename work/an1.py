import polars as pl, sys, time
sys.path.insert(0, "src")
from er.norm import fold, tokens, LEGAL, ADDR_STOP
t=time.time()
def prep(path):
    return (pl.scan_parquet(path).select(
        pl.col("entity_id").alias("id"),
        tokens(fold(pl.col("business_name"))).list.eval(pl.element().filter(~pl.element().is_in(list(LEGAL)))).alias("nt"),
        tokens(fold(pl.col("business_address"))).list.eval(pl.element().filter(~pl.element().is_in(list(ADDR_STOP)))).alias("at"),
    ))
s1 = prep("work/train_s1.parquet").collect()
r = pl.concat([prep("work/train_s2.parquet"), prep("work/train_s3.parquet")]).collect()
print("prep", time.time()-t, flush=True)
gt = pl.read_parquet("work/train_gt.parquet").filter(pl.col("matched_entity_ids")!="").select(
    pl.col("source1_entity_id").alias("a"), pl.col("matched_entity_ids").str.split(",").alias("b")).explode("b")
p = gt.join(s1.rename({"id":"a","nt":"nt1","at":"at1"}), on="a").join(r.rename({"id":"b","nt":"nt2","at":"at2"}), on="b")
p = p.with_columns(
    pl.col("nt1").list.set_intersection("nt2").list.len().alias("nn"),
    pl.col("at1").list.set_intersection("at2").list.len().alias("na"),
    pl.col("nt2").list.len().alias("len2"),
    pl.col("at2").list.len().alias("alen2"),
)
print(len(p))
print("name overlap dist", p["nn"].value_counts().sort("nn").head(8))
print("addr overlap dist", p["na"].value_counts().sort("na").head(8))
print("no name overlap & addr overlap<2:", p.filter((pl.col("nn")==0)&(pl.col("na")<2)).height)
print("no name overlap & empty addr:", p.filter((pl.col("nn")==0)&(pl.col("alen2")==0)).height)
pl.Config.set_fmt_str_lengths(120); pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
print(p.filter((pl.col("nn")==0)&(pl.col("na")<2)).sample(25, seed=1).select("nt1","nt2","at1","at2"))
p.select("a","b","nn","na").write_parquet("work/an1_pairs.parquet")
