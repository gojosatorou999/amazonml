import polars as pl, time, sys
base = "student_resource/dataset"
for split in ["train", "test"]:
    for s in [1, 2, 3]:
        t = time.time()
        df = pl.read_csv(f"{base}/{split}/{split}_source{s}.tsv", separator="\t", quote_char=None,
                         schema_overrides={c: pl.Utf8 for c in ["entity_id","business_name","business_address","country"]},
                         missing_utf8_is_empty_string=True)
        df.write_parquet(f"work/{split}_s{s}.parquet")
        print(split, s, df.shape, df["entity_id"].n_unique(), round(time.time()-t,1), flush=True)
gt = pl.read_csv(f"{base}/train/train_ground_truth.tsv", separator="\t", quote_char=None,
                 schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8}, missing_utf8_is_empty_string=True)
gt.write_parquet("work/train_gt.parquet"); print(gt.shape)
