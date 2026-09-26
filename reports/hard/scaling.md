# Blocking scaling benchmark

Records replicated from `data_hard/test` ×1…×8 (fresh IDs, one character dropped per replica name). Single machine, 8 CPU threads, no GPU. Reproduce: `make scaling`.

| factor | records | mode | seconds | fused pool pairs / S1 |
|---|---|---|---|---|
| x1 | 10,058 | capped | 5.74 | 34.4 |
| x1 | 10,058 | exact | 7.28 | 32.6 |
| x2 | 20,116 | capped | 14.59 | 32.5 |
| x2 | 20,116 | exact | 25.53 | 31.3 |
| x4 | 40,232 | capped | 16.68 | 32.6 |
| x4 | 40,232 | exact | 99.43 | 31.7 |
| x8 | 80,464 | capped | 36.89 | 35.6 |
| x8 | 80,464 | exact | 426.43 | 32.5 |

## Fitted growth  t = a · N^b

| mode | exponent b | projected wall-clock at 10^9 records (one 8-thread box) |
|---|---|---|
| exact | **1.96** | 11,693,877 hours |
| capped | **0.82** | 23 hours |

## Reading this

- **exact** scores each S1 entity against every record sharing *any* n-gram — quadratic (b ≈ 2), the same
  behaviour as a brute-force TF-IDF cosine. At challenge scale it is affordable and has the best recall, so
  it is the default for the submission.
- **capped** (`retrieve(..., df_cap=1000)`) scores only through features with document frequency ≤ cap — an
  inverted index over rare n-grams/tokens (prefix filtering) with a sparse top-k that never materialises
  a dense row. Work per record is bounded ⇒ **~linear growth**; ×8 data costs ×6.4 time.
- Replication is a *pessimistic* test for the cap: every replica multiplies each n-gram's document frequency,
  whereas genuinely new businesses bring new rare tokens.
- Beyond one machine: capped channels shard by resolved country group and postcode prefix (embarrassingly
  parallel); the key channel is a hash join; the pruner and matcher are per-pair and stream. No stage needs
  an all-pairs pass, and the per-S1 candidate count is flat (~33 in the pool, ~2 after pruning) as N grows.
- Honest caveat: the synthetic names come from a ~26-word vocabulary, so few n-grams are rare and the capped
  mode loses recall there (99.5% → 86.6% pool recall on `data_hard/train`). Real business names are far more
  diverse, which is the regime prefix filtering is designed for — re-measure with `make scaling` on real data.
