# Architecture

Five stages. Each writes a cached artifact so any stage can be re-run alone.

```
 raw TSV
   │
 ┌─▼──────────────┐  ber/normalize/    fold · legal-suffix strip · core-name extraction
 │ 1 NORMALISE    │                    ADDRESS SEMANTIC CHUNKING -> slots:
 │   + CHUNK      │                    {house_no, street, locality, city, region,
 └─┬──────────────┘                     postcode, landmark, country}
   │                                   mined abbreviation map (from train pairs)
 ┌─▼──────────────┐  ber/blocking/     4 recall channels, unioned by Reciprocal Rank Fusion:
 │ 2 BLOCK        │                     a. char-ngram TF-IDF ANN  (sparse_dot_topn)
 │   (recall)     │                     b. Okapi BM25 on name+address tokens (bm25s)
 └─┬──────────────┘                     c. dense bi-encoder + FAISS HNSW (multilingual)
   │                                    d. deterministic keys: sorted-token, acronym,
   │                                       postcode, house-no+street, numeric signature
 ┌─▼──────────────┐  ber/blocking/     LightGBM ranker on ~12 cheap features
 │ 3 PRUNE        │  prune.py          ADAPTIVE-K: per-entity cut from the score gap.
 │   (precision)  │                    ── this output IS candidate_pairs.tsv ──
 └─┬──────────────┘                    target: mean ≤ 6 candidates/S1 at ≥98% recall
   │
 ┌─▼──────────────┐  ber/models/       ensemble, blended:
 │ 4 SCORE        │                     · LightGBM on ~90 engineered similarity features
 │                │                     · fine-tuned cross-encoder (reranker, ≤600M)
 └─┬──────────────┘                     · collective/graph consistency features
   │                                    -> isotonic calibration -> p(match)
 ┌─▼──────────────┐  ber/decide/       expected-F_0.5 prefix selection  (expected_f.py)
 │ 5 DECIDE       │                    + global one-to-many assignment  (assign.py)
 └─┬──────────────┘                      (S1 is deduplicated => each S2/S3 record
   │                                      belongs to at most one S1 — verify on train)
   ▼
 matching_results.tsv + candidate_pairs.tsv + blocking_report.md + Obsidian vault
```

## Why an ensemble rather than one big model
The 8B cap is generous; the *time* budget is not. GBDT on engineered features is the
proven workhorse for ER and trains in seconds, which means many iterations. The
cross-encoder catches the semantic cases features cannot (DBA/trade names,
transliteration, reordered multilingual tokens). Blending is where the top of the
leaderboard lives; running only one of the two is the common failure.

## Validation harness (build this before any modelling)
- Split by **Source-1 entity**, not by pair. Keep the *full* S2/S3 pool available to
  blocking for held-out entities so distractors are realistic.
- **Leave-one-country-out** fold (train on US → evaluate on India) as the explicit
  proxy for the France generalisation trap. Report it in the writeup.
- Every run appends to `vault/10-Experiments/` with config hash, metric breakdown,
  and candidates/S1. Version history is required by the rules anyway.
