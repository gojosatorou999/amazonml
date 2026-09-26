# Blocking report — test

- Source 1 entities: **3,000**; Source 2+3 records: **6,895**
- Full cross product: **20,685,000** pairs
- Fused retrieval pool: **98,428** pairs (32.8 per S1)
- **candidate_pairs.tsv: 5,104 pairs — 1.70 per S1** (median 2, p95 4, max 6, empty 11.7%)
- Reduction ratio: **0.999753**
- Final matched pairs: **5,087**; entities predicted as singletons: **11.7%**
- Resolved country groups: `{'IN': 'IN', 'USA': 'USA', 'India': 'IN', 'France': 'France', 'FR': 'France', 'US': 'USA', 'United States': 'USA'}`
- Timings (s): `{'prepare': 1.06, 'block': 6.55, 'pair_features': 8.07, 'score+decide': 2.88}`

## Synthetic hold-out (test truth is known only for the synthetic harness)

- pool recall **0.9998**, candidate recall **0.9951**
- macro F0.5 **0.99749** (singletons 0.9943, has-match 0.9979)

## Recall ceiling (measured on train, out-of-fold)

- pool recall **0.9996** → candidate recall **0.9969** at **1.72** candidates per S1
