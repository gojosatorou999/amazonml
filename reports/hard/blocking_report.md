# Blocking report — test

- Source 1 entities: **3,000**; Source 2+3 records: **7,058**
- Full cross product: **21,174,000** pairs
- Fused retrieval pool: **97,681** pairs (32.6 per S1)
- **candidate_pairs.tsv: 5,431 pairs — 1.81 per S1** (median 2, p95 4, max 6, empty 10.6%)
- Reduction ratio: **0.999744**
- Final matched pairs: **5,188**; entities predicted as singletons: **11.3%**
- Resolved country groups: `{'IN': 'IN', 'India': 'IN', 'United States': 'USA', 'FR': 'France', 'France': 'France', 'US': 'USA', 'USA': 'USA'}`
- Timings (s): `{'prepare': 1.26, 'block': 7.74, 'pair_features': 9.85, 'score+decide': 4.28}`

## Synthetic hold-out (test truth is known only for the synthetic harness)

- pool recall **0.9975**, candidate recall **0.9922**
- macro F0.5 **0.98956** (singletons 0.9793, has-match 0.9909)

## Recall ceiling (measured on train, out-of-fold)

- pool recall **0.9945** → candidate recall **0.9919** at **1.75** candidates per S1
