# Cross-validated results (entity folds)

All numbers are **out-of-fold**: every prediction comes from a model that never saw that
Source-1 entity. Folds are grouped by Source-1 entity.

## Decision-layer ablation

Two OOF schemes: **CV** (folds by entity) and **LOCO** (train on one country group, predict the
other; thresholds re-used from CV so no held-out-country label is seen). The test file has
**w = 20.0%** of Source-1 entities in a country never seen in training, so the rule is chosen by
`(1 − w)·CV + w·LOCO` — the expected score on a test set with that mix.

| decision rule | CV macro F0.5 | singleton F | has-match F | micro P | micro R | LOCO macro F0.5 | selection objective |
|---|---|---|---|---|---|---|---|
| global threshold (tuned) | 0.99765 | 0.9957 | 0.9979 | 0.9984 | 0.9965 | 0.99593 | **0.99731** |
| global threshold + exclusivity ⭐ | 0.99768 | 0.9968 | 0.9978 | 0.9986 | 0.9963 | 0.99592 | **0.99733** |
| expected-F0.5, isotonic p | 0.99717 | 0.9957 | 0.9974 | 0.9986 | 0.9953 | 0.99503 | **0.99674** |
| expected-F0.5, isotonic p + exclusivity | 0.99705 | 0.9957 | 0.9972 | 0.9987 | 0.9952 | 0.99490 | **0.99662** |
| expected-F0.5, model p | 0.99748 | 0.9957 | 0.9977 | 0.9984 | 0.9962 | 0.99599 | **0.99718** |
| expected-F0.5, model p + exclusivity | 0.99754 | 0.9957 | 0.9978 | 0.9985 | 0.9962 | 0.99600 | **0.99723** |

Chosen for inference: **global threshold + exclusivity** (selected on out-of-fold data, not assumed).

## Blocking and candidate set (train)

| stage | pairs | recall of true pairs | pairs per S1 |
|---|---|---|---|
| full cross product | 148,496,000 | 1.0000 | 18,562 |
| fused retrieval pool | 265,375 | 0.9996 | 33.2 |
| **candidate_pairs (after learned prune)** | **13,789** | **0.9969** | **1.72** |

Reduction ratio vs cross product: **0.999907**. Candidates per S1: `{'mean': 1.723625, 'median': 2.0, 'p95': 4.0, 'max': 6, 'zero_share': 0.11475}`. Prune threshold τ = 0.1.

### Prune curve (OOF stage-1 probability threshold)

| τ | true pairs lost | candidate pairs |
|---|---|---|
| 0.0005 | 0.0218% | 14,438 |
| 0.001 | 0.0218% | 14,250 |
| 0.002 | 0.0364% | 14,114 |
| 0.005 | 0.0437% | 13,999 |
| 0.01 | 0.0509% | 13,930 |
| 0.02 | 0.1164% | 13,880 |
| 0.03 | 0.1455% | 13,851 |
| 0.05 | 0.1964% | 13,823 |
| 0.075 | 0.2328% | 13,801 |
| 0.1 | 0.2692% | 13,789 |
| 0.15 | 0.3201% | 13,768 |
| 0.2 | 0.3492% | 13,758 |

## Resolved country groups

`{'IN': 'India', 'United States': 'US', 'US': 'US', 'India': 'India', 'USA': 'US'}`

## Learned lexicon

- mined abbreviations: 18
- mined legal/generic tail tokens: `['actions', 'and', 'anonyme', 'chemicals', 'company', 'consultancy', 'corporation', 'electricals', 'enterprises', 'fabrics', 'foods', 'incorporated', 'industries', 'liability', 'limited', 'llc', 'logistics', 'motors', 'packaging', 'par', 'private', 'sa', 'sarl', 'sas', 'services', 'simplifiee', 'societe', 'solutions', 'sons', 'systems', 'textiles', 'trading']`
- landmark cues: `['adj', 'adjacent', 'behind', 'beside', 'besides', 'cote', 'derriere', 'face', 'infront', 'near', 'next', 'nr', 'opp', 'opposite', 'pres']`

## Top matcher features (mean gain over the fold bag)

| feature | gain |
|---|---|
| `ctx_excl_j` | 4,119 |
| `ctx_sum_j` | 711 |
| `ctx_p` | 507 |
| `ctx_rank_j` | 338 |
| `ctx_gap_i` | 125 |
| `ctx_gap_j` | 121 |
| `ctx_sum_i` | 72 |
| `full_char_rank` | 62 |
| `full_char_sim` | 59 |
| `a_miss_b` | 55 |
| `ctx_support` | 50 |
| `st_soft` | 47 |
| `ctx_cnt_j` | 46 |
| `a_ratio` | 40 |
| `rrf` | 35 |
| `n_miss_a` | 32 |
| `n_tsort` | 32 |
| `a_tset` | 30 |
| `word_sim` | 26 |
| `n_soft` | 26 |

Timings (s): `{'prepare': 6.4, 'block': 35.88, 'pair_features': 18.65, 'fit+evaluate': 50.89, 'loco': 18.14}`
