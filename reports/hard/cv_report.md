# Cross-validated results (entity folds)

All numbers are **out-of-fold**: every prediction comes from a model that never saw that
Source-1 entity. Folds are grouped by Source-1 entity.

## Decision-layer ablation

Two OOF schemes: **CV** (folds by entity) and **LOCO** (train on one country group, predict the
other; thresholds re-used from CV so no held-out-country label is seen). The test file has
**w = 18.5%** of Source-1 entities in a country never seen in training, so the rule is chosen by
`(1 − w)·CV + w·LOCO` — the expected score on a test set with that mix.

| decision rule | CV macro F0.5 | singleton F | has-match F | micro P | micro R | LOCO macro F0.5 | selection objective |
|---|---|---|---|---|---|---|---|
| global threshold (tuned) | 0.99114 | 0.9880 | 0.9915 | 0.9953 | 0.9887 | 0.98000 | **0.98907** |
| global threshold + exclusivity | 0.99114 | 0.9880 | 0.9915 | 0.9953 | 0.9887 | 0.98002 | **0.98908** |
| expected-F0.5, isotonic p | 0.99100 | 0.9880 | 0.9914 | 0.9955 | 0.9870 | 0.97433 | **0.98791** |
| expected-F0.5, isotonic p + exclusivity | 0.99098 | 0.9880 | 0.9914 | 0.9955 | 0.9869 | 0.97496 | **0.98801** |
| expected-F0.5, model p | 0.99126 | 0.9880 | 0.9917 | 0.9955 | 0.9883 | 0.98205 | **0.98956** |
| expected-F0.5, model p + exclusivity ⭐ | 0.99126 | 0.9880 | 0.9917 | 0.9955 | 0.9883 | 0.98207 | **0.98956** |

Chosen for inference: **expected-F0.5, model p + exclusivity** (selected on out-of-fold data, not assumed).

## Blocking and candidate set (train)

| stage | pairs | recall of true pairs | pairs per S1 |
|---|---|---|---|
| full cross product | 148,760,000 | 1.0000 | 18,595 |
| fused retrieval pool | 260,279 | 0.9945 | 32.5 |
| **candidate_pairs (after learned prune)** | **13,965** | **0.9919** | **1.75** |

Reduction ratio vs cross product: **0.999906**. Candidates per S1: `{'mean': 1.745625, 'median': 2.0, 'p95': 4.0, 'max': 6, 'zero_share': 0.113125}`. Prune threshold τ = 0.03.

### Prune curve (OOF stage-1 probability threshold)

| τ | true pairs lost | candidate pairs |
|---|---|---|
| 0.0005 | 0.0292% | 15,577 |
| 0.001 | 0.0438% | 15,053 |
| 0.002 | 0.0949% | 14,669 |
| 0.005 | 0.1606% | 14,341 |
| 0.01 | 0.2190% | 14,167 |
| 0.02 | 0.2555% | 14,034 |
| 0.03 | 0.2628% | 13,965 |
| 0.05 | 0.3285% | 13,906 |
| 0.075 | 0.3942% | 13,855 |
| 0.1 | 0.4818% | 13,820 |
| 0.15 | 0.5548% | 13,785 |
| 0.2 | 0.6132% | 13,754 |

## Resolved country groups

`{'IN': 'India', 'United States': 'USA', 'US': 'USA', 'India': 'India', 'USA': 'USA'}`

## Learned lexicon

- mined abbreviations: 21
- mined legal/generic tail tokens: `['actions', 'and', 'anonyme', 'chemicals', 'company', 'consultancy', 'corporation', 'electricals', 'enterprises', 'fabrics', 'foods', 'imited', 'incorporated', 'industries', 'liability', 'limited', 'llc', 'logistics', 'motors', 'packaging', 'par', 'private', 'sa', 'sarl', 'sas', 'services', 'simplifiee', 'societe', 'solutions', 'sons', 'systems', 'td', 'textiles', 'trading']`
- landmark cues: `['adj', 'adjacent', 'behind', 'beside', 'besides', 'cote', 'derriere', 'face', 'infront', 'near', 'next', 'nr', 'opp', 'opposite', 'pres']`

## Top matcher features (mean gain over the fold bag)

| feature | gain |
|---|---|
| `ctx_excl_j` | 11,387 |
| `ctx_p` | 3,028 |
| `ctx_gap_i` | 1,021 |
| `ctx_sum_j` | 555 |
| `ctx_support` | 245 |
| `ctx_sum_i` | 222 |
| `rrf` | 202 |
| `ctx_cnt_j` | 190 |
| `ctx_rank_j` | 171 |
| `a_miss_b` | 152 |
| `word_rev_sim` | 150 |
| `ctx_margin_i` | 140 |
| `n_miss_a` | 128 |
| `a_ratio` | 121 |
| `n_tsort` | 111 |
| `n_partial` | 111 |
| `word_rank` | 104 |
| `word_sim` | 102 |
| `n_soft` | 100 |
| `a_miss_a` | 87 |

Timings (s): `{'prepare': 7.7, 'block': 43.64, 'pair_features': 25.97, 'fit+evaluate': 71.44, 'loco': 23.47}`
