# Leave-one-country-out validation

Each country group is predicted by models trained **only on the other** — the proxy for the unseen-country (France) trap in test.

All numbers are **out-of-fold**: every prediction comes from a model that never saw that
Source-1 entity. Folds are grouped by Source-1 entity.

## Decision-layer ablation

| decision rule | macro F0.5 | singleton F | has-match F | micro P | micro R |
|---|---|---|---|---|---|
| global threshold (tuned) | 0.99531 | 0.9859 | 0.9965 | 0.9964 | 0.9960 |
| global threshold + exclusivity ⭐ | 0.99535 | 0.9859 | 0.9966 | 0.9967 | 0.9956 |
| expected-F0.5, isotonic p | 0.99279 | 0.9859 | 0.9937 | 0.9973 | 0.9874 |
| expected-F0.5, isotonic p + exclusivity | 0.99247 | 0.9859 | 0.9933 | 0.9974 | 0.9871 |
| expected-F0.5, model p | 0.99509 | 0.9859 | 0.9963 | 0.9974 | 0.9930 |
| expected-F0.5, model p + exclusivity | 0.99497 | 0.9859 | 0.9962 | 0.9977 | 0.9927 |

Chosen for inference: **global threshold + exclusivity** (selected on out-of-fold data, not assumed).

## Blocking and candidate set (train)

| stage | pairs | recall of true pairs | pairs per S1 |
|---|---|---|---|
| full cross product | 148,496,000 | 1.0000 | 18,562 |
| fused retrieval pool | 265,375 | 0.9996 | 33.2 |
| **candidate_pairs (after learned prune)** | **13,875** | **0.9969** | **1.73** |

Reduction ratio vs cross product: **0.999907**. Candidates per S1: `{'mean': 1.734375, 'median': 2.0, 'p95': 4.0, 'max': 6, 'zero_share': 0.113}`. Prune threshold τ = 0.03.

### Prune curve (OOF stage-1 probability threshold)

| τ | true pairs lost | candidate pairs |
|---|---|---|
| 0.0005 | 0.0437% | 14,626 |
| 0.001 | 0.0509% | 14,356 |
| 0.002 | 0.0873% | 14,191 |
| 0.005 | 0.1164% | 14,049 |
| 0.01 | 0.1819% | 13,968 |
| 0.02 | 0.2328% | 13,903 |
| 0.03 | 0.2692% | 13,875 |
| 0.05 | 0.3128% | 13,838 |
| 0.075 | 0.3565% | 13,810 |
| 0.1 | 0.4001% | 13,796 |
| 0.15 | 0.4802% | 13,769 |
| 0.2 | 0.5238% | 13,754 |

## Resolved country groups

`{'IN': 'India', 'United States': 'US', 'US': 'US', 'India': 'India', 'USA': 'US'}`

## Learned lexicon

- mined abbreviations: 18
- mined legal/generic tail tokens: `['actions', 'and', 'anonyme', 'chemicals', 'company', 'consultancy', 'corporation', 'electricals', 'enterprises', 'fabrics', 'foods', 'incorporated', 'industries', 'liability', 'limited', 'llc', 'logistics', 'motors', 'packaging', 'par', 'private', 'sa', 'sarl', 'sas', 'services', 'simplifiee', 'societe', 'solutions', 'sons', 'systems', 'textiles', 'trading']`
- landmark cues: `['adj', 'adjacent', 'behind', 'beside', 'besides', 'cote', 'derriere', 'face', 'infront', 'near', 'next', 'nr', 'opp', 'opposite', 'pres']`

Timings (s): `{'prepare': 8.38, 'block': 32.05, 'pair_features': 18.61, 'fit+evaluate': 22.76}`
