# Leave-one-country-out validation

Each country group is predicted by models trained **only on the other** — the proxy for the unseen-country (France) trap in test.

All numbers are **out-of-fold**: every prediction comes from a model that never saw that
Source-1 entity. Folds are grouped by Source-1 entity.

## Decision-layer ablation

| decision rule | macro F0.5 | singleton F | has-match F | micro P | micro R |
|---|---|---|---|---|---|
| global threshold (tuned) | 0.98442 | 0.9770 | 0.9854 | 0.9908 | 0.9821 |
| global threshold + exclusivity ⭐ | 0.98464 | 0.9738 | 0.9860 | 0.9901 | 0.9838 |
| expected-F0.5, isotonic p | 0.97341 | 0.9596 | 0.9752 | 0.9873 | 0.9620 |
| expected-F0.5, isotonic p + exclusivity | 0.97444 | 0.9628 | 0.9759 | 0.9888 | 0.9618 |
| expected-F0.5, model p | 0.98258 | 0.9814 | 0.9827 | 0.9935 | 0.9726 |
| expected-F0.5, model p + exclusivity | 0.98260 | 0.9814 | 0.9828 | 0.9936 | 0.9725 |

Chosen for inference: **global threshold + exclusivity** (selected on out-of-fold data, not assumed).

## Blocking and candidate set (train)

| stage | pairs | recall of true pairs | pairs per S1 |
|---|---|---|---|
| full cross product | 148,760,000 | 1.0000 | 18,595 |
| fused retrieval pool | 260,279 | 0.9945 | 32.5 |
| **candidate_pairs (after learned prune)** | **14,755** | **0.9917** | **1.84** |

Reduction ratio vs cross product: **0.999901**. Candidates per S1: `{'mean': 1.844375, 'median': 2.0, 'p95': 4.0, 'max': 8, 'zero_share': 0.102875}`. Prune threshold τ = 0.002.

### Prune curve (OOF stage-1 probability threshold)

| τ | true pairs lost | candidate pairs |
|---|---|---|
| 0.0005 | 0.1168% | 15,722 |
| 0.001 | 0.1825% | 15,146 |
| 0.002 | 0.2774% | 14,755 |
| 0.005 | 0.3942% | 14,393 |
| 0.01 | 0.5329% | 14,197 |
| 0.02 | 0.6351% | 14,025 |
| 0.03 | 0.7008% | 13,964 |
| 0.05 | 0.8687% | 13,857 |
| 0.075 | 1.0293% | 13,779 |
| 0.1 | 1.1096% | 13,739 |
| 0.15 | 1.2775% | 13,688 |
| 0.2 | 1.4746% | 13,638 |

## Resolved country groups

`{'IN': 'India', 'United States': 'USA', 'US': 'USA', 'India': 'India', 'USA': 'USA'}`

## Learned lexicon

- mined abbreviations: 21
- mined legal/generic tail tokens: `['actions', 'and', 'anonyme', 'chemicals', 'company', 'consultancy', 'corporation', 'electricals', 'enterprises', 'fabrics', 'foods', 'imited', 'incorporated', 'industries', 'liability', 'limited', 'llc', 'logistics', 'motors', 'packaging', 'par', 'private', 'sa', 'sarl', 'sas', 'services', 'simplifiee', 'societe', 'solutions', 'sons', 'systems', 'td', 'textiles', 'trading']`
- landmark cues: `['adj', 'adjacent', 'behind', 'beside', 'besides', 'cote', 'derriere', 'face', 'infront', 'near', 'next', 'nr', 'opp', 'opposite', 'pres']`

Timings (s): `{'prepare': 7.49, 'block': 43.18, 'pair_features': 26.3, 'fit+evaluate': 28.18}`
