# SUMMARY — what this is, what we built, and why it should win

*Read this first. ~10 minutes. Everything else in the repo is detail behind this page.*

---

## 1. The problem in one paragraph

Amazon gives us business records from **three sources** that describe the same real-world
businesses but share no IDs. Source 1 (S1) is a clean, deduplicated reference list. Sources 2
and 3 (S2, S3) are noisy copies: abbreviations (`Pvt Ltd` / `Private Limited`), typos
(`Clerwater`), word-order swaps, missing PIN codes, landmark addresses (`Near SBI ATM`), and
country labels spelled several ways (`US`, `USA`, `United States`). For **every S1 entity** we
must list which S2/S3 records are the same business — possibly none ("singleton"), one, or many.

We submit two files:

| file | what it is | how it is judged |
|---|---|---|
| `output/matching_results.tsv` | our final matches | **leaderboard**: macro F0.5 |
| `output/candidate_pairs.tsv` | the shortlist our model looked at | **reviewed by judges**: smaller shortlist per entity *at the same recall* ranks higher |

Two traps are built into the challenge: the **test set contains France**, which never appears in
training, and **external data is banned** (no geocoding, no business registries, no
internet lookups — instant disqualification).

---

## 2. How the score works, and why it shapes everything

For each S1 entity: `F0.5 = 1.25·TP / (|predicted| + 0.25·|true|)`, then averaged over all S1
entities. Three consequences most teams miss:

1. **A wrong match costs more than a missed match** (precision counts 2×).
2. **Singletons are free points.** If an entity has no true match, an *empty* answer scores 1.0,
   and any answer at all scores 0.0. Saying "nothing" is a real decision.
3. **The best cut-off is per entity, not global.** Because `|predicted|` is in the denominator,
   whether to add a 2nd candidate depends on what you already emitted *for that entity*.

Every competitor repo we found on GitHub ends with one global probability threshold tuned by grid
search. We end with a **decision layer that picks, per entity, the answer set that maximises
expected F0.5** given match probabilities — mathematically the right object for this metric (and we
measure which probabilities to trust, see §5).

---

## 3. The pipeline (what the code does, stage by stage)

```
raw TSVs
  │
  ├─ 1. PREPARE   normalise text; split names into identity tokens vs legal tail;
  │               parse addresses into slots (house no., street, locality, postcode, landmark)
  │               → everything learned from the data, nothing hard-coded (see §4)
  │
  ├─ 2. BLOCK     4 retrieval channels × both directions, fused by Reciprocal Rank Fusion
  │               ~33 candidates per S1, recall 99.5–99.98%
  │
  ├─ 3. PRUNE     stage-1 LightGBM scores the pool; keep pairs with p ≥ τ
  │               → THIS is candidate_pairs.tsv  (1.7–1.8 per S1, recall 99.2–99.5%)
  │
  ├─ 4. MATCH     stage-2 LightGBM on ~75 features incl. "context" features
  │               (how this pair compares to its competitors) → isotonic calibration
  │
  └─ 5. DECIDE    one-owner-per-record adjustment + expected-F0.5 set selection
                  (rule chosen on CV + leave-one-country-out, weighted by the measured
                   unseen-country share of the test file)  → matching_results.tsv
```

Code map: `src/ber/` — `prepare.py`, `normalize/`, `blocking/retrieve.py`, `features/`,
`models/gbm.py`, `decide/`, `pipeline.py`, `cli.py`, `reporting.py`, `service/`.

---

## 4. The ideas that make us stand out (ranked by how much they matter)

### A. Scoring edges (move the leaderboard)

| # | idea | what it does | where |
|---|---|---|---|
| 1 | **Expected-F0.5 decision layer** | For each entity, tries every "top-k" answer (including *empty*) and picks the one with the highest expected score. Singletons are won by the empty set competing on equal terms, not by a hand-tuned rule. | `decide/expected_f.py` |
| 2 | **Shift-aware choice of probabilities** | Expected-value reasoning needs real probabilities. We found isotonic calibration *breaks* on an unseen country, so the pipeline compares calibrated vs. model probabilities (and a threshold baseline) under both normal CV and leave-one-country-out, weighted by the unseen-country share it measures in the test file. | `pipeline.py` |
| 3 | **One-owner constraint** | S1 is deduplicated ⇒ a real S2/S3 record belongs to at most one S1 entity. When several entities claim a record, their probabilities are renormalised as competing hypotheses (`q = o / (1 + Σo)`, o = odds). Verified on train ground truth before use (0 violations). | `decide/assign.py` |
| 4 | **Context / competition features** | A 0.8 name match is decisive if it's the only one, meaningless if 40 chain branches score 0.8. Features: rank, gap to best, margin over runner-up — from *both* the entity's side and the record's side ("how many other entities want this record?"). | `features/context.py` |
| 5 | **Cross-source sibling support** | If S2-a confidently matches an entity and S3-b is a near-copy of S2-a, S3-b gains evidence. One round of propagation over the candidate graph = collective resolution. | `features/context.py` |
| 6 | **Soft-TF-IDF with abbreviation-aware token matching** | Rare-token agreement is near-proof, common-token agreement (`Pvt`, `Road`) is worth ~nothing. Tokens match softly: equal / abbreviation (`bd`≈`boulevard`, no dictionary) / typo (Jaro-Winkler). Classic Cohen et al. 2003 measure. | `features/pairwise.py` |
| 7 | **Identity tokens vs legal tail** | `Onyx Systems LLC` vs `Systems Onyx & Sons LLC` must match; `Harbor Consultancy LLC` vs `Quarry Maplecrest LLC` at the same address must *not*. Comparing non-legal tokens order-free fixed both error types. | `prepare.py` |

### B. Candidate-set edges (the separately-judged criterion)

| # | idea | result |
|---|---|---|
| 8 | **4 channels × 2 directions + RRF** | char n-grams (typos), address+name char n-grams, word TF-IDF (rare tokens), exact keys (postcode, house+street, name+house, acronym). Reverse direction (record → its best S1) rescues pairs that crowded-out generic names lose. |
| 9 | **Learned, adaptive pruning** | Competitors ship a fixed top-12 or top-20 per entity. We keep only pairs a stage-1 model finds plausible: **~1.8 candidates per entity** at ~99% recall — roughly 7–11× smaller. The cut adapts: an obvious match keeps 1 candidate, an ambiguous chain keeps 5; a hopeless entity keeps 0. |
| 10 | **Blocking + scaling reports, unprompted** | `reports/blocking_report.md` (recall ceiling, reduction ratio, candidates/entity distribution) and `reports/scaling.md` (timings at ×1…×8 data, fitted growth exponent, the linear-time capped mode). The judges said they'd analyse exactly this. |

### C. Generalisation + fair-play edges (protect us from the traps)

| # | idea | why it matters |
|---|---|---|
| 11 | **Country labels resolved by ER itself** | No country list. Labels are merged when near-identical records link them (`FR`↔`France` merge on the test file alone). Output on test: `{US, USA, United States}`, `{IN, India}`, `{FR, France}`. |
| 12 | **Everything learned from provided data** | Abbreviations (`rd→road`, `pvt→private`) mined from training pairs with a vote test that rejects coincidences; legal-suffix vocabulary mined from name endings (learns `sarl`, `sas` from test names transductively); landmark cue words mined from one-sided address clauses. No gazetteers. |
| 13 | **Leave-one-country-out validation, used for decisions** | Train on US only → predict India, and vice-versa, with thresholds carried over (no target labels). Not just reported: it feeds the decision-rule selection. |
| 14 | **Minimal, permissive, offline stack** | numpy, scipy, pandas, scikit-learn, LightGBM, rapidfuzz — all BSD/MIT. We even removed `Unidecode` (GPL) and replaced it with stdlib Unicode folding. No network calls anywhere. |

### D. Presentation edges (what judges see when they open the repo)

| # | idea |
|---|---|
| 15 | **Review console** (`make app && make serve`) — a React app on the real pipeline. **Resolve** a record you type, live, with the expected-F0.5 decision curve drawn; **Browse** every test entity (filters: no match, contested, unseen country); **Model** shows the blocking funnel, decision-rule table and learned lexicon. Every decision comes with exact **TreeSHAP** reasons. |
| 16 | **Review queue ranked by expected value of perfect information** — the pairs where a human label would raise expected F0.5 the most. Active learning with the competition metric as the acquisition function. |
| 17 | **Honest ablation** — six decision rules × two validation schemes in `reports/cv_report.md`, including the one where our headline idea (calibrated expected-F) *loses*. We don't claim a gain we didn't measure. |
| 18 | **11 behavioural tests** (`make test`) pin every claim the docs make — the metric's worked example, the expected-F decisions, abbreviation mining rejecting `sons→solutions`, etc. |

---

## 5. Current numbers

All numbers are measured, reproducible with the commands in §6, on the synthetic harness
(real data not yet in the repo). "Hold-out" = the synthetic **test** split, scored with its hidden
truth; it contains **France, which never appears in training** (≈19–20% of test entities).

| | clean synthetic (`data/`) | hard synthetic (`data_hard/`) |
|---|---|---|
| OOF macro F0.5 (5-fold, grouped by entity) | 0.99770 | 0.99126 |
| Leave-one-country-out macro F0.5 (chosen rule, honest) | 0.99590 | 0.98207 |
| **Hold-out test macro F0.5 (incl. unseen France)** | **0.99745** | **0.98956** |
| fused pool recall (test) | 99.98% | 99.75% |
| **candidate_pairs recall (test)** | **99.51%** | **99.22%** |
| **candidates per S1 entity (test)** | **1.70** (median 2, p95 4) | **1.81** (median 2, p95 4) |
| reduction ratio vs full cross product | 0.99975 | 0.99974 |
| end-to-end test inference time | ~20 s | ~23 s |

For comparison, public competitor repos report fixed top-12/top-20 candidate lists with 96–98.5%
blocking recall.

### The most important finding: calibration does not travel across countries

The decision-rule ablation on the hard set (`reports/hard/cv_report.md`):

| decision rule | CV | leave-one-country-out | selection objective (w = 18.5%) |
|---|---|---|---|
| global threshold (tuned) | 0.99114 | 0.98000 | 0.98907 |
| global threshold + exclusivity | 0.99114 | 0.98002 | 0.98908 |
| expected-F0.5, isotonic-calibrated p | 0.99100 | 0.97433 | 0.98791 |
| expected-F0.5, isotonic p + exclusivity | 0.99098 | 0.97496 | 0.98801 |
| expected-F0.5, model p | 0.99126 | 0.98205 | 0.98956 |
| **expected-F0.5, model p + exclusivity ⭐** | **0.99126** | **0.98207** | **0.98956** |

An isotonic calibrator fit on one country *under*-states match probability on another, so
expected-F becomes too timid on the unseen country (recall 0.962). The log-loss-trained model
probability degrades gracefully. The pipeline now makes this choice itself: it runs both CV schemes,
**measures w (the share of test entities whose country label never appears in train) from the test
file** and picks the rule maximising `(1−w)·CV + w·LOCO`. On the clean set it picks plain expected-F
on model p (the rules are within 0.0005 there); on the hard set expected-F + exclusivity. Measured,
not assumed.

### What the matcher relies on (top features by gain, hard set)

`ctx_excl_j` (exclusive-owner share) ≫ `ctx_p` (stage-1 p) > `ctx_gap_i` > `ctx_sum_j` > `ctx_support`
(cross-source sibling support). Context and competition features dominate — identity is comparative.

### Blocking scaling (`reports/hard/scaling.md`)

| records | exact mode | capped mode (`df_cap`) |
|---|---|---|
| 10k | 7.3 s | 5.7 s |
| 80k | 426 s | 36.9 s |
| fitted growth | N^1.96 (quadratic) | ~linear |

Exact mode is the default at challenge scale (best recall, affordable); capped mode is the
production path.


---

## 6. What still needs YOU (important)

1. **Drop the real data in.** Everything above was built and measured on our synthetic harness
   (`data/` = clean, `data_hard/` = chains + co-located businesses + heavy noise, France only in
   test). Put the real files at
   `data/train/train_source{1,2,3}.tsv`, `data/train/train_ground_truth.tsv`,
   `data/test/test_source{1,2,3}.tsv`. Nothing else changes.
2. **Copy Amazon's validator** to `utils/validate_submission.py` — `make validate` runs ours and
   then theirs.
3. Run:
   ```bash
   make all          # eda → train → predict → validate   (≈6 min on the synthetic sizes)
   make loco         # leave-one-country-out report
   make scaling      # blocking scaling report
   make app          # build the review console (needs Node 18+)
   make serve        # review console on http://127.0.0.1:8000
   make package TEAM=<your_team_name>   # builds <team>_submission.zip in the required layout
   ```
4. Read `reports/eda.md` first on real data: it tells us the singleton rate and whether the
   one-owner assumption holds. If it does **not** hold, the pipeline still measures and picks the
   best decision rule by itself — but tell me and I'll adjust.
5. Upload `output/matching_results.tsv` to the portal. Budget: 5/day — use them to *confirm*
   local CV, not to explore.

---

## 7. Research: what others did (GitHub, Sept 2026) and how we differ

| what others do | what we do instead |
|---|---|
| char-3/4-gram TF-IDF top-k (12–20) | 4 channels × 2 directions, RRF, then a learned adaptive prune (~1.8/entity) |
| 27–55 rapidfuzz features | Soft-TF-IDF + slot-wise address + identity/legal split + context + sibling features |
| LightGBM / XGB / CatBoost blend | two-stage cascade (pruner → matcher) with cross-fitting and calibration |
| one global threshold, grid-searched | per-entity expected-F0.5 decision (+ measured ablation) |
| greedy 1:1 bipartite matching | probabilistic one-owner renormalisation (keeps many-to-one from the S1 side) |
| hard-coded country / suffix lists | country groups, suffixes, abbreviations and landmark cues all mined from data |

Sources: public challenge repos (e.g. purvanshjoshi/business-entity-resolution,
ankitpaul6201/Amazon-ML-Challenge-26, Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution),
Splink's term-frequency adjustments (the same idea as our IDF weighting), Dedupe's learned blocking,
Ditto / Sudowoodo (transformer EM — deliberately *not* used: CPU budget, and the gradient-boosted
cascade already sits near the ceiling on these features).

---

## 8. Glossary

- **Blocking / candidate generation** — cheaply shortlisting plausible pairs so we never compare everything with everything.
- **Recall ceiling** — share of true pairs that survive blocking; anything dropped there can never be matched later.
- **RRF (Reciprocal Rank Fusion)** — combine several ranked lists by summing `1/(60 + rank)`; no score calibration needed.
- **OOF (out-of-fold)** — predictions made by a model that never trained on that entity; the only honest local score.
- **Calibration** — making "p = 0.8" actually mean right 80% of the time.
- **Expected F0.5** — the average score we'd get over all possible truths, weighted by our probabilities.
- **SHAP** — per-feature contribution to a single prediction.
