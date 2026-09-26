# Business Entity Resolution — Methodology

**Team:** TEAM_NAME · **Challenge:** Amazon ML Challenge 2026 · **Metric:** macro F0.5

> Reproduce end-to-end: `make setup && make all` (CPU only, fully offline). Section numbers below
> map one-to-one to modules in `src/ber/`.

---

## 1. Methodology used

We treat the task as **metric-driven entity resolution**: every design decision is derived from
the scoring rule rather than from a generic similarity threshold.

Per Source-1 entity, `F0.5 = 1.25·TP / (|S| + 0.25·|T|)` (S = emitted set, T = true set),
macro-averaged over all entities including singletons. Three consequences drive the design:

1. `|S|` is in the denominator ⇒ the optimal cut is **per entity**, not a global threshold.
2. A singleton scores 1.0 only for an empty answer ⇒ **abstention is an action** with expected value.
3. Expected-value reasoning requires **calibrated** probabilities, not just a good ranking.

The pipeline is a five-stage cascade; every threshold and choice on the training side is made on
**out-of-fold predictions grouped by Source-1 entity** (never split by pair).

```
PREPARE → BLOCK (recall) → PRUNE (candidate_pairs.tsv) → MATCH + CALIBRATE → DECIDE (matching_results.tsv)
```

### 1.1 Data-driven normalisation (`normalize/`, `prepare.py`)

Nothing is hard-coded to the training countries; every vocabulary is learned from provided data.

| component | how it is learned |
|---|---|
| Unicode folding | stdlib NFKC → casefold → NFKD diacritic strip; dotted acronyms collapsed (`S.A.R.L.`→`sarl`) |
| Abbreviation map | For every true training pair, tokens present on one side only are aligned; `short→long` votes are counted when `short` is an ordered subsequence of `long` sharing the first letter. Kept only if the pair explains ≥30% of the short token's one-sided occurrences (rejects coincidences like `sons`~`solutions`). |
| Legal / generic tail | Iterative peeling of tokens with high P(last position) across all names (train + test names, transductive) — learns `private limited`, `llc`, `& sons`, and test-only forms like `sarl`, `sas`. |
| Identity tokens | All name tokens not in the legal vocabulary, order-free — robust to word-order transposition. |
| Address slots | house/municipal number (`23/33`), postcode (last ≥5-digit run), street clause, locality tail, landmark clause removed. Landmark cues = generic prepositions + words mined as openers of clauses present on only one side of true pairs. |
| Country groups | Labels merged when near-identical cross-source records link them (union-find over best-neighbour pairs). No country list; unseen countries resolve on the test file itself. |

### 1.2 Model training protocol

- 5-fold GroupKFold by Source-1 entity; blocking for held-out entities searches the **full** S2/S3 pool.
- Stage-2 features use stage-1 **OOF** probabilities (stacking without leakage).
- Isotonic calibration is **cross-fitted** (fit on 4 folds, applied to the 5th) for evaluation, and
  refit on all OOF rows for inference; whether calibrated or model probabilities feed the decision
  layer is itself selected (§3.3).
- Test inference averages the 5 fold models (bagging).
- Leave-one-country-out re-fit (folds = country groups, thresholds carried over) feeds the decision-rule selection.

---

## 2. Candidate generation / blocking strategy

**Goal:** maximum recall ceiling, then the *smallest* candidate set that preserves it.

### 2.1 Recall stage — fused retrieval pool (`blocking/retrieve.py`)

| channel | representation | catches |
|---|---|---|
| `name_char` | char 2–4-gram TF-IDF, core name | typos, spacing, concatenations |
| `full_char` | char 3-gram TF-IDF, core name + address | generic names disambiguated by address |
| `word` | word TF-IDF (sublinear), name + address + numbers | rare-token agreement |
| `keys` | inverted index: postcode, house+street, name+house, order-free name, concatenated name, acronym | exact structural agreement; blocks > 60 records skipped |

Each TF-IDF channel runs **both directions** (S1→R top-20 and R→S1 top-5): a record's best Source-1
entity is not always in that entity's own top-k when the entity has a generic name. Lists are
fused with **Reciprocal Rank Fusion** (k = 60); the fused top-30 plus every pair a reverse channel
ranked 1–2 form the pool.

### 2.2 Precision stage — learned adaptive prune

A stage-1 LightGBM scores every pool pair on the full pairwise feature set. We keep pairs with
`p₁ ≥ τ`, where τ is the largest value losing ≤ 0.3% of the pool's true pairs on OOF data.
Because the cut is on probability, not rank, the candidate count **adapts per entity**: 1 for an
unambiguous match, several for chain branches, 0 for a hopeless singleton.

**This pruned set is `candidate_pairs.tsv`** — exactly the rows the stage-2 matcher runs inference on.

### 2.3 Blocking quality

Measured on the synthetic harness (replace with real-data numbers from `reports/` after `make all`):

| stage | pairs / S1 (test) | recall of true pairs (test) | recall (train, OOF) |
|---|---|---|---|
| full cross product | 7,058 | 1.0000 | 1.0000 |
| fused retrieval pool | 32.6 | 0.9975 | 0.9945 |
| **candidate_pairs.tsv** | **1.81** (median 2, p95 4, max 6) | **0.9922** | **0.9919** |

(hard synthetic set; clean set: 1.70 per S1 at 0.9951 recall.) Reduction ratio 0.99974.

### 2.4 Scale

All retrieval is chunked sparse matmul with per-row top-k (memory O(chunk × |R|)). A `df_cap`
switch restricts the product to features with document frequency ≤ cap — an inverted index over
rare n-grams (prefix filtering) — bounding work per record and making growth linear. At production
scale the channels shard by resolved country group and postcode prefix; the key channel is a hash
join. Measured timings and fitted growth exponents: `reports/scaling.md`.

---

## 3. Model architecture and feature engineering

### 3.1 Architecture

Two-stage gradient-boosted cascade (LightGBM 4.5, MIT; ~2 MB total, far below the 8B cap):

| stage | input | output | role |
|---|---|---|---|
| Pruner (5-fold bag) | ~65 pairwise + retrieval features | p₁ | defines the candidate set |
| Matcher (5-fold bag) | pairwise + retrieval + **context** features | p₂ → isotonic → p | final probability |

Hyper-parameters: learning rate 0.05, 63 leaves, feature/bagging fraction 0.8, L2 = 1,
300 (pruner) / 500 (matcher) rounds, deterministic.

### 3.2 Features

| family | features |
|---|---|
| name | ratio / token-sort / token-set / partial ratio on the full expanded name; ratio, token-set, Jaro-Winkler on the core name and on the concatenated core; **Soft-TF-IDF** (Cohen et al. 2003) on full, core and identity tokens; max IDF of an unmatched rare token on each side; first-token agreement; acronym hit; **legal-form match** (abbreviation-aware: `llc`≈`limited liability company`) |
| address | Soft-TF-IDF on all tokens, street slot, locality slot; ratio / token-set / token-sort; postcode state (missing / one missing / equal / conflict) and 3-digit prefix; house-number state and partial agreement (`23/33` vs `23`); number-set Jaccard and conflict; landmark flags |
| country | same resolved country group; same raw label |
| density | log-frequency of the core name in S1 and in S2/S3 (chains, franchises) |
| retrieval | per-channel rank and similarity, RRF score, number of channels, fused rank |
| **context** (stage 2) | stage-1 p; rank / gap / margin over runner-up among the entity's candidates; the same from the record's side (competing Source-1 claims, count, gap); exclusive-owner share `o/(1+Σo)`; **cross-source sibling support** `max p(s,r')·sim(r,r')` |

Soft-TF-IDF token similarity: 1 if equal; 0.9 if one token is an ordered-subsequence abbreviation
of the other (dictionary-free — this carries unseen-language abbreviations like `bd`→`boulevard`);
else Jaro-Winkler, accepted at ≥ 0.88.

### 3.3 Decision layer (`decide/`)

1. **One-owner renormalisation.** S1 is deduplicated, so a record has at most one true owner
   (verified: 0 violations in train ground truth). Competing claims on a record are converted to a
   mutually exclusive posterior `q_i = o_i / (1 + Σ_j o_j)`, `o = p/(1−p)`; uncontested pairs are unchanged.
2. **Expected-F0.5 set selection.** With calibrated `p` sorted descending, the optimal set is a
   prefix; we evaluate all `n+1` prefixes exactly using Poisson-binomial PMFs for true positives
   emitted (X) and withheld (Y): `E[F(k)] = Σ P(X=x)P(Y=y)·1.25x/(k+0.25(x+y))`, with `F=1` for
   `k=0, x+y=0`. The empty set competes as `k = 0`.

Which combination is used is **selected on OOF macro F0.5**, alongside a tuned global threshold as
the baseline:

| decision rule | CV macro F0.5 | LOCO macro F0.5 | objective (w = 18.5%) |
|---|---|---|---|
| global threshold (tuned) | 0.99114 | 0.98000 | 0.98907 |
| global threshold + exclusivity | 0.99114 | 0.98002 | 0.98908 |
| expected-F0.5, isotonic p | 0.99100 | 0.97433 | 0.98791 |
| expected-F0.5, isotonic p + exclusivity | 0.99098 | 0.97496 | 0.98801 |
| expected-F0.5, model p | 0.99126 | 0.98205 | 0.98956 |
| **expected-F0.5, model p + exclusivity (chosen)** | **0.99126** | **0.98207** | **0.98956** |

*LOCO* re-fits with folds = country groups and re-uses the CV thresholds, so no held-out-country
label is used. `w` is the share of test Source-1 entities whose country label never occurs in
training, measured from the test file. The rule maximising `(1−w)·CV + w·LOCO` is used for
inference. Key finding: isotonic calibration fit on one country under-states probabilities on
another (expected-F recall drops to 0.962), whereas the log-loss-trained model probability
transfers — so the chosen rule uses model probabilities.

---

## 4. Other relevant information

### 4.1 Fair play and licences

- No external data, APIs, geocoders, gazetteers or registries. No network access at any stage.
- All vocabularies (abbreviations, legal forms, landmark cues, country groups) are learned from
  the provided train/test files.
- Dependencies: numpy, scipy, pandas (BSD); scikit-learn (BSD); LightGBM (MIT); rapidfuzz (MIT).
  No pretrained language model is used.

### 4.2 Unseen country

Country is never one-hot encoded or filtered; labels are resolved into groups by ER itself; the
legal-form vocabulary is mined transductively from test names; abbreviation matching is
dictionary-free. LOCO results:

see the LOCO column in §3.3. Held-out synthetic test (France ≈ 19% of entities, never in
training): macro F0.5 **0.98956** on the hard set, **0.99745** on the clean set; France's resolved
country group (`FR` ∪ `France`) is discovered on the test file alone.

### 4.3 Explainability and review tooling

`make serve` launches a stdlib-only review UI: per-entity candidates with token alignment,
calibrated probabilities, exact TreeSHAP attributions, and a review queue ordered by the
**expected value of perfect information** — the metric as the active-learning acquisition function.

### 4.4 Experiments log (chronological)

| # | change | effect (hard synthetic) |
|---|---|---|
| 1 | baseline cascade: 4-channel RRF blocking, pairwise features, pruner + matcher, expected-F | OOF 0.9909, hold-out 0.9865 |
| 2 | identity tokens (order-free non-legal tokens) + abbreviation-aware legal-form match | hold-out 0.9865 → 0.9886 (fixed transposition FNs and same-address different-business FPs) |
| 3 | abbreviation miner: normalise votes by one-sided occurrences | removed spurious `sons→solutions`, `the→thistle`, `mp→maplecrest` |
| 4 | stdlib Unicode folding replaces Unidecode (GPL) + dotted-acronym collapse | licence-clean; `S.A.R.L.`→`sarl` |
| 5 | sparse top-k for df-capped retrieval | capped blocking growth quadratic → ~linear (×8 data: 426 s → 37 s) |
| 6 | LOCO-aware decision-rule selection (model p vs isotonic p) | LOCO 0.9750 → 0.9821; hold-out 0.9896 |
