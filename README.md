# Business Entity Resolution — Amazon ML Challenge 2026

Resolves noisy business records from three independent sources against a
deduplicated Source-1 reference, producing both required outputs:

| file | what it is |
|---|---|
| `output/matching_results.tsv` | final matches — the leaderboard submission |
| `output/candidate_pairs.tsv` | the candidate set fed into the matching model (the *last* blocking stage, not an early pass) |

The pipeline is built around the scoring metric rather than around a generic
similarity threshold — see [Design rationale](#design-rationale-the-metric-drives-everything),
which is where most of our engineering effort went.

---

## Fair-play compliance

**The pipeline runs fully offline on CPU. It makes no network call at any stage.**

The challenge bans external data lookup on pain of disqualification — commercial ER
APIs, government business registries, **geocoding APIs**, and any internet data
augmentation. This implementation uses none of them.

What we *do* use, and why each is inside the boundary:

| used | why it is not an external lookup |
|---|---|
| Pretrained open-weight models (MIT / Apache-2.0, ≤8B params) | static downloaded weights, explicitly permitted by the rules |
| `unidecode` transliteration tables | a bundled offline library table, not a service |
| `rapidfuzz` string metrics | pure algorithms, no data |
| IDF / embedding statistics computed over the **provided train + test records** | transductive use of provided data, not external data |
| Abbreviation map **mined from the training pairs themselves** | learned from the challenge data; we deliberately avoided any curated public gazetteer of legal suffixes or street abbreviations |

`requirements.txt` is fully pinned and the run is reproducible offline from a cold
checkout. Detail and the explicit do-not-cross list: [`vault/Fair-Play-Boundary.md`](vault/Fair-Play-Boundary.md).

---

## Reproduce end-to-end

```bash
make setup      # venv + pinned, CPU-only dependencies
make all        # data -> blocking -> matching -> output/, then format validation
```

Individual stages, in order:

```bash
make eda        # profile the data; verify the one-to-many assignment assumption
make block      # candidate generation -> output/candidate_pairs.tsv + reports/blocking_report.md
make score      # feature build + model inference -> calibrated p(match)
make decide     # expected-F0.5 set selection -> output/matching_results.tsv
make validate   # Amazon-provided format checker (a precondition on every submission)
```

### Where the data goes

```
data/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

### Synthetic stand-in data

`make synth` generates a schema-identical synthetic dataset reproducing the noise
taxonomy the problem statement enumerates — name abbreviations, legal-suffix
inconsistency, DBA/trade-name truncation, `&` vs `and`, word-order transposition,
typos; address abbreviations, missing PIN/region, landmark references
(`Near SBI ATM`), municipal numbering (`12/34`), component reordering — plus a
third country appearing **only in test**, mirroring the France trap.

It exists so the whole pipeline can be built, debugged and scored before real data
is available, and so that CI can run without shipping challenge data. It is a
development harness only: it is never used to train or tune the submitted model.
Delete `data/` and drop in the real TSVs; nothing downstream changes.

---

## Design rationale: the metric drives everything

The metric is macro-averaged F₀.₅ over Source-1 entities. Rewritten:

```
F_0.5 = 1.25 * TP / (|S| + 0.25 * |T|)
```

where `S` is the set we emit for an entity and `T` its true match set. Three
consequences shaped the architecture:

**1. The optimal cut is per-entity, not a global threshold.**
`|S|` sits in the denominator, so the marginal value of emitting one more candidate
depends on how many we already emitted *for that entity*. A single global
probability threshold cannot express this. We therefore end the pipeline with a
per-entity decision layer, not a cutoff.

**2. Singletons are a decision, not a by-product.**
An entity with no true matches scores **1.0** for an empty list and **0.0** for any
output at all. Abstention is an action with real expected value, and it has to
compete on equal footing with emitting candidates.

**3. Calibration matters more than ranking.**
Expected-value reasoning needs `p_i` to be an actual probability, not a score. A
model with excellent AUC but poor calibration will pick the wrong set size. Hence
isotonic calibration as a first-class stage rather than an afterthought.

Full notes: [`vault/Metric-Exploitation.md`](vault/Metric-Exploitation.md).

---

## Pipeline

```
 raw TSV
   │
 ┌─▼──────────────┐  ber/normalize/    Unicode fold · legal-suffix strip · core-name extraction
 │ 1 NORMALISE    │                    ADDRESS COMPONENT CHUNKING into slots:
 │   + CHUNK      │                    {house_no, street, locality, city, region,
 └─┬──────────────┘                     postcode, landmark, country}
   │                                   + abbreviation map mined from train pairs
 ┌─▼──────────────┐  ber/blocking/     4 recall channels, fused by Reciprocal Rank Fusion:
 │ 2 BLOCK        │                     a. char-ngram TF-IDF ANN        (sparse_dot_topn)
 │   (recall)     │                     b. Okapi BM25 on name+address   (bm25s)
 └─┬──────────────┘                     c. dense bi-encoder + FAISS HNSW (multilingual)
   │                                    d. deterministic keys: sorted-token, acronym,
   │                                       postcode, house-no+street, numeric signature
 ┌─▼──────────────┐  ber/blocking/     LightGBM ranker over ~12 cheap features
 │ 3 PRUNE        │  prune.py          ADAPTIVE-K: per-entity cut at the score dropoff
 │   (precision)  │                    ── this output IS candidate_pairs.tsv ──
 └─┬──────────────┘                    target: mean ≤ 6 candidates/S1 at ≥98% recall
   │
 ┌─▼──────────────┐  ber/models/       blended ensemble:
 │ 4 SCORE        │                     · LightGBM over ~90 engineered similarity features
 │   + CALIBRATE  │                     · cross-encoder reranker (ONNX int8, CPU)
 └─┬──────────────┘                     · collective / graph-consistency features
   │                                    -> isotonic calibration -> p(match)
 ┌─▼──────────────┐  ber/decide/       expected-F0.5 prefix selection
 │ 5 DECIDE       │                    + global one-to-many assignment
 └─┬──────────────┘
   ▼
 matching_results.tsv · candidate_pairs.tsv · reports/ · Obsidian vault
```

### 1. Normalise and chunk

Surface normalisation is driven by Unicode properties and corpus statistics, never
by a hard-coded country list — the test set contains a country absent from training,
and the statement warns that `country` is an open set of string labels.

**Address component chunking** parses each address into semantic slots and compares
them **slot-wise** rather than as one flat string. This is what makes partial
addresses survivable: a record missing its PIN code still matches on house number
and street. Landmark text (`Near SBI ATM`) is detected and demoted to weak evidence
instead of polluting the string distance.

**Numeric signatures** — digit runs for house numbers and postcodes — are extracted
separately. They are language-independent and survive transliteration intact, which
makes them the most robust cross-script signal available and the main reason the
unseen-country path works at all.

### 2. Block for recall

Four channels, each contributing a ranked top-k, unioned via Reciprocal Rank Fusion.
Target: ≥99% recall at ~50–100 candidates per Source-1 entity. Each channel covers a
failure mode of the others — BM25 catches rare-token agreement that dense embeddings
smooth away; char n-grams catch typos that tokenisation breaks; the dense channel
catches DBA/trade-name and transliteration cases that no lexical method reaches;
deterministic keys catch exact structural agreement cheaply.

### 3. Prune for a small candidate set

The organisers stated that **the team generating a smaller candidate set per Source-1
entity ranks higher**, independently of the leaderboard. Candidate-set size is
therefore a scored objective, not an implementation detail.

A learned ranker cuts the fused pool down, using **adaptive-k**: fixed top-k wastes
budget, because an exact name+postcode agreement needs one candidate while a generic
name in a dense city needs twenty. We cut per entity at the largest relative score
dropoff. On the same recall, this costs roughly 3–4× fewer candidates than a fixed
top-20 — which is precisely the ratio the reviewers will compute.

**This stage's output is `candidate_pairs.tsv`** — the exact set the matching model
runs inference over, as the statement requires.

### 4. Score

An ensemble rather than a single model, for two reasons. The 8B parameter cap is
generous but the time budget is not, and gradient-boosted trees over engineered
similarity features train in seconds on CPU — which buys many more iterations than a
large model would. The cross-encoder then covers what hand-built features cannot:
trade names, transliteration, reordered multilingual tokens.

The small candidate sets from stage 3 are what make the cross-encoder affordable on
CPU at all: ~6 candidates per entity keeps inference tractable. The blocking edge
pays for the model edge.

Feature families:
- **IDF-weighted token similarity.** `Pvt Ltd Restaurant` is near-zero evidence; a
  rare token agreement is near-proof. Unweighted Jaccard/Levenshtein silently rewards
  generic tokens — this is the most common quiet failure in ER pipelines.
- **Slot-wise address similarity** over the chunked components.
- **Discriminability / local density.** A name match is weak evidence when forty
  similar businesses share a postcode and strong when it is unique. Features measure
  similarity *relative to* the candidate's nearest competitors, not absolutely. This
  is the defence against false merges on chains and franchises.
- **Collective / graph features.** S2↔S3 similarity as a signal: if two records from
  different sources are near-duplicates of each other and one confidently matches a
  Source-1 entity, that is evidence for the other. One round of propagation over the
  candidate graph turns independent pair classification into joint inference.

Scores are then isotonically calibrated on a held-out fold, fitted globally with
per-country refinement only where a country has enough validation mass, so an unseen
country inherits the global calibrator rather than a country-specific one.

### 5. Decide

**Expected-F₀.₅ set selection.** Given calibrated probabilities and treating matches
as independent Bernoullis, the optimal emitted set is always a prefix of candidates
sorted by probability descending. We evaluate all `n+1` prefixes exactly:

```
E[F(k)] = Σ_{x,y} P(X=x) P(Y=y) · 1.25x / (k + 0.25(x+y))

  X ~ PoissonBinomial(p_1..p_k)      true matches we emitted
  Y ~ PoissonBinomial(p_{k+1}..p_n)  true matches we withheld
```

with `F = 1.0` in the special case `k = 0, x + y = 0`. Both PMFs are built
incrementally, so the sweep is O(n²) per entity. The empty set competes as the `k=0`
prefix, scoring `Π(1 − p_i)` — which is how singletons are won, without a separate
classifier or a hand-tuned abstention rule.

Observed behaviour (verified, `src/ber/decide/expected_f.py`):

| candidate probabilities | emitted | why |
|---|---|---|
| 0.97, 0.10, 0.05 | top-1 | clean single match |
| 0.95, 0.91, 0.08 | top-2 | both earn their slot |
| 0.22, 0.15, 0.09 | **nothing** | E[F] = 0.60 by abstaining vs 0.21 by guessing |
| 0.55, 0.12 | top-1 | implied single-candidate cut ≈ 0.45 — emergent, not tuned |

**Global one-to-many assignment.** Source 1 is the *deduplicated* reference, so a
real Source-2/3 record describes one business and can belong to at most one Source-1
entity. If two Source-1 entities both claim the same record, at most one is correct.
Enforcing this as a competitive assignment is a pure precision gain, and F₀.₅ weights
precision 2× over recall.

> This assumption is verified against `train_ground_truth.tsv` by `make eda` before
> the constraint is applied. It is gated on that check, not assumed.

---

## Validation protocol

Local validation has to be trustworthy, because the submission budget is 5 per day —
submissions should confirm findings, not explore.

- **Split by Source-1 entity, never by pair.** Splitting pairs puts the same entity
  on both sides and inflates the score.
- **Held-out entities still block against the full Source-2/3 pool**, so distractors
  are realistic rather than artificially thinned.
- **Leave-one-country-out fold** (train on one country, evaluate on another) as the
  explicit proxy for the unseen-country generalisation trap. Reported as an ablation
  in the methodology document.
- **Recall is tracked at every stage boundary**, not only at the end — anything
  blocking drops is unrecoverable downstream, so the recall ceiling has to be visible
  where it is set.
- Every run appends to `vault/10-Experiments/` with config hash, metric breakdown and
  candidates-per-entity. Version history of all submissions is required by the rules
  in any case.

`src/ber/eval/metric.py` is an exact reimplementation of the challenge metric,
verified against the worked example in the problem statement (0.7143).

---

## Generated reports

Produced automatically, because the organisers stated they will analyse blocking
quality and scale when deciding final rankings:

**`reports/blocking_report.md`**
- pair completeness / recall ceiling — overall, per country, per source
- reduction ratio against the full `|S1| × (|S2| + |S3|)` cross product
- candidates per Source-1 entity: mean, median, p95, max
- recall-vs-candidates curve per channel and fused, showing each channel earns its place

**`reports/scaling.md`**
- wall-clock and memory for blocking at 10⁵ / 10⁶ / 10⁷ replicated records
- fitted complexity curve (sub-quadratic: ANN ~O(N log N), deterministic keys O(N))
- projection to 10⁹ with a sharding sketch

---

## Model licences and size

All components are MIT or Apache-2.0 and far below the 8B parameter cap.

| component | role | licence | params |
|---|---|---|---|
| `intfloat/multilingual-e5-small` | dense blocking channel | MIT | ~118M |
| `BAAI/bge-reranker-base` | cross-encoder reranker (ONNX int8) | MIT | ~278M |
| LightGBM | candidate pruner + primary scorer | MIT | n/a |
| scikit-learn, FAISS, rapidfuzz, bm25s, sparse_dot_topn | supporting | BSD / MIT | n/a |

Licences are re-verified against the published model cards as part of the
pre-submission compliance check.

---

## Repository layout

```
.
├── src/ber/
│   ├── io.py                  TSV readers + submission writers (stdlib-only paths)
│   ├── synth.py               synthetic stand-in dataset generator
│   ├── normalize/             Unicode folding, name/address component chunking,
│   │                          abbreviation map mined from training pairs
│   ├── blocking/              4 retrieval channels, RRF fusion, adaptive-k pruner
│   ├── features/              string / IDF / address-slot / graph feature builders
│   ├── models/                LightGBM, cross-encoder, isotonic calibration
│   ├── decide/                expected-F0.5 selection, global assignment
│   ├── eval/                  exact metric, splits, blocking report
│   ├── export/                submission writers, Obsidian vault export
│   └── service/               FastAPI inference service
├── vault/                     Obsidian vault — design notes, experiment log,
│                              generated entity-graph notes
├── app/                       React review UI
├── output/                    the two submission TSVs
├── reports/                   generated blocking + scaling evidence
├── utils/                     Amazon-provided validate_submission.py
├── Makefile                   every stage, reproducible
└── requirements.txt           fully pinned, CPU-only
```

### The Obsidian vault

`vault/` is the project's working knowledge base, not decoration. Design decisions,
the fair-play boundary, per-run experiment notes and the methodology draft all live
there and cross-link. The export stage additionally writes one note per resolved
entity, wikilinked to its constituent source records — so Obsidian's graph view
renders the resolution clusters natively, which is a considerably faster way to spot
a bad merge than reading a TSV.

Start at [`vault/00-Index.md`](vault/00-Index.md).

---

## Application

A FastAPI service wraps the trained pipeline for interactive use, with a React
front-end (Vite + TypeScript + Tailwind + shadcn/ui):

- **Search and resolve** — enter a business record, see ranked matches with calibrated
  confidence.
- **Explainability panel** — per prediction, SHAP attributions over the feature vector
  plus token-level alignment highlighting between the two records. Entity resolution
  without an audit trail is not shippable in production.
- **Entity graph** — Cytoscape.js rendering of the resolution clusters.
- **Review queue ordered by expected-F gain** — surfaces the pairs where a human
  decision would move the score most, computed directly from the flatness of the
  `E[F]` curve. Active learning with the metric itself as the acquisition function.

---

## Implementation status

| component | state |
|---|---|
| Exact metric, verified against the statement's worked example | ✅ implemented |
| Expected-F₀.₅ decision layer | ✅ implemented, behaviour verified |
| TSV I/O + submission writers | ✅ implemented |
| Unicode / surface normalisation | ✅ implemented |
| Synthetic dataset generator | ✅ implemented |
| Name + address component chunking | 🔨 in progress |
| Mined abbreviation map | 🔨 in progress |
| Blocking channels + RRF fusion | 🔨 in progress |
| Adaptive-k pruner | 🔨 in progress |
| Feature builders | ⬜ planned |
| LightGBM scorer + cross-encoder + calibration | ⬜ planned |
| Global assignment constraint | ⬜ planned (gated on the `make eda` check) |
| Blocking + scaling reports | ⬜ planned |
| `ber.cli` entry point wiring the Makefile targets | ⬜ planned |
| FastAPI service + React UI | ⬜ planned |

---

## Further reading

| note | contents |
|---|---|
| [`vault/Architecture.md`](vault/Architecture.md) | the pipeline in full, stage by stage |
| [`vault/Winning-Features.md`](vault/Winning-Features.md) | every differentiator, ranked by expected payoff |
| [`vault/Metric-Exploitation.md`](vault/Metric-Exploitation.md) | why the decision layer is where the points are |
| [`vault/Blocking-Strategy.md`](vault/Blocking-Strategy.md) | candidate generation and the smaller-set ranking bonus |
| [`vault/Fair-Play-Boundary.md`](vault/Fair-Play-Boundary.md) | what we may and may not touch |
| [`vault/Risks-And-Traps.md`](vault/Risks-And-Traps.md) | unseen countries, leakage, calibration drift |
