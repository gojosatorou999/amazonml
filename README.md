# Business Entity Resolution — Amazon ML Challenge 2026

Resolves noisy business records from three independent sources against the deduplicated
Source-1 reference and produces both required outputs:

| file | what it is |
|---|---|
| `output/matching_results.tsv` | final matches — the leaderboard submission |
| `output/candidate_pairs.tsv` | the exact set the matching model runs inference on (the *last* filtering stage) |

**New here? Read [`SUMMARY.md`](SUMMARY.md) first** — the approach in plain language.
The methodology write-up for the submission zip is [`docs/Documentation_template.md`](docs/Documentation_template.md).

---

## Reproduce end-to-end

```bash
make setup      # venv + pinned, CPU-only dependencies (numpy, scipy, pandas, scikit-learn, lightgbm, rapidfuzz)
make all        # eda -> train -> predict -> validate  => output/*.tsv + reports/
```

Individual stages:

```bash
make eda        # profile the data; verify the one-owner-per-record assumption -> reports/eda.md
make train      # cross-fitted pruner + matcher + calibration -> artifacts/, reports/cv_report.md
make predict    # blocking -> prune -> match -> decide -> output/*.tsv, reports/blocking_report.md
make validate   # our checker, then Amazon's utils/validate_submission.py if present
make loco       # leave-one-country-out validation -> reports/loco_report.md
make scaling    # blocking scaling benchmark -> reports/scaling.md
make app        # build the React review console (Node 18+)
make serve      # API + review console at http://127.0.0.1:8000
make package TEAM=name   # <name>_submission.zip in the organisers' layout
```

Without `make`: `PYTHONPATH=src python -m ber.cli all --data data`.

### Getting the dataset

The real competition data (2.5 GB, every TSV > GitHub's 100 MB file limit) is attached to the
`dataset-v1` release of this private repo rather than committed. From the repo root:

```bash
gh release download dataset-v1 --dir dist_release
unzip dist_release/train.zip -d 6ab10eb3b23ba_student_resource/student_resource
unzip dist_release/test.zip  -d 6ab10eb3b23ba_student_resource/student_resource
```

This restores `6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}/`, next to the
organisers' `utils/validate_submission.py`, `README.md` and `Documentation_template.md` (those
are committed). Findings from the first look at the real data are in `HANDOFF.md`.

### Data layout

```
data/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

`make synth` / `make synth-hard` generate schema-identical synthetic datasets reproducing the
statement's noise taxonomy (hard mode adds chain branches, co-located businesses and heavier
noise; a third country appears only in test). They are a development harness only and are never
used to train the submitted model.

---

## Fair-play compliance

**The pipeline runs fully offline on CPU. It makes no network call at any stage.**

| used | why it is inside the boundary |
|---|---|
| LightGBM (MIT) — the only model, ~2 MB | trained here on the provided data only; far below the 8B cap |
| stdlib Unicode folding (NFKC / NFKD) | no transliteration table, no external data |
| `rapidfuzz`, `scikit-learn` TF-IDF | pure algorithms, no data |
| IDF statistics, legal-form vocabulary, country groups computed over provided train + test records | transductive use of provided data, not external data |
| Abbreviation map and landmark cues **mined from the training pairs** | learned from challenge data; no gazetteer of suffixes, street types, cities or countries |

No geocoding, registries, commercial ER services or internet augmentation. No pretrained language
model is used. All dependencies are BSD or MIT.

---

## Design rationale: the metric drives everything

`F0.5 = 1.25·TP / (|S| + 0.25·|T|)` per Source-1 entity, macro-averaged. Therefore:

1. **The optimal cut is per entity** — `|S|` is in the denominator, so a global threshold cannot express it.
2. **Singletons are a decision** — empty output scores 1.0 on a true singleton, anything else 0.0.
3. **Calibration matters more than ranking** — expected-value decisions need real probabilities.

The pipeline ends in a per-entity expected-F0.5 decision over match probabilities, and the
choice between it and a tuned global threshold is **made on out-of-fold data** and reported
(`reports/cv_report.md`).

## Pipeline

```
 raw TSV
   │
 ┌─▼──────────────┐ prepare.py, normalize/   fold · identity tokens vs mined legal tail
 │ 1 PREPARE      │                          address slots {house, postcode, street, locality, landmark}
 └─┬──────────────┘                          mined abbreviations · country groups resolved by ER
 ┌─▼──────────────┐ blocking/retrieve.py     4 channels × 2 directions, Reciprocal Rank Fusion
 │ 2 BLOCK        │                           name char-ngrams · name+address char-ngrams ·
 │   (recall)     │                           word TF-IDF · exact keys (postcode, house+street, …)
 └─┬──────────────┘
 ┌─▼──────────────┐ pipeline.py              stage-1 LightGBM, keep p ≥ τ  (τ: ≤0.3% OOF recall loss)
 │ 3 PRUNE        │                          adaptive per entity: 0, 1 … n candidates
 └─┬──────────────┘                          ── this output IS candidate_pairs.tsv ──
 ┌─▼──────────────┐ features/, models/       stage-2 LightGBM: pairwise + context + sibling features
 │ 4 MATCH        │                          cross-fitted isotonic calibration
 └─┬──────────────┘
 ┌─▼──────────────┐ decide/                  one-owner renormalisation  q = o/(1+Σo)
 │ 5 DECIDE       │                          expected-F0.5 prefix selection (empty set competes)
 └─┬──────────────┘
   ▼
 matching_results.tsv · candidate_pairs.tsv · reports/
```

### Blocking
Each channel covers another's blind spot: char n-grams catch typos and concatenations, word
TF-IDF catches rare-token agreement, exact keys catch structural agreement, and the combined
name+address channel disambiguates generic names. Every TF-IDF channel also runs **record → S1**,
rescuing pairs a generic S1 name's crowded forward list would drop. The learned prune then cuts
the ~30-per-entity pool to a handful **by probability, not rank**, so the count adapts per entity.
The organisers rank smaller candidate sets higher at equal recall; `reports/blocking_report.md`
reports recall ceiling, reduction ratio and the per-entity distribution.

### Features
Soft-TF-IDF (IDF-weighted, soft token matching: equal / dictionary-free abbreviation /
Jaro-Winkler) over full name, core name, identity tokens and address slots; classic edit and
token ratios; postcode / house-number agreement states; legal-form agreement (abbreviation-aware);
name density; retrieval ranks. Stage 2 adds **context** features — the pair's rank, gap and margin
among the entity's candidates and among the record's competing S1 claimants — and **cross-source
sibling support** (one round of propagation over the candidate graph).

### Validation
- Folds grouped by Source-1 entity; held-out entities still block against the full S2/S3 pool.
- Every downstream choice (prune τ, calibration, decision rule) is made on OOF predictions.
- Leave-one-country-out (`make loco`) as the explicit proxy for the unseen-country trap.
- `src/ber/eval/metric.py` is an exact reimplementation of the metric (worked example: 0.7143).

---

## Generated reports

| report | contents |
|---|---|
| `reports/eda.md` | sizes, singleton rate, match-size distribution, one-owner check, country labels |
| `reports/cv_report.md` | OOF decision-rule ablation, recall at every stage, prune curve, learned lexicon, feature importance |
| `reports/loco_report.md` | the same, with folds = country groups |
| `reports/blocking_report.md` | test-side candidate statistics, reduction ratio, timings |
| `reports/scaling.md` | blocking time at ×1…×8 data, fitted growth exponent, exact vs linear capped mode |

---

## Repository layout

```
src/ber/
├── cli.py                 entry point: eda | train | predict | loco | validate | all
├── pipeline.py            train (cross-fitted) and predict
├── prepare.py             Lexicon (learned surface forms) + record preparation
├── normalize/             text folding, address slots, legal tail, abbreviation mining, country groups
├── blocking/retrieve.py   retrieval channels, bidirectional top-k, RRF fusion, df-capped mode
├── features/              pairwise.py (Soft-TF-IDF etc.), context.py (competition + sibling support)
├── models/gbm.py          LightGBM cross-fitting, bagging, cross-fitted isotonic calibration
├── decide/                expected_f.py (E[F0.5] prefix selection, EVPI), assign.py (one-owner)
├── eval/                  metric.py, validate.py, scaling.py
├── reporting.py           markdown reports
├── export/package.py      submission zip
├── service/               stdlib API: live resolver, TreeSHAP, EVPI queue; serves app/dist
└── synth.py               synthetic harness
app/                             React + TypeScript review console (Vite)
docs/Documentation_template.md   methodology write-up
SUMMARY.md                       plain-language overview
```

## Review console (`app/`)

A React + TypeScript app (Vite, no UI kit, fonts bundled so it runs offline) on top of a
standard-library Python API (`src/ber/service/`). Build once, then serve:

```bash
make predict    # produces the artifacts the console reads
make app        # npm ci && npm run build  ->  app/dist
make serve      # API + console on http://127.0.0.1:8000
make dev        # optional: hot-reload on :5173, proxying /api to make serve
```

| view | what it does |
|---|---|
| **Resolve** | Type a messy record and run it through the *submitted* pipeline live: the same retrieval index (exact reverse ranks), features, pruner, context features computed against the whole batch, matcher and decision rule. Shows the expected-F0.5 curve over how many records to emit, each candidate with shared words highlighted, and TreeSHAP reasons. An optional one-owner toggle shows when a record already belongs to an existing Source 1 entity. |
| **Browse** | Every test entity, filterable to matched, no match, contested (a record claimed by several entities) and unseen country. Shows the candidates blocking cut, not just the ones it kept. |
| **Review** | Entities ranked by expected value of perfect information: how much expected F0.5 rises if a person labels one pair. j / k to step through. |
| **Model** | Log-scale blocking funnel, the decision-rule table (cross-validation vs leave-one-country-out), what the matcher relies on, and the learned lexicon (country groups, abbreviations, legal forms, landmark words). |

Design: colour is reserved for source identity (S1/S2/S3, the first three slots of a
validated categorical palette, all-pairs CVD-checked in light and dark), every source mark is
also labelled, light/dark/system themes, keyboard navigation, reduced-motion respected,
responsive to phone width. Without a build, `make serve` falls back to a single-file page.
