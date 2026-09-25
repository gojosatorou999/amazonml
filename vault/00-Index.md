# Amazon ML Challenge 2026 — Business Entity Resolution

**Deadline:** 27 Sep 2026, 23:59 IST · **Submissions:** 5/day · **Metric:** macro F_0.5 (precision-heavy)

## Map
- [[Architecture]] — the pipeline, stage by stage
- [[Winning-Features]] — the differentiators, ranked by expected payoff
- [[Metric-Exploitation]] — why the decision layer is where the points are
- [[Blocking-Strategy]] — candidate generation + the "smaller is ranked higher" bonus
- [[Fair-Play-Boundary]] — what we may and may not touch
- [[Risks-And-Traps]] — France, leakage, calibration drift
- `10-Experiments/` — one note per run, auto-appended by the pipeline
- `20-Methodology/` — builds into the final `Documentation_template.md`
- `30-Entities/` — auto-generated resolved-entity notes (graph view = the ER clusters)

## Status
- [x] Repo scaffold, metric, expected-F decision layer
- [ ] Datasets dropped into `data/`
- [ ] EDA + validation split
- [ ] Blocking v1 → first submission
