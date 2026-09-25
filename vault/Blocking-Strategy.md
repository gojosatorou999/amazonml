# Blocking Strategy

The organisers said it twice: blocking must scale, and **the team with the smaller
candidate set per Source-1 entity ranks higher**, independently of the leaderboard.
So candidate-set size is a *scored objective*, not an engineering detail.

## The two-phase shape
**Phase A — recall, generously.** Union of four channels, each contributing top-k,
fused with Reciprocal Rank Fusion. Target: ≥99% recall at ~50–100 candidates/S1.
**Phase B — prune, aggressively.** A learned ranker cuts to a handful. This is the
file we submit. Target: **≥98% recall at ≤6 candidates/S1.**

## Adaptive-k — the actual differentiator
Fixed top-k wastes the budget: an exact name+postcode hit needs 1 candidate, a generic
name in a dense city needs 20. We cut per entity where the pruner score *drops off*
(largest relative gap, floored by the expected-F layer's own need for a tail). A fixed
top-20 on the same recall costs ~3–4× the mean candidate count. That ratio is exactly
what the judges will compute.

## Scale evidence (do this, nobody else will)
They said "billions of records". Ship `reports/scaling.md`: replicate the record pool
to 10^5 / 10^6 / 10^7, plot wall-clock and memory for blocking, fit the curve, state
the complexity (sub-quadratic: ANN is ~O(N log N), deterministic keys are O(N)), and
project to 10^9 with a sharding sketch. It answers their stated #1 concern with a graph
instead of a claim.

## Metrics we report for them, unprompted -> `reports/blocking_report.md`
- pair completeness / recall ceiling (overall, per country, per source)
- reduction ratio vs the full |S1|×(|S2|+|S3|) cross product
- candidates per S1: mean, median, p95, max
- recall-vs-candidates curve, per channel and fused (shows each channel earns its place)
