# Winning Features — ranked by expected payoff

Tier S = likely decides our rank. Tier A = strong, cheap. Tier B = demo/judging surface.

## Tier S — scoring edges
**1. Expected-F_0.5 per-entity set selection.** [[Metric-Exploitation]]. Others ship a
global threshold. Implemented: `ber/decide/expected_f.py`.

**2. Global one-to-many assignment.** Source 1 is *deduplicated*, so a real S2/S3 record
describes one business and can belong to at most one S1 entity. If two S1s both claim
S2-00047, at most one is right. Verify the constraint holds in `train_ground_truth.tsv`,
then enforce it as a competitive assignment (greedy by margin, or min-cost flow if it
pays). Pure precision gain — and F_0.5 weights precision 2×.

**3. Adaptive-k blocking.** [[Blocking-Strategy]]. Wins the explicitly-stated
"smaller candidate set ranks higher" bonus, which sits *outside* the leaderboard.

**4. Corpus-IDF-weighted similarity.** "Pvt Ltd Restaurant" is near-zero evidence;
a rare token match is near-proof. IDF computed over train+test pools. Most teams use
unweighted Jaccard/Levenshtein and silently reward generic tokens.

**5. Discriminability / local-density features.** A name match is weak evidence when 40
similar businesses sit in the same postcode, strong when it is unique. Feature =
similarity *relative to* the candidate's nearest competitors, not absolute. This is how
you stop false merges on chains and franchises — the classic ER killer.

**6. Collective / graph features.** S2↔S3 similarity as a feature: if S2-a and S3-b are
near-duplicates of each other and S2-a confidently matches S1-x, that is evidence for
S3-b. One round of propagation over the candidate graph. Turns independent pair
classification into joint inference.

## Tier A — quality edges
**7. Abbreviation map mined from the data, not from a public list.** Align matched pairs
in train, extract systematic substitutions (corp↔corporation, pvt↔private, rd↔road,
transliteration variants). Fair-play safe ([[Fair-Play-Boundary]]), and it learns
variants no curated gazetteer contains.

**8. Address semantic chunking.** Parse into slots {house_no, street, locality, city,
region, postcode, landmark} and compare **slot-wise**, with landmark text
("Near SBI ATM") demoted to weak evidence rather than polluting the string distance.
Slot-wise comparison is what makes partial addresses survivable.

**9. Numeric-signature agreement.** Digit runs (house no., postcode) are
language-independent and survive transliteration intact — the single most robust
cross-script signal available, and the one that will carry France.

**10. Domain-adapted bi-encoder.** Contrastive fine-tune (MultipleNegativesRanking) on
train pairs with hard negatives mined from blocking. Off-the-shelf embeddings are tuned
for sentences, not for noisy business names.

**11. Leave-one-country-out validation.** Train on US, evaluate on India, to prove the
France path works. Report the ablation in the writeup — it shows the trap was seen.

## Tier B — the judging surface others ignore
**12. `blocking_report.md` + `scaling.md`, unprompted.** They told us they will analyse
recall ceiling and reduction ratio. Hand them the exact analysis, pre-computed.

**13. Explainable matches.** Per prediction: SHAP attributions over the feature vector
plus token-level alignment highlighting between the two records. ER without an audit
trail is unshippable in production; a panel that shows *why* reads as engineering
maturity.

**14. Review queue ordered by expected-F gain.** Surface the pairs where a human
decision would move the score most — that is computable directly from the E[F] curve's
flatness. Active learning, with the metric as the acquisition function.

**15. Obsidian vault as the deliverable's knowledge layer.** Every resolved entity
becomes a note wikilinked to its source records; Obsidian's graph view renders the ER
clusters natively, and the methodology doc lives in the same vault it was researched in.
Unusual, genuinely useful, and it demos in ten seconds.

**16. Reproducibility theatre that is not theatre.** `make all` regenerates both TSVs
from raw data, seeds locked, artifact hashes manifested, fully offline. The top packages
get reproduced by hand; ours should run first try.
