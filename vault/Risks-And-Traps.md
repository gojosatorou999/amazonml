# Risks and Traps

**France.** Appears only in test. Never one-hot the country; use it as a blocking
partition and a boolean `country_match` feature. Every learned component must see
country as an opaque string. Guard: leave-one-country-out validation.

**Validation leakage.** Split by Source-1 entity. If we split pairs, the same entity
lands on both sides and the score inflates. Also: blocking for held-out entities must
search the *full* S2/S3 pool, or we validate against an unrealistically clean pool.

**Calibration drift on unseen countries.** Isotonic fitted on US/India may be wrong for
France. Fit a global calibrator and only add per-country refinement where a country has
enough validation mass; unseen countries fall back to global.

**Blocking recall is a hard ceiling.** Anything Phase B drops is unrecoverable
downstream. Track recall at *every* stage boundary, not just at the end.

**Submission budget: 5/day.** Local validation must be trustworthy enough that we spend
submissions confirming, not exploring. Build the harness before the model.

**Format rejection.** Every S1 test entity needs exactly one row, empty lists allowed,
S2-/S3- IDs only, no dupes, matches ⊆ candidates. Run `utils/validate_submission.py`
(Amazon-provided, drop it into `utils/`) as a Makefile precondition on every write.

**Over-fitting the public leaderboard.** It is a subset; the private split decides.
Trust the local macro-F_0.5 with its country-held-out fold over public deltas of <0.005.
