# Handoff: first look at the real dataset (2026-09-26)

Everything in `src/ber/` was developed on a 3k-row synthetic harness. The real data is ~4000x
larger, and the current pipeline will not run on it as-is.

## Scale

| split | S1 | S2 | S3 | ground truth rows |
|---|---|---|---|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 | 2,206,821 |
| test  | 1,732,544 | 4,887,273 | 5,082,316 | — |

## Ground-truth facts (train)

- Matches per S1: mean 3.46; distribution 0:5.6%, 1:5.4%, 2:17%, 3:24%, 4:22%, 5:15%, 6:7.5%, 7+:4%.
- Singleton rate 5.6% (123,247 S1 entities with no match).
- One-owner holds exactly: 7,638,365 matched IDs, all unique (no S2/S3 record belongs to two S1s).
- ~26% of S2/S3 records match nothing, so they are pure distractors.
- Countries: train is US (60%) / India (40%) in every source; test adds France.
- S1 addresses are never empty; ~3.4% of S2/S3 addresses are empty.

## Noise seen in samples

- India: many S2/S3 names are in Devanagari (`व्हाइट बिल्डर्स प्राइवेट लिमिटेड` = White Builders
  Private Limited). Needs transliteration or a learned token dictionary; addresses carry most signal.
- Domain-style names: `WHITEBUILDERS.COM`, `raabmoderntreasury.com` (concatenated name).
- DBA: `Aviyuma One DBA: Velez Schwab`; legal suffix moved to front: `Llc Raab Modern Treasury,`.
- Typos in names and streets (`Treoasubr,y`, `Strhet`), accents injected (`Prógram`, `Léarning`).
- Address reordering (`Aurora, 2893 Dorothy Drive, IL`), state abbreviated vs full, missing house
  numbers, ranges (`3231-3233`), and even S1's own house number can disagree with all its matches.
- France (test only): `R.` = Rue, `Sarl`, `S.A.S`, `SCI`, accented names.

## Why the current code will not scale

- `blocking/retrieve.py` fits char 2-4-gram TF-IDF over all S1+R texts (~12M docs) and does
  bidirectional sparse products: memory blows up on a 16 GB machine.
- Pairwise features are computed row-by-row in Python: ~2M S1 x ~30 candidates = ~60M pairs.

## Planned direction (not yet implemented)

- Columnar processing (polars/pyarrow) instead of pandas object columns.
- Key-based blocking with composite keys (name token x locality, house no. x street token,
  concatenated name, name-token pairs for empty addresses, Devanagari -> Latin skeleton keys),
  block-size caps, and a reverse-direction pass exploiting one-owner.
- Vectorised pair features (`rapidfuzz.process.cpdist` with workers=-1, or GPU), LightGBM on a
  subsample of train S1 entities, held-out S1 split for F0.5 validation.
- Target: macro F0.5 > 0.99 on a held-out train split.
