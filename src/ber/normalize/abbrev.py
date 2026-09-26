"""Abbreviation map mined from the training pairs -- no external gazetteer.

For every true match (S1 record, S2/S3 record) we look at the tokens that do
*not* agree between the two sides.  When a short token on one side is a
plausible abbreviation of a long token on the other (a prefix, or an ordered
subsequence sharing the first letter: rd->road, blvd->boulevard, pvt->private),
that is one vote for `short -> long`.  Votes aggregated across thousands of
pairs separate systematic abbreviations from coincidence: a typo ('streett')
never reaches the vote threshold because its corruption varies pair to pair.

Why mine rather than use a curated list:
  * fair play -- the rules ban external data augmentation; this is learned
    purely from the provided training pairs;
  * coverage -- it learns the variants that actually occur in *this* data,
    including transliteration pairs no public list has;
  * it is the same mechanism for names (corp, ltd, pvt) and addresses (rd, st).

Unseen-country abbreviations (a French 'bd' in test only) are handled
downstream by the abbreviation-aware soft token match in `features`, which
applies the same prefix / subsequence test without needing a mined entry.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from .text import tokens


def is_abbrev(short: str, long: str) -> bool:
    """`short` could abbreviate `long`: same first letter, ordered subsequence."""
    if len(short) >= len(long) or len(short) < 1 or short[0] != long[0]:
        return False
    if short.isdigit() or long.isdigit():
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def mine_abbreviations(
    pairs: Iterable[tuple[str, str]],
    min_votes: int = 5,
    min_share: float = 0.6,
    min_len_gap: int = 2,
    min_explained: float = 0.3,
) -> dict[str, str]:
    """Return {short: long} from (text_a, text_b) pairs of matched records."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    # How often each token is present on one side only.  A true abbreviation
    # explains a large share of its own one-sided occurrences ('rd' is unmatched
    # almost exactly when the other side says 'road'); a coincidental
    # subsequence ('sons' ~ 'solutions', 'the' ~ 'thistle') explains very few.
    opportunities: Counter[str] = Counter()
    for a, b in pairs:
        ta, tb = set(tokens(a)), set(tokens(b))
        only_a, only_b = ta - tb, tb - ta
        opportunities.update(only_a)
        opportunities.update(only_b)
        if not only_a or not only_b or len(only_a) > 8 or len(only_b) > 8:
            continue
        for x in only_a:
            for y in only_b:
                if len(y) - len(x) >= min_len_gap and is_abbrev(x, y):
                    votes[x][y] += 1
                elif len(x) - len(y) >= min_len_gap and is_abbrev(y, x):
                    votes[y][x] += 1

    out: dict[str, str] = {}
    for short, cands in votes.items():
        long, n = cands.most_common(1)[0]
        total = sum(cands.values())
        if n >= min_votes and n / total >= min_share and n / opportunities[short] >= min_explained:
            out[short] = long
    # never map a token that is itself a mined long form (avoids chains/cycles)
    longs = set(out.values())
    return {s: l for s, l in out.items() if s not in longs}


def expand(toks: Iterable[str], table: dict[str, str]) -> list[str]:
    return [table.get(t, t) for t in toks]
