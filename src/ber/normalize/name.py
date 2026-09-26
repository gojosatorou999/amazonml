"""Business-name decomposition: core name vs legal/generic tail.

'Summit Packaging & Sons Incorporated' carries two kinds of tokens: the ones
that identify *this* business (summit, packaging) and the ones that describe
its legal form (and, sons, incorporated).  Sources disagree about the second
kind all the time, so it must not dominate the name similarity.

The legal-tail vocabulary is mined, not listed.  A legal token is one that
lives at the *end* of names: we compute P(token is last | token occurs) over
the whole corpus, peel off tokens above a threshold, then recompute on the
peeled names -- iterating turns 'private limited' and '& sons ltd' into
multi-token tails without any of them being hard-coded.  Mining over
train + test names means an unseen country's forms (SARL, SAS, GmbH) are learned
transductively from the test file itself.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def mine_legal_tokens(
    names: Iterable[list[str]],
    rounds: int = 4,
    min_last_share: float = 0.55,
    min_df: float = 0.002,
) -> frozenset[str]:
    names = [n for n in names if n]
    n_docs = max(len(names), 1)
    legal: set[str] = set()
    cur = names
    for _ in range(rounds):
        occ: Counter[str] = Counter()
        last: Counter[str] = Counter()
        for toks in cur:
            if len(toks) < 2:
                continue
            occ.update(set(toks))
            last[toks[-1]] += 1
        new = {
            t
            for t, c in last.items()
            if occ[t] / n_docs >= min_df and c / occ[t] >= min_last_share and not t.isdigit()
        }
        new -= legal
        if not new:
            break
        legal |= new
        cur = [strip_tail(t, legal)[0] for t in cur]
    return frozenset(legal)


def strip_tail(toks: list[str], legal: frozenset[str] | set[str]) -> tuple[list[str], list[str]]:
    """Split tokens into (core, legal_tail).  Never strips the whole name."""
    end = len(toks)
    while end > 1 and toks[end - 1] in legal:
        end -= 1
    return toks[:end], toks[end:]
