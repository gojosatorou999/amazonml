"""Address component chunking into comparable slots.

A flat string distance over an address is dominated by whatever varies most --
reordering, missing PIN, a landmark clause -- none of which says anything about
identity.  We parse each address into slots and compare slot-wise:

    house      first house/municipal number ('33/40', '21', '12a')
    postcode   last standalone run of >= 5 digits (PIN, ZIP, code postal)
    street     alpha tokens of the clause holding the house number
    tail       alpha tokens of the trailing clauses (locality / city / region)
    tokens     every non-landmark alpha token (abbreviations expanded)
    landmark   True if a landmark clause ('Near SBI ATM') was detected and removed

Nothing here is country-specific.  Slot boundaries come from punctuation and
digit structure; the landmark cue list is a handful of generic prepositions in
the languages the statement mentions, extended at fit time with cue words mined
from the training pairs (tokens that open a clause present on only one side of a
true match).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import fold

# Generic landmark prepositions (EN / Indian-English / FR).  Mined cues are added
# on top of these by `mine_landmark_cues`.
BASE_LANDMARK_CUES = frozenset(
    "near opp opposite behind beside besides adjacent adj next nr infront "
    "pres face cote derriere".split()
)

_NUM = re.compile(r"\d+(?:\s*[/-]\s*\d+)*[a-z]?")
_CLAUSE_SPLIT = re.compile(r"[,;|]+")


@dataclass(slots=True)
class Address:
    house: str = ""
    postcode: str = ""
    numbers: frozenset[str] = frozenset()
    street: tuple[str, ...] = ()
    tail: tuple[str, ...] = ()
    tokens: tuple[str, ...] = ()
    landmark: bool = False
    landmark_tokens: tuple[str, ...] = field(default=())


def _clause_tokens(clause: str) -> list[str]:
    """Fold a clause but keep municipal numbers like '23/33' as one token."""
    raw = clause.lower()
    out: list[str] = []
    pos = 0
    for m in _NUM.finditer(raw):
        out.extend(fold(raw[pos : m.start()]).split())
        out.append(re.sub(r"\s+", "", m.group(0)))
        pos = m.end()
    out.extend(fold(raw[pos:]).split())
    return out


def _is_num(t: str) -> bool:
    return bool(t) and t[0].isdigit()


def parse_address(
    raw: str,
    expand: dict[str, str] | None = None,
    cues: frozenset[str] = BASE_LANDMARK_CUES,
) -> Address:
    if not raw:
        return Address()
    expand = expand or {}
    clauses = [c for c in (_clause_tokens(c) for c in _CLAUSE_SPLIT.split(raw)) if c]

    kept: list[list[str]] = []
    landmark_toks: list[str] = []
    for toks in clauses:
        # A landmark clause starts with a cue.  In comma-less addresses the cue
        # clause runs up to the first number, and swallows that number only when
        # another number follows ('Near Metro Pillar 42 33 Stonebridge Rd').
        out: list[str] = []
        i = 0
        while i < len(toks):
            if toks[i] in cues:
                j = i + 1
                while j < len(toks) and not _is_num(toks[j]):
                    j += 1
                if j + 1 < len(toks) and _is_num(toks[j]) and _is_num(toks[j + 1]):
                    j += 1
                landmark_toks.extend(toks[i:j])
                i = j
                continue
            out.append(toks[i])
            i += 1
        if out:
            kept.append(out)

    kept = [[expand.get(t, t) for t in c] for c in kept]
    flat = [t for c in kept for t in c]
    nums = [t for t in flat if _is_num(t)]

    postcode = ""
    for t in reversed(nums):
        digits = re.sub(r"\D", "", t)
        if len(digits) >= 5 and "/" not in t and "-" not in t:
            postcode = digits
            break
    house = next((t for t in nums if re.sub(r"\D", "", t) != postcode), "")

    street: list[str] = []
    if house:
        for c in kept:
            if house in c:
                idx = c.index(house)
                street = [t for t in c[idx + 1 :] if not _is_num(t)]
                # comma-less address: the street is the 2-3 tokens after the number
                if len(kept) == 1:
                    street = street[:3]
                break
    tail_clauses = [c for c in kept if house not in c][-2:] if len(kept) > 1 else [flat[-2:]]
    tail = [t for c in tail_clauses for t in c if not _is_num(t)]

    number_set = set()
    for t in nums:
        number_set.add(re.sub(r"\D", "", t) if len(re.sub(r"\D", "", t)) >= 5 else t)
        for part in re.split(r"[/-]", t):
            if part:
                number_set.add(part)

    return Address(
        house=house,
        postcode=postcode,
        numbers=frozenset(number_set),
        street=tuple(street),
        tail=tuple(tail),
        tokens=tuple(t for t in flat if not _is_num(t)),
        landmark=bool(landmark_toks),
        landmark_tokens=tuple(landmark_toks),
    )


def mine_landmark_cues(
    pairs: list[tuple[str, str]],
    min_count: int = 15,
    min_one_sided: float = 0.9,
) -> frozenset[str]:
    """Clause-opening words that almost always appear on only one side of a match.

    For every true (S1 address, S2/S3 address) pair, a clause present on one
    side but whose tokens are entirely absent from the other is extra
    information about *location* rather than identity -- a landmark.  Its first
    word is a cue candidate.  We keep words that open such clauses often and
    are almost never shared across the pair.
    """
    from collections import Counter

    opens: Counter[str] = Counter()
    shared: Counter[str] = Counter()
    anywhere: Counter[str] = Counter()     # every occurrence, any position
    for a, b in pairs:
        ta = set(fold(a).split())
        tb = set(fold(b).split())
        for raw, other in ((a, tb), (b, ta)):
            for clause in _CLAUSE_SPLIT.split(raw):
                toks = fold(clause).split()
                anywhere.update(toks)
                if len(toks) < 2 or _is_num(toks[0]):
                    continue
                if not (set(toks) & other):
                    opens[toks[0]] += 1
                elif toks[0] in other:
                    shared[toks[0]] += 1
    # A cue is a *function word*: it opens one-sided clauses and is rarely seen
    # anywhere else.  Locality names fail the second test ('Lantern Nagar' vs
    # 'Lantern Road'), which keeps them out of the cue list.
    mined = {
        w
        for w, c in opens.items()
        if c >= min_count
        and c / (c + shared[w]) >= min_one_sided
        and c / anywhere[w] >= 0.8
        and 2 <= len(w) <= 8
        and not w.isdigit()
    }
    return BASE_LANDMARK_CUES | frozenset(mined)
