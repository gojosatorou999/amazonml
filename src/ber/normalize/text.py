"""Script- and language-agnostic surface normalisation.

Deliberately *not* tied to {US, India}: the test set contains France, and the
statement warns that country is an open set of labels.  Everything here is
driven by Unicode properties and corpus statistics, never by a hard-coded
country list.
"""
from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

_AMP = re.compile(r"\s*&\s*")
_NONALNUM = re.compile(r"[^0-9a-z]+")
_WS = re.compile(r"\s+")
_DIGIT_RUN = re.compile(r"\d+")


def fold(s: str) -> str:
    """Unicode NFKC -> accent/script fold -> lowercase -> punctuation squash.

    `unidecode` is a bundled transliteration table, not a network lookup, so it
    is fair-play safe.  It handles Devanagari->Latin for the India records and
    strips French diacritics (Societe/Societe) in one pass.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _AMP.sub(" and ", s)
    s = unidecode(s).lower()
    s = _NONALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


def tokens(s: str) -> list[str]:
    return fold(s).split()


def token_set(s: str) -> frozenset[str]:
    return frozenset(tokens(s))


def sorted_key(s: str) -> str:
    """Order-invariant signature -- kills word-order transpositions."""
    return " ".join(sorted(set(tokens(s))))


def char_ngrams(s: str, n: int = 3) -> frozenset[str]:
    p = f" {fold(s)} "
    return frozenset(p[i : i + n] for i in range(max(len(p) - n + 1, 0)))


def numbers(s: str) -> frozenset[str]:
    """Digit runs: house numbers, PIN/ZIP codes, suite numbers.

    Numeric agreement is one of the highest-precision address signals there is,
    and it is language-independent -- it survives transliteration entirely.
    """
    return frozenset(_DIGIT_RUN.findall(s or ""))


def acronym(s: str) -> str:
    """First letter of each token: 'Tata Consultancy Services' -> 'tcs'."""
    return "".join(t[0] for t in tokens(s) if t)
