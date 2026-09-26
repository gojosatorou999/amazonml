"""Script- and language-agnostic surface normalisation.

Deliberately *not* tied to {US, India}: the test set contains France, and the
statement warns that country is an open set of labels.  Everything here is
driven by Unicode properties and corpus statistics, never by a hard-coded
country list.
"""
from __future__ import annotations

import re
import unicodedata

_AMP = re.compile(r"\s*&\s*")
_NONWORD = re.compile(r"[\W_]+")
_WS = re.compile(r"\s+")
_DIGIT_RUN = re.compile(r"\d+")
_DOTTED = re.compile(r"(?<!\w)(?:[^\W\d_]\.){2,}(?:[^\W\d_](?!\w))?")
_LIGATURES = str.maketrans({"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "ł": "l", "đ": "d", "ı": "i"})


def fold(s: str) -> str:
    """NFKC -> lowercase -> strip diacritics -> punctuation squash.

    Standard-library only (no transliteration table, no licence questions):
    NFKD decomposition separates base letters from combining marks, which are
    dropped, so 'Société Générale' and 'Societe Generale' fold identically.
    Letters of any script survive as word characters rather than being deleted.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _AMP.sub(" and ", s).lower().translate(_LIGATURES)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = _DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)   # 's.a.r.l.' -> 'sarl'
    s = _NONWORD.sub(" ", s)
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
