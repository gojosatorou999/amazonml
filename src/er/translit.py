"""Brahmic-script -> Latin transliteration built from Unicode character names (stdlib only).

No transliteration table is shipped: each code point's Unicode name ("DEVANAGARI LETTER KHA",
"BENGALI VOWEL SIGN II", "TELUGU SIGN VIRAMA") already spells its sound. Consonants carry an
inherent 'a' that a following vowel sign replaces and a virama deletes; word-final inherent 'a'
is dropped (schwa deletion), which is how these names are romanised in the Latin sources.

The output is then only ever compared through `skeleton()`, a phonetic key that forgives the
remaining vowel / aspiration / retroflex differences between an Indic spelling and an English one.
"""
from __future__ import annotations

import re
import unicodedata

import polars as pl

_BLOCKS = [(0x0900, 0x0DFF)]  # Devanagari .. Malayalam (Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada)
_VOWELS = {"A", "AA", "I", "II", "U", "UU", "E", "EE", "AI", "O", "OO", "AU", "VOCALIC R", "VOCALIC RR",
           "VOCALIC L", "VOCALIC LL", "SHORT E", "SHORT O", "CANDRA E", "CANDRA O", "CANDRA A"}
_SIMPLE = {"AA": "a", "II": "i", "UU": "u", "EE": "e", "OO": "o", "VOCALIC R": "ri", "VOCALIC RR": "ri",
           "VOCALIC L": "li", "VOCALIC LL": "li", "SHORT E": "e", "SHORT O": "o", "CANDRA E": "e",
           "CANDRA O": "o", "CANDRA A": "a"}
_CONS_FIX = {"TT": "t", "TTH": "th", "DD": "d", "DDH": "dh", "NN": "n", "SS": "sh", "SH": "sh", "LL": "l",
             "LLL": "l", "RR": "r", "RRR": "r", "NNN": "n", "NY": "ny", "NG": "ng", "JNY": "gy", "YY": "y",
             "KSSA": "ksh", "V": "v", "W": "v"}

# Private-use markers: DEL removes the preceding inherent 'a'.
DEL = "\x01"


def _table() -> dict[str, str]:
    out: dict[str, str] = {}
    for lo, hi in _BLOCKS:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            name = unicodedata.name(ch, "")
            if not name:
                continue
            parts = name.split(" ", 1)
            if len(parts) < 2:
                continue
            rest = parts[1]
            if rest.startswith("DIGIT "):
                out[ch] = str(unicodedata.digit(ch, 0))
            elif rest.startswith("LETTER "):
                s = rest[len("LETTER "):]
                if s in _VOWELS:
                    out[ch] = _SIMPLE.get(s, s.lower())
                elif s.endswith("A") and len(s) >= 2:
                    base = s[:-1]
                    out[ch] = _CONS_FIX.get(base, base.lower()) + "a"
            elif rest.startswith("VOWEL SIGN "):
                s = rest[len("VOWEL SIGN "):]
                out[ch] = DEL + _SIMPLE.get(s, s.lower())
            elif rest in ("SIGN VIRAMA", "SIGN HALANT") or rest.endswith("VIRAMA"):
                out[ch] = DEL
            elif rest in ("SIGN ANUSVARA", "SIGN CANDRABINDU"):
                out[ch] = "n"
            elif rest == "SIGN VISARGA":
                out[ch] = "h"
            elif rest in ("SIGN NUKTA", "AU LENGTH MARK", "SIGN AVAGRAHA"):
                out[ch] = ""
            elif rest.startswith("LETTER") is False and "SIGN" in rest:
                out[ch] = ""
    return out


TABLE = _table()
_KEYS = list(TABLE.keys())
_VALS = [TABLE[k] for k in _KEYS]


def has_indic(e: pl.Expr) -> pl.Expr:
    return e.str.contains(r"[\x{0900}-\x{0DFF}]")


def translit(e: pl.Expr) -> pl.Expr:
    """Romanise any Brahmic characters in a string column; Latin text passes through unchanged."""
    return (
        e.str.replace_many(_KEYS, _VALS)
        .str.replace_all("a" + DEL, "")
        .str.replace_all(DEL, "")
        .str.replace_all(r"([bcdfghjklmnpqrstvwxyz])a\b", "$1")  # schwa deletion at word end
    )


def translit_py(s: str) -> str:
    for k, v in TABLE.items():
        s = s.replace(k, v)
    s = s.replace("a" + DEL, "").replace(DEL, "")
    return re.sub(r"([bcdfghjklmnpqrstvwxyz])a\b", r"\1", s)


def skeleton(e: pl.Expr) -> pl.Expr:
    """Phonetic key of a folded Latin token string: consonant classes, vowels dropped after the first letter."""
    e = (
        e.str.replace_all(r"ph", "f")
        .str.replace_all(r"(ck|q|c)", "k")
        .str.replace_all(r"[vw]", "w")
        .str.replace_all(r"z", "s")
        .str.replace_all(r"([a-z])h", "$1")   # aspiration / digraph h
        .str.replace_all(r"\b[aeiouy]+", "a")  # a leading vowel survives as 'a'
        .str.replace_all(r"([a-z])[aeiouy]+", "$1")
    )
    for c in "bdfgjklmnprstwx":
        e = e.str.replace_all(c + c + "+", c)
    return e
