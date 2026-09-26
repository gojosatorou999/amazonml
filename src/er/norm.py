"""Vectorised text normalisation (polars expressions, run in Rust)."""
from __future__ import annotations

import polars as pl

# Legal forms, honorifics and web/format noise. These carry no identity: two different businesses
# routinely share them, and the noise process adds, drops, abbreviates and moves them freely.
LEGAL = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "pvt",
    "private", "plc", "lp", "llp", "pllc", "pc", "pa", "sarl", "sas", "sasu", "sa", "eurl", "sci",
    "snc", "selarl", "gmbh", "opc", "the", "and", "of", "m", "s", "dba", "doing", "business", "as",
    "www", "com", "net", "org", "in", "fr", "dr", "mr", "mrs", "ms", "shri", "sri", "smt", "et",
    "de", "du", "des", "la", "le", "les", "l", "d",
}

# Address tokens that carry no locating power on their own.
ADDR_STOP = {
    "unit", "apt", "apartment", "suite", "ste", "no", "number", "h", "hno", "door", "po", "box",
    "near", "nr", "opp", "opposite", "behind", "floor", "flr", "ground", "first", "second", "n", "a",
    "null", "na", "nan", "none", "c", "o", "the", "and", "of", "at", "de", "du", "des", "la", "le",
}

# Street-type spellings -> one canonical form, so "Rd" and "Road" agree as blocking keys.
STREET = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive",
    "ln": "lane", "ct": "court", "blvd": "boulevard", "bd": "boulevard", "pl": "place", "pkwy": "parkway",
    "hwy": "highway", "cir": "circle", "trl": "trail", "ter": "terrace", "sq": "square", "pt": "point",
    "mt": "mount", "ft": "fort", "cres": "crescent", "hts": "heights", "rte": "route", "expy": "expressway",
    "fwy": "freeway", "tpke": "turnpike", "aly": "alley", "xing": "crossing", "cv": "cove", "pky": "parkway",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "r": "rue", "bis": "bis", "nagar": "nagar", "marg": "marg",
}


def fold(e: pl.Expr) -> pl.Expr:
    """Unicode fold: NFKD, strip combining marks, lowercase, punctuation -> space."""
    return (
        e.fill_null("")
        .str.normalize("NFKD")
        .str.replace_all(r"\p{M}", "")
        .str.to_lowercase()
        .str.replace_all(r"['’`]", "")
        .str.replace_all(r"[&+]", " and ")
        .str.replace_all(r"[^\p{L}\p{N}]+", " ")
        .str.strip_chars()
    )


def tokens(e: pl.Expr) -> pl.Expr:
    return e.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
