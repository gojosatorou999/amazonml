"""Raw records -> comparable fields, via a Lexicon learned from the data.

The Lexicon holds everything the pipeline *learns* about surface forms:

    abbrev     short -> long map, mined from training pairs      (normalize.abbrev)
    legal      legal/generic tail vocabulary, mined transductively (normalize.name)
    cues       landmark cue words, base prepositions + mined     (normalize.address)

It is fitted once on the training data (plus test *names* for the legal tail,
which is transductive use of provided data), serialised to JSON, and applied
identically to train and test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .normalize.abbrev import mine_abbreviations
from .normalize.address import BASE_LANDMARK_CUES, mine_landmark_cues, parse_address
from .normalize.name import mine_legal_tokens, strip_tail
from .normalize.text import tokens


@dataclass
class Lexicon:
    abbrev: dict[str, str]
    legal: frozenset[str]
    cues: frozenset[str]

    @classmethod
    def fit(
        cls,
        s1: pd.DataFrame,
        r: pd.DataFrame,
        truth: dict[str, set[str]],
        extra_names: list[str] | None = None,
    ) -> "Lexicon":
        s1i = s1.set_index("entity_id")
        ri = r.set_index("entity_id")
        name_pairs, addr_pairs = [], []
        for a, ms in truth.items():
            if a not in s1i.index:
                continue
            for b in ms:
                if b in ri.index:
                    name_pairs.append((s1i.at[a, "business_name"], ri.at[b, "business_name"]))
                    addr_pairs.append((s1i.at[a, "business_address"], ri.at[b, "business_address"]))
        abbrev = mine_abbreviations(name_pairs + addr_pairs)
        names = list(s1.business_name) + list(r.business_name) + list(extra_names or [])
        legal = mine_legal_tokens([[abbrev.get(t, t) for t in tokens(n)] for n in names])
        cues = mine_landmark_cues(addr_pairs)
        return cls(abbrev=abbrev, legal=legal, cues=cues)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"abbrev": self.abbrev, "legal": sorted(self.legal), "cues": sorted(self.cues)},
                indent=1,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(abbrev=d["abbrev"], legal=frozenset(d["legal"]), cues=frozenset(d["cues"]))

    @classmethod
    def empty(cls) -> "Lexicon":
        return cls(abbrev={}, legal=frozenset(), cues=BASE_LANDMARK_CUES)


def prepare(df: pd.DataFrame, lex: Lexicon) -> pd.DataFrame:
    """One row per record with every field the blocker and features need."""
    rows = []
    for eid, name, addr, country in df[["entity_id", "business_name", "business_address", "country"]].itertuples(
        index=False
    ):
        ntoks = [lex.abbrev.get(t, t) for t in tokens(name)]
        core, tail = strip_tail(ntoks, lex.legal)
        # identity tokens: every non-legal token, wherever it sits -- robust to
        # word-order transposition, which tail-stripping alone is not
        ident = tuple(t for t in ntoks if t not in lex.legal) or tuple(core)
        a = parse_address(addr, lex.abbrev, lex.cues)
        rows.append(
            (
                eid,
                int(eid[1]) if len(eid) > 1 and eid[1].isdigit() else 0,
                name,
                addr,
                country.strip(),
                tuple(ntoks),
                tuple(core),
                tuple(tail),
                " ".join(ntoks),
                " ".join(core),
                "".join(core),
                "".join(t[0] for t in core if t),
                a.house,
                a.postcode,
                a.numbers,
                a.street,
                a.tail,
                a.tokens,
                " ".join(a.tokens),
                a.landmark,
                ident,
                " ".join(sorted(ident)),
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "id", "src", "name_raw", "addr_raw", "country",
            "name_toks", "core", "legal_tail", "name_str", "core_str", "core_cat", "acr",
            "house", "postcode", "numbers", "street", "addr_tail", "addr_toks", "addr_str",
            "landmark", "ident", "ident_str",
        ],
    )
