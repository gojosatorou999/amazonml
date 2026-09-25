"""Synthetic stand-in for the challenge dataset.

The real TSVs are not available yet, so this generates train/test files with the
same schema and -- more importantly -- the same *noise taxonomy* the problem
statement enumerates:

  names      abbreviations, legal-suffix inconsistency, DBA/trade names,
             '&' vs 'and', word-order transposition, typos
  addresses  Rd/Road abbreviation, missing components (no PIN, no state),
             landmark references ('Near SBI ATM'), municipal numbering formats,
             component reordering
  structure  Source-1 is deduplicated; an S1 entity has zero, one or many
             matches; a third country (France) appears only in test

It exists so the entire pipeline can be built, debugged and scored before the
real data lands.  Swap the input path and nothing downstream changes.

    python -m ber.synth --out data --seed 0
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

# --- surface vocabularies (invented, not scraped -- see vault/Fair-Play-Boundary) ---

CORE = """apex vertex summit harbor ironwood bluepeak silverline northgate
stonebridge redwood clearwater goldleaf brightpath ravenwood sunstone maplecrest
quarry lantern meridian cobalt orchid thistle falcon juniper kestrel onyx""".split()

CORE2 = """trading solutions logistics textiles foods motors systems industries
enterprises consultancy packaging chemicals fabrics electricals services""".split()

LEGAL = {
    "US": [("Corporation", "Corp"), ("Incorporated", "Inc"),
           ("Limited Liability Company", "LLC"), ("Company", "Co")],
    "IN": [("Private Limited", "Pvt Ltd"), ("Limited", "Ltd"), ("Private Limited", "P Ltd")],
    "FR": [("Societe Anonyme", "SA"), ("Societe par Actions Simplifiee", "SAS"),
           ("Sarl", "S.A.R.L.")],
}
STREET = {
    "US": [("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave"),
           ("Boulevard", "Blvd"), ("Drive", "Dr")],
    "IN": [("Road", "Rd"), ("Marg", "Marg"), ("Cross", "Crs"),
           ("Main", "Mn"), ("Street", "St")],
    "FR": [("Rue", "R"), ("Avenue", "Av"), ("Boulevard", "Bd"), ("Place", "Pl")],
}
CITY = {
    "US": ["Austin", "Fresno", "Akron", "Tacoma", "Peoria", "Mobile", "Lubbock"],
    "IN": ["Pune", "Indore", "Kochi", "Surat", "Nagpur", "Jaipur", "Coimbatore"],
    "FR": ["Lyon", "Nantes", "Rennes", "Dijon", "Toulouse", "Reims", "Angers"],
}
REGION = {
    "US": ["TX", "CA", "OH", "WA", "IL", "AL"],
    "IN": ["Maharashtra", "MP", "Kerala", "Gujarat", "Rajasthan", "Tamil Nadu"],
    "FR": ["Rhone", "Loire-Atlantique", "Bretagne", "Occitanie", "Grand Est"],
}
LANDMARK = {
    "US": ["Near Exit 14", "Behind the Civic Center", "Opp. City Library"],
    "IN": ["Near SBI ATM", "Opp. Bus Depot", "Behind Ganesh Temple", "Near Metro Pillar 42"],
    "FR": ["Pres de la Gare", "Face a la Mairie", "A cote du Marche"],
}
COUNTRY_LABEL = {
    "US": ["US", "USA", "United States"],
    "IN": ["India", "IN"],
    "FR": ["France", "FR"],
}

LOCALITY_SUFFIX = {"US": "Park", "IN": "Nagar", "FR": "Quartier"}


def _typo(s: str, rng: random.Random) -> str:
    """One realistic edit: transpose, drop, or double a character."""
    if len(s) < 4:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.choice(("swap", "drop", "double"))
    if op == "swap":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == "drop":
        return s[:i] + s[i + 1:]
    return s[:i] + s[i] + s[i:]


@dataclass
class Record:
    entity_id: str
    name: str
    address: str
    country: str


class Generator:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    # ---------- canonical entity ----------

    def canonical(self, cc: str) -> tuple[str, dict]:
        r = self.rng
        core = f"{r.choice(CORE).capitalize()} {r.choice(CORE2).capitalize()}"
        if r.random() < 0.25:
            core = f"{r.choice(CORE).capitalize()} {core}"
        legal_long, legal_short = r.choice(LEGAL[cc])
        street_long, street_short = r.choice(STREET[cc])
        house = (str(r.randrange(1, 400)) if r.random() < 0.9
                 else f"{r.randrange(1, 40)}/{r.randrange(1, 90)}")
        spec = {
            "cc": cc,
            "core": core,
            "legal": (legal_long, legal_short),
            "house": house,
            "street_name": r.choice(CORE).capitalize(),
            "street_long": street_long,
            "street_short": street_short,
            "locality": f"{r.choice(CORE).capitalize()} {LOCALITY_SUFFIX[cc]}",
            "city": r.choice(CITY[cc]),
            "region": r.choice(REGION[cc]),
            "post": (str(r.randrange(100000, 999999)) if cc == "IN"
                     else str(r.randrange(10000, 99999))),
            "landmark": r.choice(LANDMARK[cc]),
        }
        return f"{core} {legal_long}", spec

    # ---------- noisy views of that entity ----------

    def variant_name(self, spec: dict) -> str:
        r = self.rng
        legal_long, legal_short = spec["legal"]
        toks = spec["core"].split()
        if r.random() < 0.15 and len(toks) > 1:            # word-order transposition
            r.shuffle(toks)
        if r.random() < 0.20 and len(toks) > 1:            # DBA / trade-name truncation
            toks = toks[:-1]
        name = " ".join(toks)
        if r.random() < 0.15:                              # & vs and
            name = name + (" and Sons" if r.random() < 0.5 else " & Sons")
        p = r.random()
        if p < 0.45:
            name = f"{name} {legal_short}"
        elif p < 0.80:
            name = f"{name} {legal_long}"
        if r.random() < 0.25:                              # typo
            name = _typo(name, r)
        if r.random() < 0.10:                              # punctuation noise
            name = name.replace(" ", ", ", 1)
        return name.strip()

    def variant_address(self, spec: dict) -> str:
        r = self.rng
        st = spec["street_short"] if r.random() < 0.5 else spec["street_long"]
        parts = [f"{spec['house']} {spec['street_name']} {st}"]
        if r.random() < 0.75:
            parts.append(spec["locality"])
        if r.random() < 0.20:                              # landmark instead of structure
            parts.insert(r.randrange(len(parts) + 1), spec["landmark"])
        parts.append(spec["city"])
        if r.random() < 0.60:                              # region often missing
            parts.append(spec["region"])
        if r.random() < 0.65:                              # PIN often missing
            parts.append(spec["post"])
        if r.random() < 0.10:                              # component reordering
            parts.append(parts.pop(r.randrange(len(parts))))
        out = (", " if r.random() < 0.8 else " ").join(parts)
        if r.random() < 0.15:
            out = _typo(out, r)
        return out

    # ---------- dataset assembly ----------

    def build(self, n_entities: int, countries: dict[str, float]):
        r = self.rng
        s1: list[Record] = []
        s2: list[Record] = []
        s3: list[Record] = []
        gt: dict[str, list[str]] = {}
        ccs, weights = list(countries), list(countries.values())

        for i in range(n_entities):
            cc = r.choices(ccs, weights)[0]
            canon_name, spec = self.canonical(cc)
            s1_id = f"S1-{i:06d}"
            s1.append(Record(s1_id, canon_name, self.variant_address(spec),
                             r.choice(COUNTRY_LABEL[cc])))

            matches: list[str] = []
            # zero / one / many matches per source, mirroring real ER skew
            for bucket, pref in ((s2, "S2"), (s3, "S3")):
                k = r.choices((0, 1, 2, 3), (0.34, 0.50, 0.12, 0.04))[0]
                for _ in range(k):
                    eid = f"{pref}-{len(bucket):06d}"
                    bucket.append(Record(eid, self.variant_name(spec),
                                         self.variant_address(spec),
                                         r.choice(COUNTRY_LABEL[cc])))
                    matches.append(eid)
            gt[s1_id] = matches

        # distractors: unmatched records so blocking has to earn its keep, including
        # near-miss names (the chain / franchise false-merge failure mode)
        for pref, bucket in (("S2", s2), ("S3", s3)):
            for _ in range(int(0.35 * len(bucket))):
                cc = r.choices(ccs, weights)[0]
                _, spec = self.canonical(cc)
                bucket.append(Record(f"{pref}-{len(bucket):06d}", self.variant_name(spec),
                                     self.variant_address(spec),
                                     r.choice(COUNTRY_LABEL[cc])))
        r.shuffle(s2)
        r.shuffle(s3)
        return s1, s2, s3, gt


def _write_source(path: Path, recs: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for rec in recs:
            fh.write(f"{rec.entity_id}\t{rec.name}\t{rec.address}\t{rec.country}\n")


def _write_truth(path: Path, order: list[Record], gt: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for rec in order:
            fh.write(f"{rec.entity_id}\t{','.join(gt[rec.entity_id])}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-entities", type=int, default=8000)
    ap.add_argument("--test-entities", type=int, default=3000)
    args = ap.parse_args()
    out = Path(args.out)

    g = Generator(args.seed)
    s1, s2, s3, gt = g.build(args.train_entities, {"US": 0.5, "IN": 0.5})
    _write_source(out / "train" / "train_source1.tsv", s1)
    _write_source(out / "train" / "train_source2.tsv", s2)
    _write_source(out / "train" / "train_source3.tsv", s3)
    _write_truth(out / "train" / "train_ground_truth.tsv", s1, gt)

    # the test set carries the unseen third country, exactly like the real one
    g2 = Generator(args.seed + 1000)
    t1, t2, t3, tgt = g2.build(args.test_entities, {"US": 0.40, "IN": 0.40, "FR": 0.20})
    _write_source(out / "test" / "test_source1.tsv", t1)
    _write_source(out / "test" / "test_source2.tsv", t2)
    _write_source(out / "test" / "test_source3.tsv", t3)
    # held back for our own scoring only -- never fed to the pipeline
    _write_truth(out / "test" / "_synthetic_truth.tsv", t1, tgt)

    singles = sum(1 for v in gt.values() if not v)
    print(f"train: S1={len(s1)} S2={len(s2)} S3={len(s3)}  "
          f"singletons={singles} ({singles / len(s1):.1%})")
    print(f"test : S1={len(t1)} S2={len(t2)} S3={len(t3)}  (US/IN/FR)")
    print(f"wrote -> {out.resolve()}")


if __name__ == "__main__":
    main()
