"""Submission-format checker (stdlib only), mirroring every rule in the statement.

Used as a gate on every write.  If Amazon's `utils/validate_submission.py` is
present, the Makefile runs that as well -- theirs is authoritative.
"""
from __future__ import annotations

import csv
from pathlib import Path


def _read(path: Path, id_col: str, problems: list[str]) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(rd, None)
        if header != ["source1_entity_id", id_col]:
            problems.append(f"{path.name}: header {header} != ['source1_entity_id', '{id_col}']")
        for n, row in enumerate(rd, start=2):
            if len(row) not in (1, 2):
                problems.append(f"{path.name}:{n}: expected 2 tab-separated columns, got {len(row)}")
                continue
            s1 = row[0]
            ids = [x for x in (row[1].split(",") if len(row) == 2 and row[1] else [])]
            if s1 in rows:
                problems.append(f"{path.name}:{n}: duplicate source1_entity_id {s1}")
            if len(ids) != len(set(ids)):
                problems.append(f"{path.name}:{n}: duplicate IDs in list for {s1}")
            rows[s1] = ids
    return rows


def _ids(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(rd)
        return [row[0] for row in rd if row]


def validate(matching: Path, candidate: Path, test_dir: Path) -> list[str]:
    problems: list[str] = []
    for p in (matching, candidate):
        if not p.exists():
            problems.append(f"missing {p}")
    if problems:
        return problems
    split = "test"
    s1 = _ids(test_dir / f"{split}_source1.tsv")
    valid_r = set(_ids(test_dir / f"{split}_source2.tsv")) | set(_ids(test_dir / f"{split}_source3.tsv"))

    m = _read(matching, "matched_entity_ids", problems)
    c = _read(candidate, "candidate_entity_ids", problems)
    s1_set = set(s1)
    for name, rows in (("matching_results.tsv", m), ("candidate_pairs.tsv", c)):
        missing = s1_set - rows.keys()
        extra = rows.keys() - s1_set
        if missing:
            problems.append(f"{name}: {len(missing)} Source-1 entities missing (e.g. {sorted(missing)[:3]})")
        if extra:
            problems.append(f"{name}: {len(extra)} unknown source1_entity_id (e.g. {sorted(extra)[:3]})")
        bad = {x for ids in rows.values() for x in ids if x not in valid_r or not x.startswith(("S2-", "S3-"))}
        if bad:
            problems.append(f"{name}: {len(bad)} IDs not in test Source 2/3 (e.g. {sorted(bad)[:3]})")
    not_cand = sum(1 for s, ids in m.items() for x in ids if x not in set(c.get(s, ())))
    if not_cand:
        problems.append(f"{not_cand} matched IDs are not in candidate_pairs.tsv (pipeline bug)")
    return problems
