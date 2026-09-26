"""TSV I/O for the challenge files. Tab separator is always explicit."""
from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: str | Path) -> "pd.DataFrame":
    """Read a *_sourceN.tsv. Everything stays str; nulls become empty strings."""
    import csv as _csv

    import pandas as pd  # lazy: ground-truth + submission I/O stay stdlib-only

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,     # 'NA' is a legitimate token in names/addresses
        quoting=_csv.QUOTE_NONE,   # addresses contain quotes; never let csv eat them
        encoding="utf-8",
    )
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing} (got {list(df.columns)})")
    df = df[SOURCE_COLUMNS].fillna("")
    for c in SOURCE_COLUMNS:
        df[c] = df[c].str.strip()
    return df.reset_index(drop=True)


def read_ground_truth(path: str | Path) -> dict[str, set[str]]:
    """train_ground_truth.tsv -> {source1_entity_id: {matched ids}}."""
    out: dict[str, set[str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        if header[:2] != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"{path}: unexpected header {header}")
        for row in reader:
            if not row:
                continue
            s1 = row[0].strip()
            raw = row[1].strip() if len(row) > 1 else ""
            out[s1] = {t.strip() for t in raw.split(",") if t.strip()}
    return out


def write_pairs(
    path: str | Path,
    rows: Mapping[str, Iterable[str]],
    s1_order: Iterable[str],
    id_column: str,
) -> None:
    """Write matching_results.tsv / candidate_pairs.tsv.

    Emits exactly one row per Source-1 entity in `s1_order`, deduplicates each
    ID list while preserving order, and never quotes -- the validator expects
    bare comma-joined IDs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{id_column}\n")
        for s1 in s1_order:
            seen: dict[str, None] = {}
            for eid in rows.get(s1, ()):
                seen.setdefault(eid, None)
            fh.write(f"{s1}\t{','.join(seen)}\n")
