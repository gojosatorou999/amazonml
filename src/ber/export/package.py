"""Build <team>_submission.zip exactly as the statement specifies:

    <team>_submission.zip
    ├── output/{matching_results.tsv, candidate_pairs.tsv}
    ├── code/business_entity_resolution/{src/, README.md, requirements.txt, Makefile}
    └── Documentation_template.md

    python -m ber.export.package --team <team_name>
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--doc", default="docs/Documentation_template.md")
    a = ap.parse_args(argv)

    out = ROOT / f"{a.team}_submission.zip"
    code = "code/business_entity_resolution"
    required = [ROOT / "output" / "matching_results.tsv", ROOT / "output" / "candidate_pairs.tsv", ROOT / a.doc]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing} -- run `make all` first")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in required[:2]:
            z.write(p, f"output/{p.name}")
        z.write(ROOT / a.doc, "Documentation_template.md")
        for name in ("README.md", "requirements.txt", "Makefile"):
            z.write(ROOT / name, f"{code}/{name}")
        for p in sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "src").rglob("*.html"), *(ROOT / "tests").rglob("*.py")]):
            z.write(p, f"{code}/{p.relative_to(ROOT).as_posix()}")
        # review console source (not needed to reproduce the submission; node_modules/dist excluded)
        app = ROOT / "app"
        for p in sorted([*app.glob("*.json"), *app.glob("*.ts"), app / "index.html", *(app / "src").rglob("*")]):
            if p.is_file():
                z.write(p, f"{code}/{p.relative_to(ROOT).as_posix()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
