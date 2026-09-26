"""Command-line entry point:  python -m ber.cli <command>

    eda       profile the data; verify the one-owner-per-record assumption
    train     cross-fitted training; writes artifacts/ and reports/cv_report.md
    predict   test inference; writes output/*.tsv and reports/blocking_report.md
    loco      leave-one-country-out validation (the unseen-country proxy)
    validate  check output/ against every submission rule
    all       eda -> train -> predict -> validate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ber")
    ap.add_argument("command", choices=["eda", "train", "predict", "loco", "validate", "all"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="output")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    from . import pipeline, reporting

    cmds = ["eda", "train", "predict", "validate"] if a.command == "all" else [a.command]
    rc = 0
    for c in cmds:
        print(f"== {c}", flush=True)
        if c == "eda":
            reporting.eda(Path(a.data), Path(a.reports))
        elif c == "train":
            res = pipeline.train(a.data, a.artifacts, a.reports, n_folds=a.folds)
            reporting.cv_report(res, Path(a.reports))
            print(json.dumps({k: v["score"]["macro_f"] for k, v in res["variants"].items()}, indent=1))
        elif c == "predict":
            res = pipeline.predict(a.data, a.artifacts, a.out)
            reporting.blocking_report(res, Path(a.artifacts), Path(a.reports))
            print(json.dumps({k: v for k, v in res.items() if k != "groups"}, indent=1, default=str))
        elif c == "loco":
            res = pipeline.train(a.data, Path(a.artifacts) / "loco", a.reports, fold_mode="country", save=False, robust=False)
            reporting.loco_report(res, Path(a.reports))
            print(json.dumps({k: v["score"]["macro_f"] for k, v in res["variants"].items()}, indent=1))
        elif c == "validate":
            from .eval.validate import validate

            problems = validate(Path(a.out) / "matching_results.tsv", Path(a.out) / "candidate_pairs.tsv", Path(a.data) / "test")
            if problems:
                print("FAIL"); [print(f"  {i+1}. {p}") for i, p in enumerate(problems)]
                rc = 1
            else:
                print("PASS")
    return rc


if __name__ == "__main__":
    sys.exit(main())
