"""Markdown reports the organisers said they would compute themselves --
handed to them pre-computed: EDA, CV ablation, leave-one-country-out, blocking.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .io import read_ground_truth
from .pipeline import load_split


def _fmt_score(s: dict) -> str:
    return (f"{s['macro_f']:.5f} | {s['singleton_f']:.4f} | {s['matched_f']:.4f} | "
            f"{s['precision_micro']:.4f} | {s['recall_micro']:.4f}")


def eda(data: Path, reports: Path) -> dict:
    reports.mkdir(parents=True, exist_ok=True)
    s1, r = load_split(data, "train")
    truth = read_ground_truth(data / "train" / "train_ground_truth.tsv")
    owners = Counter(x for ms in truth.values() for x in ms)
    multi = sum(1 for c in owners.values() if c > 1)
    sizes = Counter(len(v) for v in truth.values())
    out = dict(
        n_s1=len(s1), n_r=len(r), n_true_pairs=sum(owners.values()),
        singleton_share=sizes.get(0, 0) / max(len(truth), 1),
        match_size_dist=dict(sorted(sizes.items())),
        records_with_multiple_owners=multi,
        exclusivity_holds=multi == 0,
        unmatched_records=len(r) - len(owners),
        country_labels_s1=dict(Counter(s1.country)), country_labels_r=dict(Counter(r.country)),
    )
    test = {}
    if (data / "test" / "test_source1.tsv").exists():
        t1, tr = load_split(data, "test")
        test = dict(n_s1=len(t1), n_r=len(tr), country_labels=dict(Counter(list(t1.country) + list(tr.country))))
    lines = [
        "# EDA", "",
        f"- Source 1 entities: **{out['n_s1']}**, Source 2+3 records: **{out['n_r']}**, true pairs: **{out['n_true_pairs']}**",
        f"- Singletons (no true match): **{out['singleton_share']:.1%}** — each is worth a full 1.0 for an empty prediction",
        f"- Matches per Source-1 entity: `{out['match_size_dist']}`",
        f"- Unmatched S2/S3 records (pure distractors): **{out['unmatched_records']}**",
        f"- Records claimed by more than one Source-1 entity: **{multi}** → "
        + ("exclusivity **holds**; the one-owner constraint is safe to apply." if multi == 0
           else "exclusivity is **violated**; the constraint must stay soft."),
        f"- Train country labels (S1): `{out['country_labels_s1']}`",
        f"- Train country labels (S2/S3): `{out['country_labels_r']}`",
    ]
    if test:
        lines += [f"- Test: S1 **{test['n_s1']}**, S2+S3 **{test['n_r']}**, labels `{test['country_labels']}`"]
    (reports / "eda.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return out


def cv_report(res: dict, reports: Path, name: str = "cv_report.md") -> None:
    b = res["blocking"]
    lines = [
        f"# Cross-validated results ({res['fold_mode']} folds)", "",
        "All numbers are **out-of-fold**: every prediction comes from a model that never saw that",
        "Source-1 entity. Folds are grouped by Source-1 entity.", "",
        "## Decision-layer ablation", "",
    ]
    loco = res.get("loco_variants")
    if loco:
        w = res["unseen_share"]
        lines += [
            f"Two OOF schemes: **CV** (folds by entity) and **LOCO** (train on one country group, predict the",
            f"other; thresholds re-used from CV so no held-out-country label is seen). The test file has",
            f"**w = {w:.1%}** of Source-1 entities in a country never seen in training, so the rule is chosen by",
            f"`(1 − w)·CV + w·LOCO` — the expected score on a test set with that mix.", "",
            "| decision rule | CV macro F0.5 | singleton F | has-match F | micro P | micro R | LOCO macro F0.5 | selection objective |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for k, v in res["variants"].items():
            star = " ⭐" if k == res["chosen"] else ""
            lines.append(f"| {k}{star} | {_fmt_score(v['score'])} | {loco[k]['score']['macro_f']:.5f} | "
                         f"**{res['selection'][k]:.5f}** |")
    else:
        lines += ["| decision rule | macro F0.5 | singleton F | has-match F | micro P | micro R |",
                  "|---|---|---|---|---|---|"]
        for k, v in res["variants"].items():
            star = " ⭐" if k == res["chosen"] else ""
            lines.append(f"| {k}{star} | {_fmt_score(v['score'])} |")
    lines += [
        "", f"Chosen for inference: **{res['chosen']}** (selected on out-of-fold data, not assumed).", "",
        "## Blocking and candidate set (train)", "",
        "| stage | pairs | recall of true pairs | pairs per S1 |", "|---|---|---|---|",
        f"| full cross product | {b['n_s1'] * b['n_r']:,} | 1.0000 | {b['n_r']:,} |",
        f"| fused retrieval pool | {b['pool_pairs']:,} | {b['pool_recall']:.4f} | {b['pool_pairs'] / b['n_s1']:.1f} |",
        f"| **candidate_pairs (after learned prune)** | **{b['cand_pairs']:,}** | **{b['cand_recall']:.4f}** | **{b['cand_per_s1']:.2f}** |",
        "", f"Reduction ratio vs cross product: **{b['reduction_ratio']:.6f}**. "
        f"Candidates per S1: `{b['cand_per_s1_dist']}`. Prune threshold τ = {b['prune_tau']}.", "",
        "### Prune curve (OOF stage-1 probability threshold)", "",
        "| τ | true pairs lost | candidate pairs |", "|---|---|---|",
    ]
    lines += [f"| {t} | {loss:.4%} | {n:,} |" for t, loss, n in b["prune_curve"]]
    lines += ["", "## Resolved country groups", "", f"`{res['groups']}`", "",
              "## Learned lexicon", "",
              f"- mined abbreviations: {res['lexicon']['n_abbrev']}",
              f"- mined legal/generic tail tokens: `{res['lexicon']['legal']}`",
              f"- landmark cues: `{res['lexicon']['cues']}`"]
    if res.get("importance"):
        lines += ["", "## Top matcher features (mean gain over the fold bag)", "", "| feature | gain |", "|---|---|"]
        lines += [f"| `{f}` | {g:,.0f} |" for f, g in res["importance"][:20]]
    lines += ["", f"Timings (s): `{res['timings']}`"]
    (reports / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def loco_report(res: dict, reports: Path) -> None:
    cv_report(res, reports, name="loco_report.md")
    p = reports / "loco_report.md"
    txt = p.read_text(encoding="utf-8").replace(
        "# Cross-validated results (country folds)",
        "# Leave-one-country-out validation\n\nEach country group is predicted by models trained **only on the other**"
        " — the proxy for the unseen-country (France) trap in test.\n\n> Note: here the threshold rules are"
        " tuned on the pooled out-of-fold scores, i.e. with the held-out country's labels — optimistic for them."
        " The honest comparison (thresholds carried over from entity-fold CV) is the LOCO column of `cv_report.md`.",
    )
    p.write_text(txt, encoding="utf-8")


def blocking_report(res: dict, art: Path, reports: Path) -> None:
    c = res["cand_per_s1"]
    lines = [
        f"# Blocking report — {res['split']}", "",
        f"- Source 1 entities: **{res['n_s1']:,}**; Source 2+3 records: **{res['n_r']:,}**",
        f"- Full cross product: **{res['n_s1'] * res['n_r']:,}** pairs",
        f"- Fused retrieval pool: **{res['pool_pairs']:,}** pairs ({res['pool_pairs'] / res['n_s1']:.1f} per S1)",
        f"- **candidate_pairs.tsv: {res['cand_pairs']:,} pairs — {c['mean']:.2f} per S1** "
        f"(median {c['median']:.0f}, p95 {c['p95']:.0f}, max {c['max']}, empty {c['zero_share']:.1%})",
        f"- Reduction ratio: **{1 - res['cand_pairs'] / (res['n_s1'] * res['n_r']):.6f}**",
        f"- Final matched pairs: **{res['matched_pairs']:,}**; entities predicted as singletons: **{res['empty_share']:.1%}**",
        f"- Resolved country groups: `{res['groups']}`",
        f"- Timings (s): `{res['timings']}`",
    ]
    if "holdout_score" in res:
        s = res["holdout_score"]
        lines += ["", "## Synthetic hold-out (test truth is known only for the synthetic harness)", "",
                  f"- pool recall **{res['holdout_pool_recall']:.4f}**, candidate recall **{res['holdout_cand_recall']:.4f}**",
                  f"- macro F0.5 **{s['macro_f']:.5f}** (singletons {s['singleton_f']:.4f}, has-match {s['matched_f']:.4f})"]
    train = art / "train_result.json"
    if train.exists():
        b = json.loads(train.read_text(encoding="utf-8"))["blocking"]
        lines += ["", "## Recall ceiling (measured on train, out-of-fold)", "",
                  f"- pool recall **{b['pool_recall']:.4f}** → candidate recall **{b['cand_recall']:.4f}** at "
                  f"**{b['cand_per_s1']:.2f}** candidates per S1"]
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "blocking_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
