"""End-to-end pipeline: train (cross-fitted) and predict.

    prepare -> block (pool) -> pair features -> PRUNER (stage 1)
            -> adaptive cut  == candidate_pairs.tsv
            -> + context features -> MATCHER (stage 2) -> isotonic calibration
            -> exclusivity -> expected-F0.5 set selection == matching_results.tsv

Every threshold and choice made on the training side is made on out-of-fold
predictions, grouped by Source-1 entity.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking.retrieve import retrieve
from .decide.assign import best_threshold, decide, exclusive
from .eval.metric import score
from .features.context import build_context_features
from .features.pairwise import build_pair_features
from .io import read_ground_truth, read_source, write_pairs
from .models.gbm import crossfit_isotonic, entity_folds, fit_oof, predict_bag
from .normalize.country import resolve_country_groups
from .prepare import Lexicon, prepare


@dataclass
class Timer:
    marks: dict = field(default_factory=dict)
    _t: float = field(default_factory=time.perf_counter)

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.marks[name] = round(now - self._t, 2)
        self._t = now
        print(f"  [{name}] {self.marks[name]}s", flush=True)


def load_split(root: Path, split: str):
    s1 = read_source(root / split / f"{split}_source1.tsv")
    r = pd.concat(
        [read_source(root / split / f"{split}_source2.tsv"), read_source(root / split / f"{split}_source3.tsv")],
        ignore_index=True,
    )
    return s1, r


def block(P1: pd.DataFrame, PR: pd.DataFrame, k: int = 20, pool_size: int = 30):
    pool, diag = retrieve(P1, PR, k=k, pool=pool_size)
    bi, bs = diag["best_fwd"]
    groups = resolve_country_groups(
        P1.country.tolist(), PR.country.values[bi].tolist(), bs,
        P1.country.tolist() + PR.country.tolist(),
    )
    return pool, groups


def label(pool, P1, PR, truth) -> np.ndarray:
    pos = {(a, b) for a, ms in truth.items() for b in ms}
    return np.fromiter(
        ((a, b) in pos for a, b in zip(P1.id.values[pool.i.values], PR.id.values[pool.j.values])),
        dtype=np.int8, count=len(pool),
    )


def choose_prune_threshold(p1, y, max_loss: float, grid=None):
    """Largest tau losing at most `max_loss` of the pool's true pairs."""
    grid = grid if grid is not None else [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2]
    npos = max(int(y.sum()), 1)
    rows, best = [], grid[0]
    for t in grid:
        keep = p1 >= t
        loss = 1 - y[keep].sum() / npos
        rows.append((t, float(loss), int(keep.sum())))
        if loss <= max_loss:
            best = t
    return best, rows


def _fold_ids(P1, pool, mode, groups, n_folds):
    if mode == "country":
        g = P1.country.map(lambda c: groups.get(c, c)).values
        codes = {v: k for k, v in enumerate(sorted(set(g)))}
        return np.array([codes[g[i]] for i in pool.i.values]), {v: k for v, k in codes.items()}
    ent = entity_folds(np.arange(len(P1)), n_folds=n_folds)
    return ent[pool.i.values], None


def train(
    data_root: str | Path = "data",
    art: str | Path = "artifacts",
    reports: str | Path = "reports",
    n_folds: int = 5,
    fold_mode: str = "entity",
    max_prune_loss: float = 0.003,
    save: bool = True,
    robust: bool = True,
) -> dict:
    data_root, art, reports = Path(data_root), Path(art), Path(reports)
    art.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    T = Timer()

    s1, r = load_split(data_root, "train")
    truth = read_ground_truth(data_root / "train" / "train_ground_truth.tsv")
    test_names, test_labels = [], []
    if (data_root / "test" / "test_source1.tsv").exists():
        ts1, tr = load_split(data_root, "test")
        test_names = list(ts1.business_name) + list(tr.business_name)
        test_labels = list(ts1.country)
    lex = Lexicon.fit(s1, r, truth, extra_names=test_names)
    P1, PR = prepare(s1, lex), prepare(r, lex)
    T.lap("prepare")

    pool, groups = block(P1, PR)
    y = label(pool, P1, PR, truth)
    T.lap("block")

    X1 = build_pair_features(pool, P1, PR, groups)
    T.lap("pair_features")

    all_s1 = P1.id.tolist()
    tr_truth = {s: truth.get(s, set()) for s in all_s1}

    fold_of, _ = _fold_ids(P1, pool, fold_mode, groups, n_folds)
    main = _fit_eval(X1, y, pool, P1, PR, fold_of, tr_truth, all_s1, max_prune_loss=max_prune_loss)
    T.lap("fit+evaluate")
    keep, tau, cand = main["keep"], main["tau"], main["cand"]

    # ---- robustness to an unseen country -----------------------------------
    # Re-fit with folds = country groups (train on one, predict the other),
    # re-using the thresholds tuned above so no target-country label leaks in.
    # w = share of test Source-1 entities whose country label never occurs in
    # train -- measured from the test file, not assumed.
    loco, w = None, 0.0
    n_groups = len(set(groups.get(c, c) for c in P1.country))
    if robust and fold_mode == "entity" and n_groups > 1:
        c_fold, _ = _fold_ids(P1, pool, "country", groups, n_folds)
        loco = _fit_eval(X1, y, pool, P1, PR, c_fold, tr_truth, all_s1, tau=tau, thresholds=main["thresholds"])
        if test_labels:
            seen = set(P1.country) | set(PR.country)
            w = float(np.mean([c not in seen for c in test_labels]))
        T.lap("loco")

    def objective(name):
        f = main["variants"][name][0].macro_f
        return f if loco is None else (1 - w) * f + w * loco["variants"][name][0].macro_f

    best_name = max(main["variants"], key=objective)

    # blocking quality on train (the ceiling every later stage lives under)
    n_pos = sum(len(v) for v in truth.values())
    blk = dict(
        n_s1=len(P1), n_r=len(PR), n_true_pairs=n_pos,
        pool_pairs=len(pool), pool_recall=float(y.sum() / max(n_pos, 1)),
        cand_pairs=int(keep.sum()), cand_recall=float(y[keep].sum() / max(n_pos, 1)),
        cand_per_s1=float(keep.sum() / len(P1)),
        cand_per_s1_dist=_dist(cand.groupby("i").size().reindex(range(len(P1)), fill_value=0).values),
        reduction_ratio=float(1 - keep.sum() / (len(P1) * len(PR))),
        prune_tau=tau, prune_curve=main["prune_curve"],
    )
    result = dict(
        fold_mode=fold_mode, timings=T.marks, blocking=blk, chosen=best_name,
        variants={k: dict(score=vars(v[0]), cfg=v[1]) for k, v in main["variants"].items()},
        loco_variants=None if loco is None else {k: dict(score=vars(v[0])) for k, v in loco["variants"].items()},
        unseen_share=w, selection={k: objective(k) for k in main["variants"]},
        groups=groups, n_features=main["X2"].shape[1],
        train_labels=sorted(set(P1.country) | set(PR.country)),
        lexicon=dict(n_abbrev=len(lex.abbrev), legal=sorted(lex.legal), cues=sorted(lex.cues)),
    )
    # raw OOF arrays for offline analysis (never serialised to JSON)
    result["_oof"] = main["oof"]

    if save:
        lex.save(art / "lexicon.json")
        with open(art / "models.pkl", "wb") as fh:
            pickle.dump(
                dict(pruners=main["pruners"], matchers=main["matchers"], iso=main["iso"], tau=tau,
                     x1_cols=list(X1.columns), x2_cols=list(main["X2"].columns),
                     decision=main["variants"][best_name][1]),
                fh,
            )
        # feature importance (gain) of the matcher bag, for the methodology doc
        imp = np.mean([m.feature_importance("gain") for m in main["matchers"]], axis=0)
        result["importance"] = sorted(zip(main["X2"].columns, map(float, imp)), key=lambda t: -t[1])[:30]
        (art / "train_result.json").write_text(
            json.dumps({k: v for k, v in result.items() if not k.startswith("_")}, indent=1, default=str),
            encoding="utf-8")
    return result


def _fit_eval(X1, y, pool, P1, PR, fold_of, tr_truth, all_s1, *, tau=None, thresholds=None, max_prune_loss=0.003):
    """Cross-fit pruner + matcher on `fold_of`, then score every decision rule on OOF.

    `tau` / `thresholds` given => re-use them (the leave-one-country-out run must
    not tune anything on the held-out country).
    """
    p1, pruners = fit_oof(X1.values, y, fold_of, rounds=300)
    prune_curve = None
    if tau is None:
        tau, prune_curve = choose_prune_threshold(p1, y, max_prune_loss)
    keep = p1 >= tau
    ctx = build_context_features(pool, p1, PR)
    X2 = pd.concat([X1, ctx], axis=1)
    raw, matchers = fit_oof(X2.values[keep], y[keep], fold_of[keep], rounds=500)
    cal, iso = crossfit_isotonic(raw, y[keep], fold_of[keep])

    cand = pool[keep]
    S = P1.id.values[cand.i.values]
    R = PR.id.values[cand.j.values]
    J = cand.j.values

    def ev(pred):
        return score({s: pred.get(s, []) for s in all_s1}, tr_truth)

    thresholds = dict(thresholds or {})
    if "raw" not in thresholds:
        thresholds["raw"] = best_threshold(S, R, raw, tr_truth, all_s1)[1]
        thresholds["raw_excl"] = best_threshold(S, R, exclusive(J, raw), tr_truth, all_s1)[1]
    v = {}
    v["global threshold (tuned)"] = (ev(decide(S, R, raw, "threshold", thresholds["raw"])),
                                     dict(mode="threshold", excl=False, thr=thresholds["raw"], cal=False))
    v["global threshold + exclusivity"] = (ev(decide(S, R, exclusive(J, raw), "threshold", thresholds["raw_excl"])),
                                           dict(mode="threshold", excl=True, thr=thresholds["raw_excl"], cal=False))
    v["expected-F0.5, isotonic p"] = (ev(decide(S, R, cal)), dict(mode="expected_f", excl=False, cal=True))
    v["expected-F0.5, isotonic p + exclusivity"] = (ev(decide(S, R, exclusive(J, cal))), dict(mode="expected_f", excl=True, cal=True))
    v["expected-F0.5, model p"] = (ev(decide(S, R, raw)), dict(mode="expected_f", excl=False, cal=False))
    v["expected-F0.5, model p + exclusivity"] = (ev(decide(S, R, exclusive(J, raw))), dict(mode="expected_f", excl=True, cal=False))
    return dict(variants=v, thresholds=thresholds, tau=tau, keep=keep, cand=cand, prune_curve=prune_curve,
                pruners=pruners, matchers=matchers, iso=iso, X2=X2,
                oof=dict(s_ids=S, r_ids=R, j=J, raw=raw, cal=cal, all_s1=all_s1, truth=tr_truth))


def _dist(x: np.ndarray) -> dict:
    return dict(mean=float(x.mean()), median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                max=int(x.max()), zero_share=float((x == 0).mean()))


def predict(
    data_root: str | Path = "data",
    art: str | Path = "artifacts",
    out: str | Path = "output",
    split: str = "test",
) -> dict:
    data_root, art, out = Path(data_root), Path(art), Path(out)
    T = Timer()
    lex = Lexicon.load(art / "lexicon.json")
    with open(art / "models.pkl", "rb") as fh:
        M = pickle.load(fh)
    s1, r = load_split(data_root, split)
    P1, PR = prepare(s1, lex), prepare(r, lex)
    T.lap("prepare")
    pool, groups = block(P1, PR)
    T.lap("block")
    X1 = build_pair_features(pool, P1, PR, groups)[M["x1_cols"]]
    T.lap("pair_features")
    p1 = predict_bag(M["pruners"], X1.values)
    keep = p1 >= M["tau"]
    ctx = build_context_features(pool, p1, PR)
    X2 = pd.concat([X1, ctx], axis=1)[M["x2_cols"]]
    cand = pool[keep]
    raw = predict_bag(M["matchers"], X2.values[keep])
    cfg = M["decision"]
    p = M["iso"].predict(raw) if cfg.get("cal") else raw
    if cfg.get("excl"):
        p = exclusive(cand.j.values, p)
    s_ids = P1.id.values[cand.i.values]
    r_ids = PR.id.values[cand.j.values]
    pred = decide(s_ids, r_ids, p, cfg["mode"], cfg.get("thr", 0.5))
    T.lap("score+decide")

    cands: dict[str, list[str]] = {}
    order = np.argsort(-p1[keep], kind="stable")
    for s, rid in zip(s_ids[order], r_ids[order]):
        cands.setdefault(s, []).append(rid)
    out.mkdir(parents=True, exist_ok=True)
    write_pairs(out / "candidate_pairs.tsv", cands, P1.id.tolist(), "candidate_entity_ids")
    write_pairs(out / "matching_results.tsv", pred, P1.id.tolist(), "matched_entity_ids")

    # pair-level scores + feature rows for the review UI / explainability
    scored = pd.DataFrame({"s1": s_ids, "r": r_ids, "p_stage1": p1[keep], "p_match": p,
                           "emitted": [rid in set(pred.get(s, ())) for s, rid in zip(s_ids, r_ids)]})
    scored.to_csv(art / f"{split}_scored_pairs.tsv", sep="\t", index=False)
    with open(art / f"{split}_review.pkl", "wb") as fh:
        pickle.dump(dict(scored=scored.assign(p_model=raw), X=X2[keep].reset_index(drop=True),
                         s1=s1, r=r, groups=groups, pool=pool.assign(p1=p1)[["i", "j", "p1"]]), fh)

    counts = cand.groupby("i").size().reindex(range(len(P1)), fill_value=0).values
    res = dict(
        split=split, timings=T.marks, groups=groups, n_s1=len(P1), n_r=len(PR),
        pool_pairs=len(pool), cand_pairs=int(keep.sum()), cand_per_s1=_dist(counts),
        matched_pairs=int(sum(len(v) for v in pred.values())),
        empty_share=float(np.mean([len(pred.get(s, [])) == 0 for s in P1.id])),
    )
    truth_path = data_root / split / "_synthetic_truth.tsv"
    if truth_path.exists():  # only for the synthetic harness
        truth = read_ground_truth(truth_path)
        res["holdout_score"] = vars(score({s: pred.get(s, []) for s in P1.id}, truth))
        pos = {(a, b) for a, ms in truth.items() for b in ms}
        res["holdout_cand_recall"] = len(pos & set(zip(s_ids, r_ids))) / max(len(pos), 1)
        res["holdout_pool_recall"] = len(pos & set(zip(P1.id.values[pool.i.values], PR.id.values[pool.j.values]))) / max(len(pos), 1)
    (art / f"{split}_result.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res
