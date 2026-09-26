"""Live resolution of a record typed into the app -- the real pipeline, not a mock.

A query is treated as a new Source-1 entity and pushed through exactly the
stages the batch used:

    prepare (same Lexicon) -> RetrievalIndex.query (same fitted channels, exact
    reverse ranks) -> pair features (IDF over batch + query) -> pruner bag ->
    context features computed *together with the batch pool*, so reverse
    competition sees every Source-1 entity already claiming a record ->
    matcher bag -> the decision rule chosen at training time.

Building the index costs one blocking pass, so it is done once, in the
background, when the server starts.
"""
from __future__ import annotations

import pickle
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from ..blocking.retrieve import RetrievalIndex
from ..decide.assign import exclusive
from ..decide.expected_f import choose_set, expected_f_curve
from ..features.context import build_context_features
from ..features.pairwise import build_pair_features
from ..models.gbm import predict_bag
from ..prepare import Lexicon, prepare

QUERY_ID = "S1-QUERY"


class Resolver:
    def __init__(self, art: Path, review: dict, models: dict):
        self.art, self.review, self.M = art, review, models
        self.ready = threading.Event()
        self.error: str | None = None
        threading.Thread(target=self._build, daemon=True).start()

    def _build(self) -> None:
        try:
            self.lex = Lexicon.load(self.art / "lexicon.json")
            self.P1 = prepare(self.review["s1"], self.lex)
            self.PR = prepare(self.review["r"], self.lex)
            self.index = RetrievalIndex(self.P1, self.PR)
            self.groups = self.review.get("groups", {})
            pool = self.review["pool"]
            self.batch_ij = pool[["i", "j"]].reset_index(drop=True)
            self.batch_p1 = pool["p1"].to_numpy()
            sc = self.review["scored"]
            # other Source-1 entities' model probabilities per record (for exclusivity)
            col = "p_model" if "p_model" in sc else "p_match"
            self.claims = sc.groupby("r")[col].apply(list).to_dict()
            # records the batch already resolved to a Source-1 entity
            em = sc[sc.emitted]
            names = self.review["s1"].set_index("entity_id")["business_name"]
            self.owners = {r_: [dict(id=s_, name=names.get(s_, ""), p=float(p_))]
                           for s_, r_, p_ in zip(em.s1.values, em.r.values, em.p_match.values)}
            self.ready.set()
        except Exception as e:  # surfaced through /api/health
            self.error = f"{type(e).__name__}: {e}"

    def resolve(self, name: str, address: str, country: str, compete: bool = True) -> dict:
        """`compete=False` resolves the record as if the batch did not exist:
        no reverse competition, no exclusivity -- 'what would match if this were new'."""
        M, cfg = self.M, self.M["decision"]
        q = pd.DataFrame([{"entity_id": QUERY_ID, "business_name": name.strip(),
                           "business_address": address.strip(), "country": country.strip()}])
        Pq = prepare(q, self.lex)
        poolq = self.index.query(Pq)
        n1 = len(self.P1)
        out = dict(query=dict(name=name, address=address, country=country), compete=compete,
                   pool_size=int(len(poolq)),
                   candidates=[], pruned=[])
        if poolq.empty:
            out.update(chosen=[], curve=[1.0], expected_f=1.0)
            return out

        P1q = pd.concat([self.P1, Pq], ignore_index=True)
        X1 = build_pair_features(poolq, P1q, self.PR, self.groups)[M["x1_cols"]]
        p1 = predict_bag(M["pruners"], X1.values)
        keep = p1 >= M["tau"]

        if compete:
            comb = pd.concat([self.batch_ij, poolq[["i", "j"]]], ignore_index=True)
            ctx = build_context_features(comb, np.concatenate([self.batch_p1, p1]), self.PR)
            ctx = ctx.iloc[len(self.batch_ij):].reset_index(drop=True)
        else:
            ctx = build_context_features(poolq[["i", "j"]].reset_index(drop=True), p1, self.PR)
        X2 = pd.concat([X1.reset_index(drop=True), ctx], axis=1)[M["x2_cols"]]

        rids = self.PR.id.values[poolq.j.values]
        order_pruned = np.argsort(-p1)
        out["pruned"] = [dict(id=rids[k], name=self.PR.name_raw.values[poolq.j.values[k]],
                              address=self.PR.addr_raw.values[poolq.j.values[k]], p1=float(p1[k]))
                         for k in order_pruned if not keep[k]][:6]
        if not keep.any():
            out.update(chosen=[], curve=[1.0], expected_f=1.0)
            return out

        Xk = X2.values[keep]
        raw = predict_bag(M["matchers"], Xk)
        p = M["iso"].predict(raw) if cfg.get("cal") else raw
        kept_ids = rids[keep]
        if cfg.get("excl") and compete:
            # exclusivity against every batch entity already claiming each record
            q_p = []
            for rid, pi in zip(kept_ids, p):
                others = self.claims.get(rid, [])
                j = np.zeros(1 + len(others), dtype=int)
                q_p.append(exclusive(j, np.array([pi, *others]))[0])
            p = np.array(q_p)
        if cfg["mode"] == "threshold":
            chosen = [r for r, v in zip(kept_ids, p) if v >= cfg.get("thr", 0.5)]
            ef = None
        else:
            chosen, ef = choose_set(list(kept_ids), p, max_k=12)
        srt = np.sort(p)[::-1]
        contrib = np.mean([m.predict(Xk, pred_contrib=True) for m in M["matchers"]], axis=0)
        cols = list(X2.columns)
        jj = poolq.j.values[keep]
        cands = []
        for k in np.argsort(-p):
            c = contrib[k, :-1]
            top = np.argsort(-np.abs(c))[:8]
            cands.append(dict(
                id=kept_ids[k], name=self.PR.name_raw.values[jj[k]], address=self.PR.addr_raw.values[jj[k]],
                country=self.PR.country.values[jj[k]], p=float(p[k]), p1=float(p1[keep][k]),
                emitted=kept_ids[k] in chosen,
                shap=[dict(f=cols[i], v=float(Xk[k, i]), c=float(c[i])) for i in top], base=float(contrib[k, -1]),
                owned_by=self.owners.get(kept_ids[k], []),
            ))
        out.update(candidates=cands, chosen=list(chosen),
                   curve=[float(x) for x in expected_f_curve(np.clip(srt, 1e-9, 1 - 1e-9))],
                   expected_f=None if ef is None else float(ef), rule=cfg)
        return out


def load_review(art: Path, split: str) -> tuple[dict, dict]:
    with open(art / f"{split}_review.pkl", "rb") as fh:
        review = pickle.load(fh)
    with open(art / "models.pkl", "rb") as fh:
        models = pickle.load(fh)
    return review, models
