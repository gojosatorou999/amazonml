"""Local API + app server -- standard library only, fully offline.

    python -m ber.service.app [--artifacts artifacts] [--split test] [--port 8000]

Serves the built React app from app/dist (falls back to the single-file
ui.html if the app has not been built) and a JSON API:

    GET  /api/health          resolver warm-up state
    GET  /api/overview        run statistics, CV/LOCO ablation, blocking funnel,
                              feature importance, learned lexicon
    GET  /api/entities        ?q=&filter=all|matched|singleton|contested|unseen&offset=&limit=
    GET  /api/entity/<id>     candidates, decision, E[F] curve, SHAP, pruned pool, competing claims
    GET  /api/review          review queue ordered by expected value of perfect information
    GET  /api/samples         example records to try in the resolver
    POST /api/resolve         {"name", "address", "country"} -> live resolution

SHAP values come from LightGBM's native TreeSHAP (`pred_contrib=True`),
averaged over the fold bag -- exact attributions, no extra dependency.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from ..decide.expected_f import evpi, expected_f_curve
from ..normalize.text import tokens
from .resolver import Resolver, load_review

ROOT = Path(__file__).resolve().parents[3]
UI_FALLBACK = Path(__file__).with_name("ui.html")
DIST = ROOT / "app" / "dist"


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


class State:
    def __init__(self, art: Path, split: str):
        self.review, self.models = load_review(art, split)
        self.result = _read_json(art / f"{split}_result.json")
        self.train = _read_json(art / "train_result.json")
        self.lexicon = _read_json(art / "lexicon.json")
        d = self.review
        self.scored, self.X = d["scored"], d["X"]
        self.s1 = d["s1"].set_index("entity_id")
        self.r = d["r"].set_index("entity_id")
        self.s1_order = d["s1"].entity_id.to_numpy()
        self.r_order = d["r"].entity_id.to_numpy()
        self.cols = list(self.X.columns)
        self.by_s1 = {s: g.index.to_numpy() for s, g in self.scored.groupby("s1")}
        em = self.scored[self.scored.emitted]
        self.owners = em.groupby("r")["s1"].apply(list).to_dict()
        self.claims = self.scored.groupby("r")["s1"].apply(list).to_dict()
        pool = d["pool"]
        tau = self.models["tau"]
        cut = pool[pool.p1 < tau].sort_values("p1", ascending=False)
        self.pruned = {self.s1_order[i]: g for i, g in cut.groupby("i")}
        self.groups = d.get("groups", {})
        train_labels = set(self.train.get("train_labels", []))
        self.unseen = {sid for sid, c in zip(self.s1_order, d["s1"].country) if train_labels and c not in train_labels}
        self.n_match = {s: int(self.scored.emitted.values[rows].sum()) for s, rows in self.by_s1.items()}
        self.contested = {s for s, rows in self.by_s1.items()
                          if any(len(self.claims.get(r_, [])) > 1 for r_ in self.scored.r.values[rows])}
        self._queue = None
        self.resolver = Resolver(art, self.review, self.models)

    # ------------------------------------------------------------------ views
    def overview(self) -> dict:
        t, res = self.train, self.result
        return dict(
            test=dict((k, res.get(k)) for k in ("n_s1", "n_r", "pool_pairs", "cand_pairs", "cand_per_s1",
                                                  "matched_pairs", "empty_share", "holdout_score",
                                                  "holdout_cand_recall", "holdout_pool_recall", "timings")),
            groups=self.groups,
            train=dict(blocking=t.get("blocking"), variants=t.get("variants"), loco=t.get("loco_variants"),
                       selection=t.get("selection"), chosen=t.get("chosen"), unseen_share=t.get("unseen_share"),
                       importance=t.get("importance", [])[:15], n_features=t.get("n_features")),
            lexicon=dict(abbrev=self.lexicon.get("abbrev", {}), legal=self.lexicon.get("legal", []),
                         cues=self.lexicon.get("cues", [])),
            decision=self.models.get("decision"), tau=self.models.get("tau"),
        )

    def entity_row(self, sid: str) -> dict:
        rec = self.s1.loc[sid]
        rows = self.by_s1.get(sid, ())
        return dict(id=sid, name=rec.business_name, address=rec.business_address, country=rec.country,
                    n_cand=len(rows), n_match=self.n_match.get(sid, 0),
                    unseen=sid in self.unseen, contested=sid in self.contested)

    def entities(self, q: str, flt: str, offset: int, limit: int) -> dict:
        ids = self.s1_order
        if flt == "matched":
            ids = [s for s in ids if self.n_match.get(s, 0) > 0]
        elif flt == "singleton":
            ids = [s for s in ids if self.n_match.get(s, 0) == 0]
        elif flt == "contested":
            ids = [s for s in ids if s in self.contested]
        elif flt == "unseen":
            ids = [s for s in ids if s in self.unseen]
        q = q.lower().strip()
        if q:
            ids = [s for s in ids if q in s.lower() or q in self.s1.at[s, "business_name"].lower()
                   or q in self.s1.at[s, "business_address"].lower()]
        ids = list(ids)
        return dict(total=len(ids), items=[self.entity_row(s) for s in ids[offset: offset + limit]])

    def entity(self, sid: str) -> dict:
        out = self.entity_row(sid)
        rows = self.by_s1.get(sid, np.array([], dtype=int))
        s_name = set(tokens(out["name"]))
        s_addr = set(tokens(out["address"]))
        cands = []
        if len(rows):
            X = self.X.values[rows]
            contrib = np.mean([m.predict(X, pred_contrib=True) for m in self.models["matchers"]], axis=0)
            for k, row in enumerate(rows):
                sc = self.scored.iloc[row]
                rr = self.r.loc[sc.r]
                c = contrib[k, :-1]
                top = np.argsort(-np.abs(c))[:8]
                cands.append(dict(
                    id=sc.r, name=rr.business_name, address=rr.business_address, country=rr.country,
                    p=float(sc.p_match), p1=float(sc.p_stage1), emitted=bool(sc.emitted),
                    shap=[dict(f=self.cols[i], v=float(X[k, i]), c=float(c[i])) for i in top],
                    base=float(contrib[k, -1]),
                    also_claimed_by=[dict(id=o, name=self.s1.at[o, "business_name"])
                                     for o in self.claims.get(sc.r, []) if o != sid][:3],
                ))
        cands.sort(key=lambda d: -d["p"])
        srt = np.array([c["p"] for c in cands])
        curve = expected_f_curve(np.clip(srt, 1e-9, 1 - 1e-9)).tolist() if len(srt) else [1.0]
        pr = self.pruned.get(sid)
        pruned = [] if pr is None else [
            dict(id=self.r_order[j], name=self.r.at[self.r_order[j], "business_name"],
                 address=self.r.at[self.r_order[j], "business_address"], p1=float(p))
            for j, p in zip(pr.j.values[:6], pr.p1.values[:6])]
        out.update(candidates=cands, curve=curve, chosen=[c["id"] for c in cands if c["emitted"]],
                   pruned=pruned, s1_tokens=dict(name=sorted(s_name), address=sorted(s_addr)),
                   rule=self.models.get("decision"))
        return out

    def queue(self, limit: int = 150) -> list[dict]:
        if self._queue is None:
            q = []
            for sid, rows in self.by_s1.items():
                p = self.scored.p_match.values[rows]
                gain, c = evpi(p)
                if gain > 1e-4:
                    q.append(dict(self.entity_row(sid), gain=gain, pair=self.scored.r.values[rows[c]],
                                  pair_name=self.r.at[self.scored.r.values[rows[c]], "business_name"],
                                  p=float(p[c])))
            self._queue = sorted(q, key=lambda d: -d["gain"])
        return self._queue[:limit]

    def samples(self) -> list[dict]:
        """One noisy, already-matched record per country group, plus a business that isn't there."""
        out, seen = [], set()
        em = self.scored[self.scored.emitted & (self.scored.p_match > 0.6)]
        for s, rid in zip(em.s1.values, em.r.values):
            rec = self.r.loc[rid]
            g = self.groups.get(rec.country, rec.country)
            if g in seen or rec.business_name.lower() == self.s1.at[s, "business_name"].lower():
                continue
            seen.add(g)
            out.append(dict(label=f"Source {rid[1]} record, {g}", name=rec.business_name,
                            address=rec.business_address, country=rec.country))
        out.append(dict(label="A business that is not in the data", name="Northwind Lantern Bakery LLC",
                        address="77 Orchard Hollow Rd, Millbrook, 04112", country="US"))
        return out


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str, code: int = 200, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=31536000, immutable" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(json.dumps(obj, default=str).encode("utf-8"), "application/json; charset=utf-8", code)

        def log_message(self, *a):  # quiet
            pass

        def _static(self, path: str) -> None:
            if DIST.exists():
                f = (DIST / path.lstrip("/")).resolve()
                if path != "/" and f.is_file() and DIST.resolve() in f.parents:
                    ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                    return self._send(f.read_bytes(), ctype, cache="/assets/" in path)
                return self._send((DIST / "index.html").read_bytes(), "text/html; charset=utf-8")
            return self._send(UI_FALLBACK.read_bytes(), "text/html; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            arg = lambda k, d="": qs.get(k, [d])[0]  # noqa: E731
            try:
                if not u.path.startswith("/api/"):
                    return self._static(u.path)
                if u.path == "/api/health":
                    rs = state.resolver
                    self._json(dict(resolver="ready" if rs.ready.is_set() else ("error" if rs.error else "warming"),
                                    error=rs.error))
                elif u.path in ("/api/overview", "/api/summary"):
                    self._json(state.overview())
                elif u.path == "/api/entities":
                    self._json(state.entities(arg("q"), arg("filter", "all"), int(arg("offset", "0")),
                                              min(int(arg("limit", "50")), 200)))
                elif u.path.startswith("/api/entity/"):
                    self._json(state.entity(unquote(u.path.rsplit("/", 1)[1])))
                elif u.path == "/api/review":
                    self._json(state.queue())
                elif u.path == "/api/samples":
                    self._json(state.samples())
                else:
                    self._json({"error": "No such endpoint."}, 404)
            except KeyError as e:
                self._json({"error": f"No Source-1 entity with id {e}."}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/api/resolve":
                return self._json({"error": "No such endpoint."}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "Request body must be JSON."}, 400)
            name = str(body.get("name", "")).strip()
            if not name:
                return self._json({"error": "Enter a business name to resolve."}, 400)
            rs = state.resolver
            if rs.error:
                return self._json({"error": f"The resolver failed to start: {rs.error}"}, 500)
            if not rs.ready.wait(timeout=60):
                return self._json({"error": "The resolver is still building its index. Try again in a few seconds."}, 503)
            self._json(rs.resolve(name, str(body.get("address", "")), str(body.get("country", "")),
                                  compete=bool(body.get("compete", True))))

    return H


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--split", default="test")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    state = State(Path(a.artifacts), a.split)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state))
    where = "React app" if DIST.exists() else "fallback UI (run `make app` to build the React app)"
    print(f"{where} + API on http://127.0.0.1:{a.port}  (Ctrl+C to stop)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
