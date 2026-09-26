import { useState } from "react";
import { featureName, type Candidate, type DecisionRule, type Pruned, type Shap } from "../api";
import { DecisionCurve } from "./DecisionCurve";
import { MatchState, Shared, SourceTag, wordSet } from "./common";

type Anchor = { kind: "entity" | "query"; id?: string; name: string; address: string; country: string };

type Props = {
  anchor: Anchor;
  candidates: Candidate[];
  curve: number[];
  chosen: string[];
  pruned: Pruned[];
  rule?: DecisionRule;
  compete?: boolean;
  runKey?: string | number; // changes when a new result arrives, to replay the draw-in
};

export function ResolutionPanel({ anchor, candidates, curve, chosen, pruned, rule, compete, runKey }: Props) {
  const ref = wordSet(anchor.name, anchor.address);
  const k = chosen.length;
  const owners = anchor.kind === "query" && compete
    ? candidates.flatMap((c) => c.owned_by ?? []).filter((o, i, a) => a.findIndex((x) => x.id === o.id) === i)
    : [];
  const blockedByOwner = owners.length > 0 && k === 0;

  return (
    <div className="resolution">
      <div className={`anchor ${anchor.kind}`}>
        <span className="dot" aria-hidden="true" />
        <div className="rec">
          <div className="rec-name">{anchor.name}</div>
          <div className="rec-addr">{anchor.address || "No address given"}</div>
          <div className="rec-meta">
            {anchor.id ? <SourceTag id={anchor.id} /> : <span>Your record, treated as a new Source 1 entity</span>}
            {anchor.country && <span>{anchor.country}</span>}
          </div>
        </div>
      </div>

      <div className="verdict">
        <div>
          <h2>{headline(k, candidates.length, blockedByOwner)}</h2>
          <p>{explain(curve, k, candidates.length, blockedByOwner)}</p>
          {rule && <p className="sub" style={{ fontSize: 13, color: "var(--ink-3)" }}>{ruleText(rule, compete)}</p>}
          {owners.length > 0 && (
            <div className="notice">
              {blockedByOwner ? "This looks like an existing Source 1 entity. " : ""}
              The closest records already belong to{" "}
              {owners.slice(0, 2).map((o, i) => (
                <span key={o.id}>{i > 0 && " and "}<b>{o.id}</b> {o.name}</span>
              ))}
              . Source 1 is deduplicated, so a record can have only one owner.
              {blockedByOwner && " Turn off “Keep records with their current owner” to see what would match if this business were new."}
            </div>
          )}
        </div>
        {candidates.length > 0 && <DecisionCurve curve={curve} chosenK={k} byRule={rule?.mode !== "threshold"} />}
      </div>

      <section aria-labelledby="pairs-h">
        <div className="pairs-head">
          <h3 id="pairs-h">Candidates the model scored</h3>
          <span>{candidates.length === 0 ? "none reached the matcher" : `${candidates.length} from blocking, by match probability`}</span>
        </div>
        {candidates.length === 0 && (
          <p className="empty" style={{ padding: "16px 0" }}>
            {pruned.length > 0
              ? `Blocking found ${pruned.length} look-alike${pruned.length === 1 ? "" : "s"}, but the first-pass model ruled each one out before matching. `
              : "Blocking found nothing in Sources 2 and 3 that looks like this business. "}
            The answer is an empty list, which scores a full 1.0 on a business that appears only in Source 1.
          </p>
        )}
        {candidates.map((c, i) => (
          <PairRow key={`${runKey}-${c.id}`} c={c} i={i} refWords={ref} />
        ))}
      </section>

      {pruned.length > 0 && (
        <details className="pruned">
          <summary>{pruned.length} more {pruned.length === 1 ? "record was" : "records were"} found by blocking but cut before matching</summary>
          <p style={{ marginTop: 6 }}>The first-pass model gave each of these too little chance to be worth scoring. This is why the candidate list stays small.</p>
          <ul>
            {pruned.map((p) => (
              <li key={p.id}>
                <span>{p.p1 < 0.001 ? "<0.001" : p.p1.toFixed(3)}</span>
                <span><SourceTag id={p.id} /> {p.name}, {p.address}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function PairRow({ c, i, refWords }: { c: Candidate; i: number; refWords: Set<string> }) {
  const [open, setOpen] = useState(false);
  const w = 1 + 5 * c.p;
  return (
    <article className="pair" aria-label={`${c.id}, match probability ${c.p.toFixed(2)}`}>
      <div className="link">
        <span className="p">{c.p.toFixed(2)}</span>
        <svg viewBox="0 0 100 14" preserveAspectRatio="none" aria-hidden="true">
          <line className="draw" x1="2" x2="98" y1="7" y2="7" strokeWidth={w} pathLength={100}
            style={{ opacity: 0.25 + 0.75 * c.p, ["--i" as string]: i }} vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
      <div className="rec">
        <div className="rec-name"><Shared text={c.name} against={refWords} /></div>
        <div className="rec-addr"><Shared text={c.address} against={refWords} /></div>
        <div className="rec-meta">
          <SourceTag id={c.id} />
          <span>{c.country}</span>
          <span>first pass {c.p1.toFixed(2)}</span>
        </div>
        {c.also_claimed_by && c.also_claimed_by.length > 0 && (
          <p className="owned">Also a candidate for {c.also_claimed_by.map((o) => `${o.id} (${o.name})`).join(", ")}</p>
        )}
        {c.owned_by && c.owned_by.length > 0 && (
          <p className="owned">In the submitted results this record belongs to <b>{c.owned_by[0].id}</b> {c.owned_by[0].name}</p>
        )}
        <button className="why-btn" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? "Hide the reasons" : "Why this probability"}
        </button>
        {open && <ShapBars shap={c.shap} />}
      </div>
      <MatchState on={c.emitted} />
    </article>
  );
}

function ShapBars({ shap }: { shap: Shap[] }) {
  const rows = shap.slice(0, 7);
  const m = Math.max(...rows.map((s) => Math.abs(s.c)), 1e-6);
  return (
    <div className="shap" role="table" aria-label="Contributions to the match score, in log-odds">
      <div className="shap-cap" aria-hidden="true"><span>Against a match</span><span>Toward a match</span></div>
      {rows.map((s) => {
        const wpct = (50 * Math.abs(s.c)) / m;
        return (
          <div className="shap-row" role="row" key={s.f}>
            <span className="nm" role="cell" title={`${featureName(s.f)} (value ${s.v.toFixed(3)})`}>{featureName(s.f)}</span>
            <span className="track" role="cell" aria-hidden="true">
              <span className={`bar ${s.c >= 0 ? "pos" : "neg"}`}
                style={{ left: s.c >= 0 ? "50%" : `${50 - wpct}%`, width: `${wpct}%` }} />
            </span>
            <span className="num" role="cell">{s.c >= 0 ? "+" : "−"}{Math.abs(s.c).toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}

function headline(k: number, n: number, blockedByOwner: boolean) {
  if (n === 0) return "No match: nothing close enough was found";
  if (blockedByOwner) return "No new match: its records already have an owner";
  if (k === 0) return "No match: the model chose to abstain";
  return k === 1 ? "Matched to 1 record" : `Matched to ${k} records`;
}

function explain(curve: number[], k: number, n: number, blockedByOwner: boolean) {
  if (n === 0) return "An empty answer is correct for a business that appears only in Source 1.";
  const best = curve[k].toFixed(2);
  if (k === 0) {
    return blockedByOwner
      ? `Claiming them would duplicate an existing entity. Emitting nothing keeps the expected score highest (${best}).`
      : `Emitting nothing has the highest expected score (${best}). The best candidate is too uncertain to risk a false merge, which F0.5 punishes twice as hard as a miss.`;
  }
  const next = curve[k + 1];
  return next === undefined
    ? `Every candidate is worth emitting; the expected score peaks at ${best}.`
    : `Emitting the top ${k} gives the highest expected score (${best}). Adding the next candidate would drop it to ${next.toFixed(2)}.`;
}

function ruleText(rule: DecisionRule, compete?: boolean) {
  const base = rule.mode === "threshold"
    ? `Decision rule: probability at least ${rule.thr?.toFixed(2)}`
    : `Decision rule: maximise expected F0.5 using ${rule.cal ? "calibrated" : "model"} probabilities`;
  const excl = rule.excl ? (compete === false ? ", one-owner check off for this query" : ", with the one-owner check") : "";
  return `${base}${excl}. Chosen on cross-validation and leave-one-country-out data.`;
}
