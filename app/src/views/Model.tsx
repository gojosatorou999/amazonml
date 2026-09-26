import { useEffect, useState } from "react";
import { api, featureName, type Overview } from "../api";
import { ErrorNote, Loading, num, pct } from "../components/common";

export function Model() {
  const [o, setO] = useState<Overview | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.overview().then(setO).catch((e) => setError(e.message)); }, []);

  return (
    <>
      <header className="view-head">
        <h1>How the pipeline decides</h1>
        <p>The evidence behind the submission: how blocking shrinks the search, which decision rule won and why, what
          the model relies on, and everything it learned from the data instead of a hand-written list.</p>
      </header>
      {error && <ErrorNote>{error}</ErrorNote>}
      {!o && !error && <Loading lines={6} />}
      {o && (
        <div className="model">
          <Funnel o={o} />
          <Rules o={o} />
          <Importance o={o} />
          <Learned o={o} />
        </div>
      )}
    </>
  );
}

function Funnel({ o }: { o: Overview }) {
  const t = o.test;
  const b = o.train.blocking;
  const stages = [
    { label: "Every possible pair", note: "Source 1 × Sources 2–3", v: t.n_s1 * t.n_r },
    { label: "Blocking pool", note: `${(t.pool_pairs / t.n_s1).toFixed(1)} per entity, 4 search channels`, v: t.pool_pairs },
    { label: "Candidate pairs", note: `${t.cand_per_s1.mean.toFixed(2)} per entity, submitted as candidate_pairs.tsv`, v: t.cand_pairs },
    { label: "Matches", note: `${pct(t.empty_share)} of entities get an empty answer`, v: t.matched_pairs },
  ];
  const top = Math.log10(stages[0].v) / 0.82; // leave room for the value label
  return (
    <section aria-labelledby="funnel-h">
      <h2 id="funnel-h">From {num(stages[0].v)} possible pairs to {num(t.cand_pairs)} candidates</h2>
      <p>
        The judges rank smaller candidate sets higher, at equal recall. Blocking keeps
        {b ? ` ${pct(b.pool_recall, 2)} of true pairs in the pool and ${pct(b.cand_recall, 2)} after pruning` : " nearly every true pair"}
        {b ? " (measured out of fold on train)" : ""}, while cutting the search by a factor of {num(Math.round(stages[0].v / t.cand_pairs))}.
      </p>
      <div className="funnel" role="table" aria-label="Pairs at each pipeline stage">
        {stages.map((s) => (
          <div className="funnel-row" role="row" key={s.label} title={`${s.label}: ${num(s.v)} pairs`}>
            <div className="lab" role="cell"><b>{s.label}</b><span>{s.note}</span></div>
            <div className="bar-wrap" role="cell">
              <div className="fbar" style={{ width: `${(100 * Math.log10(Math.max(s.v, 1))) / top}%` }} />
              <span className="val" style={{ left: `${(100 * Math.log10(Math.max(s.v, 1))) / top}%` }}>{num(s.v)}</span>
            </div>
          </div>
        ))}
      </div>
      <p className="funnel-note">Bar length is on a log scale, so equal steps in length are equal factors of reduction.</p>
      {t.holdout_score && (
        <p className="funnel-note" style={{ fontSize: 13, color: "var(--ink-2)" }}>
          On the synthetic hold-out, the only test split with known answers, macro F0.5 is{" "}
          <b>{t.holdout_score.macro_f.toFixed(4)}</b> with candidate recall {pct(t.holdout_cand_recall ?? 0)}.
        </p>
      )}
    </section>
  );
}

function Rules({ o }: { o: Overview }) {
  const v = o.train.variants ?? {};
  const loco = o.train.loco ?? {};
  const sel = o.train.selection ?? {};
  const w = o.train.unseen_share ?? 0;
  return (
    <section aria-labelledby="rules-h">
      <h2 id="rules-h">Which decision rule, and why</h2>
      <p>
        Six ways to turn probabilities into an answer, scored on held-out folds. Cross-validation holds out entities;
        leave-one-country-out trains on one country and predicts the other, the stand-in for France. {pct(w)} of test
        entities come from a country never seen in training, so the rule is picked by the score blended in that ratio.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="rules">
          <thead>
            <tr><th>Decision rule</th><th className="num">Cross-validation</th><th className="num">Unseen country</th><th className="num">Blended</th></tr>
          </thead>
          <tbody>
            {Object.entries(v).map(([k, r]) => (
              <tr key={k} className={k === o.train.chosen ? "chosen" : ""}>
                <td>{k[0].toUpperCase() + k.slice(1)}{k === o.train.chosen ? " (used)" : ""}</td>
                <td className="num">{r.score.macro_f.toFixed(4)}</td>
                <td className="num">{loco[k] ? loco[k].score.macro_f.toFixed(4) : "n/a"}</td>
                <td className="num">{sel[k] !== undefined ? sel[k].toFixed(4) : "n/a"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Importance({ o }: { o: Overview }) {
  const imp = o.train.importance.slice(0, 12);
  const max = Math.max(...imp.map(([, g]) => g), 1);
  const total = o.train.importance.reduce((a, [, g]) => a + g, 0) || 1;
  return (
    <section aria-labelledby="imp-h">
      <h2 id="imp-h">What the matcher relies on</h2>
      <p>Share of the model's total split gain. Most of it goes to competition: how this pair compares with the other
        candidates for the same entity and the other entities claiming the same record. Identity is comparative.</p>
      <div className="imp" role="table" aria-label="Feature importance">
        {imp.map(([f, g]) => (
          <div className="imp-row" role="row" key={f} title={`${f}: gain ${num(Math.round(g))}`}>
            <span role="cell">{featureName(f)}</span>
            <span role="cell" aria-hidden="true"><div className="ibar" style={{ width: `${(100 * g) / max}%` }} /></span>
            <span className="num" role="cell">{pct(g / total)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function Learned({ o }: { o: Overview }) {
  const groups: Record<string, string[]> = {};
  Object.entries(o.groups).forEach(([label, g]) => (groups[g] ??= []).push(label));
  const abbrev = Object.entries(o.lexicon.abbrev);
  return (
    <section aria-labelledby="lex-h">
      <h2 id="lex-h">Learned from the data, not looked up</h2>
      <p>External data is banned, so nothing here comes from a gazetteer. Abbreviations were mined from the training
        pairs, legal forms from how names end, landmark words from address parts one source adds and the other doesn't,
        and country spellings were grouped by matching the records themselves.</p>
      <div className="lex">
        <div>
          <h3>Country spellings grouped ({Object.keys(groups).length})</h3>
          <div className="chips">
            {Object.entries(groups).map(([g, labels]) => <span className="chip" key={g}>{labels.sort().join(" = ")}</span>)}
          </div>
        </div>
        <div>
          <h3>Abbreviations ({abbrev.length})</h3>
          <div className="chips">
            {abbrev.map(([s, l]) => <span className="chip" key={s}>{s}<i>means</i>{l}</span>)}
          </div>
        </div>
        <div>
          <h3>Legal and generic name endings ({o.lexicon.legal.length})</h3>
          <div className="chips">{o.lexicon.legal.map((t) => <span className="chip" key={t}>{t}</span>)}</div>
        </div>
        <div>
          <h3>Landmark words ({o.lexicon.cues.length})</h3>
          <div className="chips">{o.lexicon.cues.map((t) => <span className="chip" key={t}>{t}</span>)}</div>
        </div>
      </div>
    </section>
  );
}
