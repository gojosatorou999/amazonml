import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, ApiError, type QueryResult, type Sample } from "../api";
import { ResolutionPanel } from "../components/ResolutionPanel";
import { ErrorNote, Loading } from "../components/common";

export function Resolve() {
  const [form, setForm] = useState({ name: "", address: "", country: "" });
  const [compete, setCompete] = useState(false);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [ready, setReady] = useState<"ready" | "warming" | "error">("warming");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [run, setRun] = useState(0);
  const nameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.samples().then(setSamples).catch(() => setSamples([]));
    let alive = true;
    const poll = async () => {
      try {
        const h = await api.health();
        if (!alive) return;
        setReady(h.resolver);
        if (h.resolver === "warming") setTimeout(poll, 1500);
        if (h.resolver === "error") setError(`The resolver failed to start: ${h.error}`);
      } catch (e) {
        if (alive) setError((e as Error).message);
      }
    };
    poll();
    nameRef.current?.focus();
    return () => { alive = false; };
  }, []);

  const submit = async (e?: FormEvent, override?: typeof form, competeOverride?: boolean) => {
    e?.preventDefault();
    const q = override ?? form;
    if (!q.name.trim()) {
      setError("Enter a business name to resolve.");
      nameRef.current?.focus();
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setResult(await api.resolve({ ...q, compete: competeOverride ?? compete }));
      setRun((r) => r + 1);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Resolving failed. Check the server log.");
    } finally {
      setBusy(false);
    }
  };

  const trySample = (s: Sample) => {
    const q = { name: s.name, address: s.address, country: s.country };
    setForm(q);
    submit(undefined, q);
  };

  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) => setForm({ ...form, [k]: e.target.value });

  return (
    <>
      <header className="view-head">
        <h1>Resolve a business record</h1>
        <p>
          Type a record the way a messy source would write it. It runs through the submitted pipeline: blocking,
          both models and the expected-score decision. You see what it matches and why.
        </p>
      </header>

      <form className="query-form" onSubmit={submit} aria-describedby="resolve-status">
        <div className="field">
          <label htmlFor="q-name">Business name</label>
          <input id="q-name" ref={nameRef} value={form.name} onChange={set("name")} placeholder="Tata Consultancy Svcs Pvt Ltd" autoComplete="off" />
        </div>
        <div className="field">
          <label htmlFor="q-addr">Address</label>
          <input id="q-addr" value={form.address} onChange={set("address")} placeholder="Near SBI ATM, 12/34 MG Rd, Pune" autoComplete="off" />
        </div>
        <div className="field">
          <label htmlFor="q-country">Country</label>
          <input id="q-country" value={form.country} onChange={set("country")} placeholder="IN" autoComplete="off" />
        </div>
        <button className="btn" type="submit" disabled={busy || ready !== "ready"}>
          {ready === "warming" ? "Preparing…" : busy ? "Resolving…" : "Resolve"}
        </button>
        <div className="form-foot">
          <div className="samples">
            <span>Try</span>
            {samples.map((s) => (
              <button type="button" key={s.label} onClick={() => trySample(s)} disabled={ready !== "ready"} title={`${s.name}, ${s.address}`}>
                {s.label}
              </button>
            ))}
          </div>
          <label className="check">
            <input type="checkbox" checked={compete}
              onChange={(e) => { setCompete(e.target.checked); if (result) submit(undefined, undefined, e.target.checked); }} />
            <span>Keep records with their current owner<br /><small style={{ color: "var(--ink-3)" }}>A record the batch already resolved to a Source 1 entity can't be claimed again</small></span>
          </label>
        </div>
      </form>

      <p id="resolve-status" className="sr-only" aria-live="polite">
        {ready === "warming" ? "The resolver is building its search index." : busy ? "Resolving." : result ? `${result.chosen.length} records matched.` : ""}
      </p>
      {ready === "warming" && !error && (
        <p className="count" style={{ marginTop: 12 }}>Building the search index over the test records. This takes a few seconds once per server start.</p>
      )}
      {error && <div style={{ marginTop: 16 }}><ErrorNote>{error}</ErrorNote></div>}
      {busy && !result && <div style={{ marginTop: 28 }}><Loading lines={4} /></div>}

      {result ? (
        <div style={{ opacity: busy ? 0.5 : 1, transition: "opacity .15s" }}>
          <ResolutionPanel
            anchor={{ kind: "query", ...result.query }}
            candidates={result.candidates}
            curve={result.curve}
            chosen={result.chosen}
            pruned={result.pruned}
            rule={result.rule}
            compete={result.compete}
            runKey={run}
          />
        </div>
      ) : (
        !busy && (
          <div className="empty">
            <h2>Nothing resolved yet</h2>
            <p>
              Enter a record above, or pick one of the examples. The last example is a business that isn't in the
              data: watch the model decide that the best answer is no match at all.
            </p>
          </div>
        )
      )}
    </>
  );
}
