import { useEffect, useState } from "react";
import { api, type EntityRow } from "../api";
import { EntityDetail, useListKeys } from "../components/EntityDetail";
import { ErrorNote, Loading, num } from "../components/common";

const FILTERS = [
  ["all", "All"],
  ["matched", "Matched"],
  ["singleton", "No match"],
  ["contested", "Contested"],
  ["unseen", "Unseen country"],
] as const;

const PAGE = 60;

export function Browse({ selected, select }: { selected: string | null; select: (id: string) => void }) {
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<(typeof FILTERS)[number][0]>("all");
  const [items, setItems] = useState<EntityRow[] | null>(null);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const t = setTimeout(() => {
      api.entities(q, filter, 0, PAGE)
        .then((r) => { if (alive) { setItems(r.items); setTotal(r.total); setError(null); } })
        .catch((e) => alive && setError(e.message));
    }, 180);
    return () => { alive = false; clearTimeout(t); };
  }, [q, filter]);

  const more = () =>
    api.entities(q, filter, items?.length ?? 0, PAGE).then((r) => setItems([...(items ?? []), ...r.items]));

  useListKeys(items?.map((i) => i.id) ?? [], selected, select);

  return (
    <>
      <header className="view-head">
        <h1>Browse the test entities</h1>
        <p>Every Source 1 entity in the test file, with the records the pipeline matched to it. Filter to the cases
          that matter: no match, contested records, or the country the model never saw in training.</p>
      </header>
      <div className="split">
        <div className="list-panel">
          <input className="search" type="search" value={q} onChange={(e) => setQ(e.target.value)}
            placeholder="Search name, address or ID" aria-label="Search entities" />
          <div className="seg" role="group" aria-label="Filter entities">
            {FILTERS.map(([k, label]) => (
              <button key={k} aria-pressed={filter === k} onClick={() => setFilter(k)}>{label}</button>
            ))}
          </div>
          <span className="count" aria-live="polite">{items ? `${num(total)} entities` : "Loading entities"}</span>
          {error && <ErrorNote>{error}</ErrorNote>}
          <div className="list" role="listbox" aria-label="Source 1 entities">
            {!items && <Loading lines={8} />}
            {items?.length === 0 && <div className="empty"><p>No entities match this search and filter. Clear the search or pick another filter.</p></div>}
            {items?.map((e) => (
              <button key={e.id} id={`row-${e.id}`} className="row" role="option" aria-selected={e.id === selected} onClick={() => select(e.id)}>
                <span className="t">{e.name}</span>
                <span className="n">{e.n_match === 0 ? "no match" : `${e.n_match} matched`}</span>
                <span className="s">
                  <span>{e.id}</span><span>{e.country}</span>
                  {e.unseen && <span className="flag">unseen country</span>}
                  {e.contested && <span className="flag">contested</span>}
                </span>
              </button>
            ))}
            {items && items.length < total && (
              <button className="btn-quiet more" onClick={more}>Show {Math.min(PAGE, total - items.length)} more</button>
            )}
          </div>
        </div>
        <EntityDetail id={selected} emptyText="Pick an entity on the left, or use the arrow keys to step through the list." />
      </div>
    </>
  );
}
