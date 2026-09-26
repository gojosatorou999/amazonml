import { useEffect, useState } from "react";
import { api, type QueueItem } from "../api";
import { EntityDetail, useListKeys } from "../components/EntityDetail";
import { ErrorNote, Loading } from "../components/common";

export function Review({ selected, select }: { selected: string | null; select: (id: string) => void }) {
  const [items, setItems] = useState<QueueItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.review().then((q) => {
      setItems(q);
      if (!selected && q.length) select(q[0].id);
    }).catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useListKeys(items?.map((i) => i.id) ?? [], selected, select);

  return (
    <>
      <header className="view-head">
        <h1>Review the decisions that matter most</h1>
        <p>Ordered by how much the expected score would rise if a person confirmed or rejected one pair. That is not
          the same as "closest to 50%": a doubtful pair only ranks high if its answer would change what we emit.
          Use j and k to move through the queue.</p>
      </header>
      <div className="split">
        <div className="list-panel">
          <span className="count">{items ? `${items.length} entities where a human label would help` : "Ranking decisions"}</span>
          {error && <ErrorNote>{error}</ErrorNote>}
          <div className="list" role="listbox" aria-label="Review queue">
            {!items && <Loading lines={8} />}
            {items?.length === 0 && <div className="empty"><p>Every decision is confident. Nothing needs review.</p></div>}
            {items?.map((e) => (
              <button key={e.id} id={`row-${e.id}`} className="row" role="option" aria-selected={e.id === selected} onClick={() => select(e.id)}>
                <span className="t">{e.name}</span>
                <span className="gain" title="Rise in expected F0.5 for this entity if the pair is labelled">+{e.gain.toFixed(3)}</span>
                <span className="s">
                  <span>{e.id} with {e.pair}</span>
                  <span>p {e.p.toFixed(2)}</span>
                  {e.unseen && <span className="flag">unseen country</span>}
                </span>
              </button>
            ))}
          </div>
        </div>
        <EntityDetail id={selected} emptyText="Pick an entity from the queue." />
      </div>
    </>
  );
}
