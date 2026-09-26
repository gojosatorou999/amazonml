import { useEffect, useState } from "react";
import { api, type Entity } from "../api";
import { ResolutionPanel } from "./ResolutionPanel";
import { ErrorNote, Loading } from "./common";

export function EntityDetail({ id, emptyText }: { id: string | null; emptyText: string }) {
  const [data, setData] = useState<Entity | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let alive = true;
    setError(null);
    api.entity(id)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(e.message));
    return () => { alive = false; };
  }, [id]);

  if (!id) return <div className="empty"><h2>No entity selected</h2><p>{emptyText}</p></div>;
  if (error) return <ErrorNote>{error}</ErrorNote>;
  if (!data || data.id !== id) return <Loading lines={5} />;
  return (
    <ResolutionPanel
      anchor={{ kind: "entity", id: data.id, name: data.name, address: data.address, country: data.country }}
      candidates={data.candidates}
      curve={data.curve}
      chosen={data.chosen}
      pruned={data.pruned}
      rule={data.rule}
      runKey={data.id}
    />
  );
}

/** Arrow keys (and j/k) move the selection in a list without leaving the keyboard. */
export function useListKeys(ids: string[], selected: string | null, select: (id: string) => void) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey || e.altKey) return;
      const down = e.key === "ArrowDown" || e.key === "j";
      const up = e.key === "ArrowUp" || e.key === "k";
      if (!down && !up) return;
      e.preventDefault();
      const i = selected ? ids.indexOf(selected) : -1;
      const next = ids[Math.min(ids.length - 1, Math.max(0, i + (down ? 1 : -1)))];
      if (next) {
        select(next);
        document.getElementById(`row-${next}`)?.scrollIntoView({ block: "nearest" });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [ids, selected, select]);
}
