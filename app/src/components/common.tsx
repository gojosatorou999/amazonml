import { Fragment, type ReactNode } from "react";
import { sourceOf } from "../api";

/** Fold a word the way the pipeline does closely enough for highlighting. */
export const fold = (w: string) =>
  w.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase();

export const wordSet = (...texts: string[]) =>
  new Set(texts.flatMap((t) => t.split(/[^\p{L}\p{N}]+/u).filter(Boolean).map(fold)));

/** Render text with the words it shares with `ref` highlighted. */
export function Shared({ text, against: refWords }: { text: string; against: Set<string> }) {
  const parts = text.split(/([\p{L}\p{N}]+)/u);
  return (
    <>
      {parts.map((p, i) =>
        i % 2 === 1 && refWords.has(fold(p)) ? (
          <mark key={i} className="tok-hit">
            {p}
          </mark>
        ) : (
          <Fragment key={i}>{p}</Fragment>
        ),
      )}
    </>
  );
}

export function SourceTag({ id }: { id: string }) {
  const s = sourceOf(id);
  return (
    <span className="src" data-s={s}>
      {id}
    </span>
  );
}

export function MatchState({ on }: { on: boolean }) {
  return on ? (
    <span className="state on">
      <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="8" fill="currentColor" />
        <path d="M4.5 8.3l2.3 2.3 4.7-5" fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
      Matched
    </span>
  ) : (
    <span className="state off">Not matched</span>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p className="error" role="alert">
      {children}
    </p>
  );
}

export function Loading({ lines = 3 }: { lines?: number }) {
  return (
    <div aria-busy="true" aria-label="Loading">
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className="skeleton" style={{ width: `${90 - i * 18}%` }} />
      ))}
    </div>
  );
}

export const pct = (x: number, d = 1) => `${(100 * x).toFixed(d)}%`;
export const num = (x: number) => x.toLocaleString("en-US");
