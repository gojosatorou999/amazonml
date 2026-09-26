import { useCallback, useEffect, useState } from "react";
import { Browse } from "./views/Browse";
import { Model } from "./views/Model";
import { Resolve } from "./views/Resolve";
import { Review } from "./views/Review";

type View = "resolve" | "browse" | "review" | "model";

const NAV: { id: View; label: string; hint: string }[] = [
  { id: "resolve", label: "Resolve", hint: "Try a record live" },
  { id: "browse", label: "Browse", hint: "Every test entity" },
  { id: "review", label: "Review", hint: "Decisions worth a second look" },
  { id: "model", label: "Model", hint: "The evidence" },
];

function parseHash(): { view: View; id: string | null } {
  const [, v, id] = window.location.hash.split("/");
  const view = (NAV.some((n) => n.id === v) ? v : "resolve") as View;
  return { view, id: id ? decodeURIComponent(id) : null };
}

type Theme = "system" | "light" | "dark";
const readTheme = (): Theme => {
  try { return (localStorage.getItem("ber-theme") as Theme) || "system"; } catch { return "system"; }
};

export function App() {
  const [route, setRoute] = useState(parseHash);
  const [theme, setTheme] = useState<Theme>(readTheme);

  useEffect(() => {
    const on = () => setRoute(parseHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  useEffect(() => {
    const el = document.documentElement;
    if (theme === "system") el.removeAttribute("data-theme");
    else el.setAttribute("data-theme", theme);
    try { localStorage.setItem("ber-theme", theme); } catch { /* storage unavailable */ }
  }, [theme]);

  useEffect(() => {
    const titles: Record<View, string> = { resolve: "Resolve", browse: "Browse", review: "Review", model: "Model" };
    document.title = `${titles[route.view]} | Entity Resolution Console`;
  }, [route.view]);

  const select = useCallback(
    (id: string) => { window.location.hash = `/${route.view}/${encodeURIComponent(id)}`; },
    [route.view],
  );

  const nextTheme: Record<Theme, Theme> = { system: "dark", dark: "light", light: "system" };

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <svg width="30" height="20" viewBox="0 0 30 20" aria-hidden="true">
            <circle cx="6" cy="10" r="5" fill="var(--s1)" />
            <circle cx="24" cy="5" r="4" fill="var(--s2)" />
            <circle cx="24" cy="15" r="4" fill="var(--s3)" />
            <path d="M11 10 L20 5.5 M11 10 L20 14.5" stroke="var(--ink)" strokeWidth="1.6" />
          </svg>
          <div>Entity Resolution<small>Amazon ML Challenge 2026</small></div>
        </div>
        <nav className="nav" aria-label="Views">
          {NAV.map((n) => (
            <a key={n.id} href={`#/${n.id}`} aria-current={route.view === n.id ? "page" : undefined}>
              {n.label}
              <small>{n.hint}</small>
            </a>
          ))}
        </nav>
        <div className="rail-foot">
          <div className="legend" aria-label="Source colours">
            <span className="src" data-s="1">Source 1, the reference</span>
            <span className="src" data-s="2">Source 2</span>
            <span className="src" data-s="3">Source 3</span>
          </div>
          <button className="theme-btn" onClick={() => setTheme(nextTheme[theme])} aria-label={`Colour theme: ${theme}. Change theme`}>
            Theme: {theme === "system" ? "system" : theme}
          </button>
        </div>
      </aside>
      <main>
        {route.view === "resolve" && <Resolve />}
        {route.view === "browse" && <Browse selected={route.id} select={select} />}
        {route.view === "review" && <Review selected={route.id} select={select} />}
        {route.view === "model" && <Model />}
      </main>
    </div>
  );
}
