import { useId, useState } from "react";

/**
 * Expected F0.5 as a function of how many candidates are emitted.
 *
 * The pipeline sorts candidates by probability and evaluates every prefix
 * k = 0..n exactly; k = 0 is "emit nothing", which is how singletons are won.
 * The chosen k is the argmax. This chart is that computation, drawn.
 */
export function DecisionCurve({ curve, chosenK, byRule = true }: { curve: number[]; chosenK: number; byRule?: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const titleId = useId();
  const n = curve.length - 1;

  const W = 420, H = 200, L = 34, R = 18, T = 36, B = 30, PAD = 14;
  const x = (k: number) => L + PAD + (n === 0 ? (W - L - R - PAD) / 2 : (k / n) * (W - L - R - PAD));
  const y = (v: number) => T + (1 - v) * (H - T - B);
  const pts = curve.map((v, k) => [x(k), y(v)] as const);
  const line = pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join("");
  const area = `${line}L${x(n).toFixed(1)},${y(0)}L${x(0).toFixed(1)},${y(0)}Z`;
  const kLabel = (k: number) => (k === 0 ? "none" : String(k));
  const describe = (k: number) =>
    `${k === 0 ? "Emit nothing" : `Emit the top ${k}`}: expected F0.5 ${curve[k].toFixed(3)}`;

  const next = chosenK + 1 <= n ? curve[chosenK + 1] : null;
  const summary =
    `Expected F0.5 for emitting 0 to ${n} candidates. ${byRule ? "The best is" : "The rule emitted"} ${describe(chosenK).toLowerCase()}.` +
    (next !== null ? ` Emitting one more would give ${next.toFixed(3)}.` : "");

  return (
    <figure className="curve" style={{ margin: 0, position: "relative" }}>
      <h3 id={titleId}>Expected score by how many records we emit</h3>
      <p className="sub">Each point is exact: every possible truth, weighted by its probability.</p>
      <div style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby={titleId} aria-describedby={`${titleId}-d`}
        onMouseLeave={() => setHover(null)}>
        <desc id={`${titleId}-d`}>{summary}</desc>
        <g className="grid">
          {[0, 0.5, 1].map((v) => (
            <line key={v} x1={L} x2={W - R} y1={y(v)} y2={y(v)} />
          ))}
        </g>
        <g className="axis">
          {[0, 0.5, 1].map((v) => (
            <text key={v} x={L - 8} y={y(v) + 4} textAnchor="end">{v}</text>
          ))}
          {curve.map((_, k) => (
            <text key={k} x={x(k)} y={H - 8} textAnchor="middle">{kLabel(k)}</text>
          ))}
        </g>
        {n > 0 && <path className="area" d={area} />}
        {n > 0 && <path className="line" d={line} />}
        {hover !== null && <line className="guide" x1={x(hover)} x2={x(hover)} y1={T - 6} y2={y(0)} />}
        <circle className="chosen-ring" cx={pts[chosenK][0]} cy={pts[chosenK][1]} r={11} />
        {pts.map(([px, py], k) => (
          <circle key={k} className={`pt${k === chosenK ? " chosen" : ""}`} cx={px} cy={py} r={k === chosenK ? 5.5 : 4} />
        ))}
        <text className="chosen-lbl" x={pts[chosenK][0]} y={pts[chosenK][1] - 16}
          textAnchor={chosenK === 0 ? "start" : chosenK === n ? "end" : "middle"}>
          {byRule ? "Best: " : "Emitted: "}{chosenK === 0 ? "nothing" : chosenK} ({curve[chosenK].toFixed(2)})
        </text>
        {/* hit targets wider than the marks */}
        {curve.map((_, k) => {
          const half = n === 0 ? 40 : Math.max(10, (W - L - R) / n / 2);
          return (
            <rect key={k} className="hit" x={x(k) - half} y={T - 10} width={half * 2} height={H - T - B + 12}
              onMouseEnter={() => setHover(k)} />
          );
        })}
      </svg>
      {hover !== null && (
        <div className="tip" style={{ left: `${(pts[hover][0] / W) * 100}%`, top: `${((pts[hover][1] + 18) / H) * 100}%` }}>
          {describe(hover)}
        </div>
      )}
      </div>
      <p className="sub" style={{ marginTop: 2 }}>Records emitted, most likely first</p>
      <details className="values-toggle">
        <summary>Show values</summary>
        <table>
          <thead><tr><th>Records emitted</th><th>Expected F0.5</th></tr></thead>
          <tbody>
            {curve.map((v, k) => (
              <tr key={k}><td>{k === 0 ? "None" : k}</td><td>{v.toFixed(4)}{k === chosenK ? " (chosen)" : ""}</td></tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
