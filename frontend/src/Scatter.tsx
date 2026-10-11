import { useState } from "react";
import { Maximize2, Crosshair } from "lucide-react";
import type { Result, Variant } from "./types";

export default function Scatter({
  result,
  inspect,
}: {
  result: Result;
  inspect: (v: Variant) => void;
}) {
  const [hover, setHover] = useState<Variant | null>(null);
  const [only, setOnly] = useState(false);
  const [uncertainty, setUncertainty] = useState(false);
  const points = result.variants;
  const xs = points.map((p) => p.binding_mean),
    ys = points.map((p) => p.stability_mean);
  const xMin = Math.min(...xs) - 0.15,
    xMax = Math.max(...xs) + 0.15;
  const yMin = Math.min(...ys, result.config.min_stability) - 2,
    yMax = Math.max(...ys) + 2;
  const x = (v: number) => 68 + ((v - xMin) / (xMax - xMin)) * 674;
  const y = (v: number) => 320 - ((v - yMin) / (yMax - yMin)) * 270;
  const front = points
    .filter((p) => p.pareto)
    .sort((a, b) => a.binding_mean - b.binding_mean);
  const ticks = Array.from({ length: 6 }, (_, i) => i / 5);
  return (
    <section className="panel landscape">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">THE OBJECTIVE SPACE</span>
          <h2>Find the next breakthrough</h2>
        </div>
        <button
          className="icon-button"
          title="Reset chart filters"
          aria-label="Reset chart filters"
          onClick={() => {
            setOnly(false);
            setUncertainty(false);
            setHover(null);
          }}
        >
          <Maximize2 size={16} />
        </button>
      </div>
      <div className="chart-legend">
        <button
          className={!only ? "legend active" : "legend"}
          onClick={() => setOnly(false)}
        >
          <i className="dot gray" />
          All variants
        </button>
        <button
          className={only ? "legend active" : "legend"}
          onClick={() => setOnly(!only)}
        >
          <i className="dot teal" />
          Selected 24
        </button>
        <span className="legend">
          <i className="line" />
          Pareto frontier
        </span>
        <label className="uncertainty">
          <input
            type="checkbox"
            checked={uncertainty}
            onChange={(e) => setUncertainty(e.target.checked)}
          />{" "}
          ±1σ selected
        </label>
      </div>
      <div className="scatter-wrap">
        <svg
          viewBox="0 0 800 380"
          role="img"
          aria-label="Interactive scatter plot of predicted binding versus stability"
        >
          <defs>
            <linearGradient id="plotFill" x1="0" y1="1" x2="1" y2="0">
              <stop offset="0%" stopColor="#fff" />
              <stop offset="100%" stopColor="#edf6ef" />
            </linearGradient>
          </defs>
          <rect x="68" y="50" width="674" height="270" fill="url(#plotFill)" />
          {ticks.map((t) => (
            <g key={t}>
              <line
                x1="68"
                x2="742"
                y1={y(yMin + t * (yMax - yMin))}
                y2={y(yMin + t * (yMax - yMin))}
                stroke="#e6ebe7"
                strokeDasharray="3 5"
              />
              <text
                x="55"
                y={y(yMin + t * (yMax - yMin)) + 4}
                textAnchor="end"
                className="axis"
              >
                {(yMin + t * (yMax - yMin)).toFixed(0)}
              </text>
              <text
                x={x(xMin + t * (xMax - xMin))}
                y="344"
                textAnchor="middle"
                className="axis"
              >
                {(xMin + t * (xMax - xMin)).toFixed(2)}
              </text>
            </g>
          ))}
          <line
            x1="68"
            x2="742"
            y1={y(result.config.min_stability)}
            y2={y(result.config.min_stability)}
            stroke="#d8a45f"
            strokeDasharray="5 5"
          />
          <text
            x="738"
            y={y(result.config.min_stability) - 7}
            textAnchor="end"
            className="threshold-label"
          >
            STABILITY FLOOR {result.config.min_stability}
          </text>
          <path
            d={front
              .map(
                (p, i) =>
                  `${i ? "L" : "M"} ${x(p.binding_mean)} ${y(p.stability_mean)}`,
              )
              .join(" ")}
            fill="none"
            stroke="#8ba780"
            strokeWidth="1.8"
            strokeDasharray="5 4"
          />
          {points
            .filter((p) => !p.selected && !only)
            .map((p) => (
              <circle
                key={p.variant_id}
                cx={x(p.binding_mean)}
                cy={y(p.stability_mean)}
                r={p.pareto ? 4.5 : 3.6}
                fill={
                  p.measured ? "#9a8a68" : p.eligible ? "#819991" : "#c4cbc5"
                }
                opacity={p.measured ? 0.55 : 0.4}
                onMouseEnter={() => setHover(p)}
                onMouseLeave={() => setHover(null)}
                onClick={() => inspect(p)}
                className="plot-dot"
              />
            ))}
          {points
            .filter((p) => p.selected)
            .map((p) => (
              <g
                key={p.variant_id}
                onMouseEnter={() => setHover(p)}
                onMouseLeave={() => setHover(null)}
                onClick={() => inspect(p)}
                className="plot-dot"
              >
                {uncertainty && (
                  <g stroke="#398879" strokeWidth="1" opacity=".5">
                    <line
                      x1={x(p.binding_mean - p.binding_std)}
                      x2={x(p.binding_mean + p.binding_std)}
                      y1={y(p.stability_mean)}
                      y2={y(p.stability_mean)}
                    />
                    <line
                      x1={x(p.binding_mean)}
                      x2={x(p.binding_mean)}
                      y1={y(p.stability_mean - p.stability_std)}
                      y2={y(p.stability_mean + p.stability_std)}
                    />
                  </g>
                )}
                <circle
                  cx={x(p.binding_mean)}
                  cy={y(p.stability_mean)}
                  r="7"
                  fill="#d7eee1"
                  opacity=".7"
                />
                <circle
                  cx={x(p.binding_mean)}
                  cy={y(p.stability_mean)}
                  r="4.5"
                  fill="#26816d"
                  stroke="#fff"
                  strokeWidth="1.5"
                />
              </g>
            ))}
          <text x="405" y="374" textAnchor="middle" className="axis-title">
            Predicted binding score →
          </text>
          <text
            transform="translate(18 190) rotate(-90)"
            textAnchor="middle"
            className="axis-title"
          >
            Adjusted stability score →
          </text>
          <text x="735" y="34" textAnchor="end" className="chart-hint">
            HIGHER IS BETTER ↗
          </text>
        </svg>
        {hover && (
          <div className="plot-tooltip">
            <b>{hover.variant_id}</b>
            <span>
              Binding {hover.binding_mean.toFixed(3)} ±{" "}
              {hover.binding_std.toFixed(3)}
            </span>
            <span>
              Stability {hover.stability_mean.toFixed(1)} ±{" "}
              {hover.stability_std.toFixed(1)}
            </span>
            <small>
              {hover.selected
                ? `Selected · ${hover.well}`
                : hover.measured
                  ? "Measured seed"
                  : hover.eligible
                    ? "Eligible candidate"
                    : hover.exclusions.join(", ")}
            </small>
          </div>
        )}
      </div>
      <div className="chart-foot">
        <span>
          <Crosshair size={13} />
          Click a variant to inspect its sequence
        </span>
        <span>{points.length} variants · 2 objectives</span>
      </div>
    </section>
  );
}
