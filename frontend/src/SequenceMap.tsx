import { useState } from "react";
import type { Result } from "./types";

export default function SequenceMap({ result }: { result: Result }) {
  const [chain, setChain] = useState(result.annotations[0].chain);
  const selected = result.variants
    .filter((v) => v.selected)
    .sort((a, b) => a.selection_rank! - b.selection_rank!);
  const annotation = result.annotations.find((a) => a.chain === chain)!;
  const mutationPositions = [
    ...new Set(
      selected.flatMap((v) =>
        v.mutations
          .filter((m) => m.chain === chain)
          .map((m) => `${m.position}:${m.insertion}`),
      ),
    ),
  ].sort(
    (a, b) =>
      Number(a.split(":")[0]) - Number(b.split(":")[0]) ||
      Number(a.split(":")[1]) - Number(b.split(":")[1]),
  );
  return (
    <div className="sequence-section">
      <div className="section-intro">
        <div>
          <h3>Every change, in context.</h3>
          <p>
            Selected variants against the parent. Columns show changed positions
            only; labels use parent coordinates.
          </p>
        </div>
        <div className="segmented">
          {result.annotations.map((a) => (
            <button
              key={a.chain}
              className={chain === a.chain ? "active" : ""}
              onClick={() => setChain(a.chain)}
            >
              {a.chain === "H" ? "Heavy chain" : "Light chain"}
            </button>
          ))}
        </div>
      </div>
      <div className="heatmap-scroll">
        <table className="heatmap">
          <thead>
            <tr>
              <th>Variant</th>
              {mutationPositions.map((pos) => (
                <th
                  key={pos}
                  title={`Parent position ${pos.replace(":0", "")}`}
                >
                  {pos.replace(":0", "").replace(":", "+")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {selected.map((v) => (
              <tr key={v.variant_id}>
                <td>{v.variant_id}</td>
                {mutationPositions.map((pos) => {
                  const [p, insertion] = pos.split(":").map(Number);
                  const m = v.mutations.find(
                    (m) =>
                      m.chain === chain &&
                      m.position === p &&
                      m.insertion === insertion,
                  );
                  return (
                    <td
                      key={pos}
                      title={m ? `${m.label} · ${m.region}` : "Unchanged"}
                      className={
                        m
                          ? m.to === "-"
                            ? "deletion"
                            : m.region.endsWith("FR")
                              ? "framework"
                              : "changed"
                          : ""
                      }
                    >
                      {m ? m.to : "·"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="region-tags">
        {annotation.regions.map((r) => (
          <span key={r.name}>
            {r.name}{" "}
            <b>
              {r.start}–{r.end}
            </b>
          </span>
        ))}
        <small>
          Approximate alignment · {(annotation.identity * 100).toFixed(0)}%
          reference identity · not IMGT/Kabat
        </small>
      </div>
    </div>
  );
}
