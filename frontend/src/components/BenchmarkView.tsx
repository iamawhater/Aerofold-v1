import { useEffect, useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  FlaskConical,
  Loader2,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import "./benchmark.css";

interface Dataset {
  id: string;
  name: string;
  description: string;
  variant_count: number;
  publication_url: string;
  source_url: string;
  processing: string;
  sequence_scope: string;
}
interface Config {
  dataset_id: string;
  rounds: number;
  batch_size: number;
  seed_size: number;
  seed: number;
  exploration: number;
  diversity: number;
  min_stability: number;
  confidence_beta: number;
}
interface Metrics {
  observed_count: number;
  top_hits: number;
  top_k_recall: number;
  enrichment: number;
  spearman_rho: number | null;
  hypervolume: number;
  feasible_candidates: number;
}
interface Round {
  round: number;
  evaluation_count: number;
  active: Metrics;
  random: Metrics;
}
interface Result {
  status: string;
  stop_reason: string | null;
  rounds: Round[];
  dataset: Dataset;
  config: Config;
  top_k: { actual_count_including_ties: number; cutoff_binding: number };
  summary?: {
    recall_multiplier: number | null;
    recall_difference_pp: number;
    correlation_difference: number | null;
    elapsed_seconds: number;
  };
}
interface Job {
  id: string;
  status: string;
  progress: string;
  config: Config;
  result: Result | null;
  error: string | null;
}
type Metric = "top_k_recall" | "spearman_rho" | "hypervolume";
const defaults: Config = {
  dataset_id: "titeseq_4_4_20",
  rounds: 5,
  batch_size: 24,
  seed_size: 20,
  seed: 42,
  exploration: 0.35,
  diversity: 0.55,
  min_stability: 20,
  confidence_beta: 1,
};
const pretty = (v: number | null | undefined, digits = 2) =>
  v == null ? "—" : v.toFixed(digits);
const signed = (v: number | null | undefined) =>
  v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(3)}`;

/** Paired-budget, public-data simulation; all results come from the backend oracle evaluator. */
export default function BenchmarkView() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [config, setConfig] = useState<Config>(defaults);
  const [job, setJob] = useState<Job | null>(null);
  const [jobId, setJobId] = useState(
    () => localStorage.getItem("aerofold-benchmark") || "",
  );
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [metric, setMetric] = useState<Metric>("top_k_recall");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const abort = new AbortController();
    api<Dataset[]>("/benchmark/datasets", { signal: abort.signal })
      .then(setDatasets)
      .catch((e) => {
        if (!abort.signal.aborted) setError(String(e.message));
      });
    return () => abort.abort();
  }, [retry]);
  useEffect(() => {
    if (!jobId) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await api<Job>(`/benchmark/runs/${jobId}`, {
          signal: abort.signal,
        });
        if (abort.signal.aborted) return;
        setJob(next);
        if (["running", "queued"].includes(next.status))
          timer = setTimeout(poll, 1000);
      } catch (e) {
        if (!abort.signal.aborted) setError((e as Error).message);
      }
    }
    void poll();
    return () => {
      abort.abort();
      clearTimeout(timer);
    };
  }, [jobId, retry]);
  const running =
    submitting || (!!job && ["queued", "running"].includes(job.status));
  const data = job?.result;
  const dataset = datasets.find((d) => d.id === config.dataset_id);
  const rounds = data?.rounds || [];
  const last = rounds.at(-1);
  const budget = config.seed_size + config.rounds * config.batch_size;
  const overBudget = !!dataset && budget > dataset.variant_count;
  async function run() {
    setSubmitting(true);
    setError("");
    try {
      const next = await api<Job>("/benchmark/run", {
        method: "POST",
        body: JSON.stringify(config),
      });
      setJob(next);
      setJobId(next.id);
      localStorage.setItem("aerofold-benchmark", next.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }
  function exportAudit() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(job, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `aerofold-benchmark-${job?.id.slice(0, 8)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const titles = {
    top_k_recall: "Top-binder discovery",
    spearman_rho: "Predictive rank correlation",
    hypervolume: "Cumulative Pareto hypervolume",
  };
  return (
    <div className="benchmark-view">
      {error && (
        <div className="alert error" role="alert">
          {error}
          <button
            onClick={() => {
              setError("");
              setRetry((v) => v + 1);
            }}
          >
            Retry connection
          </button>
        </div>
      )}
      {job?.error && (
        <div className="alert error" role="alert">
          {job.error}
        </div>
      )}
      <div className="benchmark-intro">
        <div>
          <span className="tag">
            <FlaskConical size={13} /> PUBLIC DMS BENCHMARK
          </span>
          <h2>Put the learning loop to the test.</h2>
          <p>
            Two agents. The same starting evidence. A fair assay budget.
            <br />
            Watch active learning and random selection discover a measured
            antibody landscape.
          </p>
        </div>
        <div className="benchmark-pair">
          <span className="active-key">AeroFold-Ab</span>
          <small>VS</small>
          <span className="random-key">Random baseline</span>
        </div>
      </div>
      <div className="benchmark-layout">
        <section
          className="benchmark-panel benchmark-controls"
          aria-labelledby="simulation-heading"
        >
          <h3 id="simulation-heading">Run new simulation</h3>
          <p className="benchmark-muted">
            Configure the next run. Each run starts fresh.
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void run();
            }}
          >
            <fieldset disabled={running}>
              <label>
                Public dataset
                <select
                  value={config.dataset_id}
                  onChange={(e) =>
                    setConfig({ ...config, dataset_id: e.target.value })
                  }
                >
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </label>
              <div className="benchmark-input-row">
                <label>
                  Rounds
                  <input
                    type="number"
                    required
                    min={1}
                    max={12}
                    value={config.rounds}
                    onChange={(e) =>
                      setConfig({ ...config, rounds: Number(e.target.value) })
                    }
                  />
                </label>
                <label>
                  Batch size
                  <select
                    value={config.batch_size}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        batch_size: Number(e.target.value),
                      })
                    }
                  >
                    {[12, 24, 48].map((v) => (
                      <option key={v}>{v}</option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="benchmark-input-row">
                <label>
                  Seed assays
                  <input
                    type="number"
                    required
                    min={6}
                    max={100}
                    value={config.seed_size}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        seed_size: Number(e.target.value),
                      })
                    }
                  />
                </label>
                <label>
                  Random seed
                  <input
                    type="number"
                    required
                    min={0}
                    max={2147483647}
                    value={config.seed}
                    onChange={(e) =>
                      setConfig({ ...config, seed: Number(e.target.value) })
                    }
                  />
                </label>
              </div>
              <label>
                Exploration <b>{Math.round(config.exploration * 100)}%</b>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={config.exploration}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      exploration: Number(e.target.value),
                    })
                  }
                />
              </label>
              <label>
                Diversity <b>{Math.round(config.diversity * 100)}%</b>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={config.diversity}
                  onChange={(e) =>
                    setConfig({ ...config, diversity: Number(e.target.value) })
                  }
                />
              </label>
              <details>
                <summary>Feasibility settings</summary>
                <label>
                  Expression-proxy floor
                  <input
                    type="number"
                    required
                    min={0}
                    max={100}
                    step={1}
                    value={config.min_stability}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        min_stability: Number(e.target.value),
                      })
                    }
                  />
                </label>
                <label>
                  Confidence β
                  <input
                    type="number"
                    required
                    min={0}
                    max={3}
                    step={0.1}
                    value={config.confidence_beta}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        confidence_beta: Number(e.target.value),
                      })
                    }
                  />
                </label>
              </details>
            </fieldset>
            <div
              className={
                overBudget ? "benchmark-budget invalid" : "benchmark-budget"
              }
            >
              <b>{budget} assays / agent</b>
              <span>
                {config.seed_size} seeds + {config.rounds} × {config.batch_size}{" "}
                selections
              </span>
              {overBudget && (
                <span role="alert">
                  Budget exceeds the {dataset?.variant_count}-variant library.
                </span>
              )}
            </div>
            <button
              className="button primary benchmark-run"
              disabled={running || !datasets.length || overBudget}
              type="submit"
            >
              {running ? (
                <Loader2 size={16} className="spin" />
              ) : (
                <FlaskConical size={16} />
              )}
              {running
                ? "Simulation running…"
                : job
                  ? "Run new simulation"
                  : "Run benchmark"}
              {!running && <ArrowRight size={16} />}
            </button>
          </form>
          <p className="benchmark-note">
            Benchmark batch size is configurable. Wet-lab selections remain
            exactly 24.
          </p>
        </section>
        <div className="benchmark-results">
          <section className="benchmark-panel benchmark-chart">
            <div className="benchmark-chart-heading">
              <div>
                <span className="eyebrow">CLOSED-LOOP PERFORMANCE</span>
                <h3>{titles[metric]}</h3>
              </div>
              <select
                aria-label="Chart metric"
                value={metric}
                onChange={(e) => setMetric(e.target.value as Metric)}
              >
                <option value="top_k_recall">Top-5% recall (%)</option>
                <option value="spearman_rho">Spearman ρ</option>
                <option value="hypervolume">Hypervolume</option>
              </select>
            </div>
            {job && (
              <div className="benchmark-progress" role="status">
                <span>{job.progress || job.status}</span>
                <small>
                  Seed {job.config.seed} · {job.config.seed_size} +{" "}
                  {job.config.rounds} × {job.config.batch_size} assays
                </small>
              </div>
            )}
            {rounds.length ? (
              <>
                <div className="benchmark-chart-canvas">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={rounds.map((r) => ({
                        round: r.round,
                        active: r.active[metric],
                        random: r.random[metric],
                      }))}
                      margin={{ top: 18, right: 25, left: 0, bottom: 20 }}
                    >
                      <CartesianGrid stroke="#e8ebe6" vertical={false} />
                      <XAxis
                        dataKey="round"
                        type="number"
                        domain={[0, Math.max(job?.config.rounds || 1, 1)]}
                        allowDecimals={false}
                        label={{
                          value: "Selection round",
                          position: "insideBottom",
                          offset: -14,
                        }}
                      />
                      <YAxis
                        width={50}
                        domain={
                          metric === "top_k_recall"
                            ? [0, 100]
                            : metric === "spearman_rho"
                              ? [-1, 1]
                              : [0, "auto"]
                        }
                        tickFormatter={(v) =>
                          metric === "top_k_recall"
                            ? `${v}%`
                            : Number(v).toFixed(1)
                        }
                      />
                      <Tooltip
                        formatter={(v) => pretty(Number(v), 3)}
                        labelFormatter={(v) => `Round ${v}`}
                      />
                      <Legend verticalAlign="top" height={38} />
                      <Line
                        name="AeroFold-Ab"
                        type="linear"
                        dataKey="active"
                        stroke="#217460"
                        strokeWidth={3}
                        dot={{ r: 4 }}
                        isAnimationActive={false}
                      />
                      <Line
                        name="Random baseline"
                        type="linear"
                        dataKey="random"
                        stroke="#bc8943"
                        strokeWidth={2}
                        strokeDasharray="5 4"
                        dot={{ r: 3 }}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="benchmark-note">
                  {metric === "top_k_recall"
                    ? `Recall of all ${data?.top_k.actual_count_including_ties} top binders, including cutoff ties. Round 0 is the shared seed set.`
                    : metric === "spearman_rho"
                      ? "Both agents are evaluated on the same still-unmeasured variants each round. This cohort changes across rounds; undefined correlations remain blank."
                      : "Hypervolume of revealed binding and expression measurements, with a fixed normalization and reference for the entire run. Higher is better."}
                </p>
              </>
            ) : (
              <div className="benchmark-empty">
                <FlaskConical size={36} />
                <h3>
                  {running
                    ? "Fitting the first pair of models"
                    : "Evidence before expectations."}
                </h3>
                <p>
                  {running
                    ? "Round-by-round results will appear here automatically."
                    : "Run a simulation to compare the two policies on real experimental measurements."}
                </p>
                <span>GP SURROGATES · MC-EHVI · DPP DIVERSITY</span>
              </div>
            )}
            {data?.stop_reason && (
              <div className="alert" role="status">
                Stopped early: {data.stop_reason}. Both agents retain equal
                assay budgets; the feasibility floor was preserved.
              </div>
            )}
          </section>
          <div className="benchmark-cards">
            <div className="benchmark-panel">
              <small>TOP-BINDER RECALL</small>
              <strong>
                {pretty(last?.active.top_k_recall, 1)}
                <em>%</em>
              </strong>
              <span>Random: {pretty(last?.random.top_k_recall, 1)}%</span>
            </div>
            <div className="benchmark-panel">
              <small>ENRICHMENT VS LIBRARY</small>
              <strong>
                {pretty(last?.active.enrichment)}
                <em>×</em>
              </strong>
              <span>Random: {pretty(last?.random.enrichment)}×</span>
            </div>
            <div className="benchmark-panel">
              <small>CORRELATION DIFFERENCE</small>
              <strong>
                {signed(
                  last &&
                    last.active.spearman_rho != null &&
                    last.random.spearman_rho != null
                    ? last.active.spearman_rho - last.random.spearman_rho
                    : null,
                )}
              </strong>
              <span>AeroFold ρ − Random ρ</span>
            </div>
          </div>
          {data?.summary && (
            <div className="benchmark-outcome">
              Final recall ratio:{" "}
              <b>{pretty(data.summary.recall_multiplier)}×</b> active / random ·{" "}
              {data.summary.recall_multiplier == null
                ? "Undefined when random finds no top binders"
                : `${signed(data.summary.recall_difference_pp)} percentage points`}{" "}
              · {pretty(data.summary.elapsed_seconds, 1)} s
            </div>
          )}
        </div>
      </div>
      <section className="benchmark-panel benchmark-protocol">
        <div>
          <h3>
            {data?.dataset.name || dataset?.name || "Public antibody landscape"}
          </h3>
          <p>{data?.dataset.description || dataset?.description}</p>
          <p>
            <b>
              Expression is a developability proxy, not measured structural
              stability.
            </b>{" "}
            This benchmark uses exact CDR mutation features and the joint
            binding-expression PyTorch GP, MC-EHVI and DPP implementations.
            The fragment-only dataset cannot support AbLang-2 scoring.
            Performance may favor either agent.
          </p>
        </div>
        <details>
          <summary>Protocol & source attribution</summary>
          <p>
            Each agent receives identical random seed assays, then reveals only
            its own selected batches. Random picks uniformly among its
            model-feasible candidates. Feasibility uses predicted expression
            mean − βσ; no hidden oracle prefilter is used.
          </p>
          <p>{dataset?.sequence_scope}</p>
          <p>{dataset?.processing}</p>
          <p>
            Enrichment = observed top-binder fraction / library top-binder
            fraction. The absolute top 5% includes all ties at the cutoff. One
            seed on one subset does not establish general superiority. Export
            the audit to inspect selected IDs, metric cohorts, runtime versions
            and the source checksum.
          </p>
          <a href={dataset?.publication_url} target="_blank" rel="noreferrer">
            Adams et al., eLife (2016)
          </a>{" "}
          ·{" "}
          <a href={dataset?.source_url} target="_blank" rel="noreferrer">
            Public Tite-Seq data
          </a>
        </details>
      </section>
      {!!rounds.length && (
        <section className="benchmark-panel benchmark-table">
          <div className="benchmark-chart-heading">
            <h3>Round-by-round evidence</h3>
            <button className="button secondary" onClick={exportAudit}>
              <ArrowDownToLine size={15} />
              Export audit JSON
            </button>
          </div>
          <div className="benchmark-table-scroll">
            <table>
              <caption>
                Metrics shown as AeroFold / Random. Correlation uses the shared
                unseen cohort.
              </caption>
              <thead>
                <tr>
                  <th>Round</th>
                  <th>Assays / agent</th>
                  <th>Recall (%)</th>
                  <th>Enrichment (×)</th>
                  <th>Spearman ρ</th>
                  <th>Hypervolume</th>
                  <th>Evaluation n</th>
                </tr>
              </thead>
              <tbody>
                {rounds.map((r) => (
                  <tr key={r.round}>
                    <td>{r.round === 0 ? "0 · Seed" : r.round}</td>
                    <td>{r.active.observed_count}</td>
                    <td>
                      {pretty(r.active.top_k_recall, 1)} /{" "}
                      {pretty(r.random.top_k_recall, 1)}
                    </td>
                    <td>
                      {pretty(r.active.enrichment)} /{" "}
                      {pretty(r.random.enrichment)}
                    </td>
                    <td>
                      {pretty(r.active.spearman_rho, 3)} /{" "}
                      {pretty(r.random.spearman_rho, 3)}
                    </td>
                    <td>
                      {pretty(r.active.hypervolume, 3)} /{" "}
                      {pretty(r.random.hypervolume, 3)}
                    </td>
                    <td>{r.evaluation_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
