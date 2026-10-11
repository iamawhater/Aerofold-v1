import { useEffect, useMemo, useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Database,
  Dna,
  FlaskConical,
  History,
  Loader2,
  Search,
  Settings2,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, token } from "./api";
import "./prediction.css";

type Task = {
  id: string;
  name: string;
  target: string;
  unit: string;
  category: string;
  maximize: boolean;
  ready: boolean;
  unique_sequences: number;
  license: string;
  publication_url: string;
  source_url: string;
  custom?: boolean;
};
type Catalog = {
  tasks: Task[];
  sources: { status: string }[];
  source_commit: string;
};
type Metrics = {
  spearman: number | null;
  rmse: number;
  top_recall: number;
  top_enrichment: number;
  random_recall: number;
  coverage?: number;
  top_k: number;
  test_count: number;
};
type Variant = {
  variant_id: string;
  heavy_sequence: string;
  light_sequence: string;
  prediction: number;
  lower: number;
  upper: number;
  rank: number;
  nearest_training_identity: number;
  support: string;
  used_as_labeled_data: boolean;
  observed_score: number | null;
};
type Card = {
  foundation_model: string | null;
  heads: Record<string, { weight: number; rmse: number; alpha: number }>;
  training: { fitting: number; calibration: number; total_labeled: number };
  nominal_coverage: number;
  uncertainty: string;
  limitations: string[];
};
type Predictor = "ensemble" | "esm2-only" | "antiberty-only";
const benchmarkModels = [
  { key: "ensemble", name: "AeroFold ensemble", color: "#217962" },
  { key: "esm2-only", name: "ESM-2-only · 8M", color: "#4965ad" },
  { key: "antiberty-only", name: "AntiBERTy-only", color: "#9b4e86" },
] as const;
type Curve = Partial<Record<Predictor | "additive" | "mean", Metrics>> & {
  training_size: number;
  repeats: number;
  recall_range?: number[];
  predictor_recall_ranges?: Partial<Record<Predictor, number[]>>;
};
type Result = {
  kind: string;
  task: Task;
  variants?: Variant[];
  model_card?: Card;
  evaluation?: Metrics;
  labeled_used?: number;
  labeled_available?: number;
  mode?: string;
  seconds: number;
  curves?: Curve[];
  protocol?: string;
  limitations?: string[];
  warnings?: string[];
  predictors?: Predictor[];
  encoder_provenance?: Partial<
    Record<Predictor, { model: string; revision: string }>
  >;
};
type Job = {
  id: string;
  kind: string;
  status: string;
  progress: string;
  error: string | null;
  created_at: string;
  result?: Result;
  config?: {
    task_id: string;
    representation: "ablang2" | "sequence";
    predictors?: Predictor[];
    train_limit: number;
    seed: number;
    repeats?: number;
    split?: string;
  };
};
type Page = "predict" | "benchmark" | "data" | "history";
const number = (n: number | null | undefined, digits = 3) =>
  n == null ? "—" : n.toFixed(digits);
const percent = (n: number | null | undefined) =>
  n == null ? "—" : `${(100 * n).toFixed(1)}%`;

export default function PredictionApp() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [page, setPage] = useState<Page>(
    window.location.hash === "#benchmark" ? "benchmark" : "predict",
  );
  const [taskId, setTaskId] = useState("titeseq_4420_binding");
  const [representation, setRepresentation] = useState("ablang2");
  const [benchmarkChoice, setBenchmarkChoice] = useState("ablang2");
  const [summaryPredictor, setSummaryPredictor] =
    useState<Predictor>("ensemble");
  const [limit, setLimit] = useState(768);
  const [seed, setSeed] = useState(42);
  const [repeats, setRepeats] = useState(3);
  const [split, setSplit] = useState("random");
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [history, setHistory] = useState<Job[]>([]);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [query, setQuery] = useState("");
  const [tablePage, setTablePage] = useState(0);
  const [inspect, setInspect] = useState<Variant | null>(null);
  const [upload, setUpload] = useState(false);
  const [settings, setSettings] = useState(false);
  const [auth, setAuth] = useState(token());
  const [metric, setMetric] = useState<"top_recall" | "spearman" | "rmse">(
    "top_recall",
  );
  const busy =
    submitting || job?.status === "queued" || job?.status === "running";
  const task = catalog?.tasks.find((t) => t.id === taskId);
  const result = job?.status === "completed" ? job.result : undefined;
  const prediction = result?.kind === "prediction" ? result : undefined;
  const benchmark =
    result?.kind === "prediction_benchmark" ? result : undefined;
  const last = benchmark?.curves?.at(-1);
  const displayedModels = benchmarkModels.filter((model) => last?.[model.key]);
  const summaryModel =
    displayedModels.find((model) => model.key === summaryPredictor) ||
    displayedModels[0];
  const summary = summaryModel && last?.[summaryModel.key];
  const recallRange =
    summaryModel &&
    (summaryModel.key === "ensemble"
      ? last?.recall_range
      : last?.predictor_recall_ranges?.[summaryModel.key]);
  const variants = useMemo(
    () =>
      prediction?.variants?.filter((v) =>
        v.variant_id.toLowerCase().includes(query.toLowerCase()),
      ) || [],
    [prediction, query],
  );

  async function refresh() {
    const [data, jobs] = await Promise.all([
      api<Catalog>("/prediction/tasks"),
      api<Job[]>("/prediction/jobs"),
    ]);
    setCatalog(data);
    setHistory(jobs);
  }
  function restoreBenchmarkSettings(saved: Job) {
    if (saved.kind !== "prediction_benchmark" || !saved.config) return;
    const config = saved.config;
    const predictors = config.predictors || ["ensemble"];
    setBenchmarkChoice(
      predictors.length > 1
        ? "compare"
        : predictors[0] === "ensemble"
          ? config.representation
          : predictors[0],
    );
    setRepresentation(config.representation);
    setTaskId(config.task_id);
    setLimit(config.train_limit);
    setSeed(config.seed);
    setRepeats(config.repeats || 3);
    setSplit(config.split || "random");
  }
  useEffect(() => {
    refresh().catch((e) => setError(e.message));
    const saved = localStorage.getItem("aerofold-prediction-job");
    if (saved)
      api<Job>(`/prediction/jobs/${saved}`)
        .then((saved) => {
          setJob(saved);
          restoreBenchmarkSettings(saved);
        })
        .catch(() => localStorage.removeItem("aerofold-prediction-job"));
  }, []);
  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    let cancelled = false;
    const timer = setTimeout(
      () =>
        api<Job>(`/prediction/jobs/${job.id}`)
          .then((next) => {
            if (cancelled) return;
            setJob(next);
            if (next.status === "failed")
              setError(next.error || "Model job failed");
            if (next.status === "completed")
              refresh().catch((e) => setError(e.message));
          })
          .catch((e) => {
            if (!cancelled) {
              setError(e.message);
              setJob({ ...job });
            }
          }),
      1800,
    );
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [job]);

  function navigate(next: Page) {
    setPage(next);
    window.location.hash = next === "predict" ? "predict" : next;
  }
  async function start() {
    setError("");
    setSubmitting(true);
    setTablePage(0);
    try {
      const config = {
        task_id: taskId,
        representation,
        train_limit: limit,
        seed,
      };
      let next: Job;
      if (page === "benchmark") {
        next = await api<Job>("/prediction/benchmark/run", {
          method: "POST",
          body: JSON.stringify({
            ...config,
            representation:
              benchmarkChoice === "ablang2" || benchmarkChoice === "sequence"
                ? benchmarkChoice
                : representation,
            predictors:
              benchmarkChoice === "compare"
                ? ["ensemble", "esm2-only", "antiberty-only"]
                : benchmarkChoice === "esm2-only" ||
                    benchmarkChoice === "antiberty-only"
                  ? [benchmarkChoice]
                  : ["ensemble"],
            train_limit: limit || 12000,
            repeats,
            split,
            test_limit: 512,
            top_k: 24,
          }),
        });
      } else if (file) {
        const form = new FormData();
        form.set("file", file);
        form.set("config", JSON.stringify(config));
        next = await api<Job>("/prediction/upload", {
          method: "POST",
          body: form,
        });
      } else {
        next = await api<Job>("/prediction/run", {
          method: "POST",
          body: JSON.stringify(config),
        });
      }
      setJob(next);
      localStorage.setItem("aerofold-prediction-job", next.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }
  async function exportResult(format: "json" | "csv") {
    if (!job) return;
    try {
      const response = await fetch(
        `/api/prediction/jobs/${job.id}/export?format=${format}`,
        { headers: token() ? { Authorization: `Bearer ${token()}` } : {} },
      );
      if (!response.ok) throw new Error("Export failed");
      const url = URL.createObjectURL(await response.blob());
      const a = document.createElement("a");
      a.href = url;
      a.download = `aerofold-${job.id.slice(0, 8)}.${format}`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function openJob(id: string) {
    try {
      const next = await api<Job>(`/prediction/jobs/${id}`);
      setJob(next);
      restoreBenchmarkSettings(next);
      navigate(next.kind === "prediction" ? "predict" : "benchmark");
      localStorage.setItem("aerofold-prediction-job", id);
      setTablePage(0);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function importTraining(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    const form = new FormData(event.currentTarget);
    try {
      const created = await api<Task>("/prediction/tasks", {
        method: "POST",
        body: form,
      });
      await refresh();
      setTaskId(created.id);
      setUpload(false);
      navigate("predict");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="prediction-app">
      <aside className="pred-sidebar">
        <a
          className="pred-brand"
          href="#predict"
          onClick={() => navigate("predict")}
        >
          <span>
            <Dna size={26} />
          </span>
          <b>
            AeroFold<span>-Ab</span>
          </b>
        </a>
        <div className="pred-workspace">
          <span className="pred-dot" /> Antibody intelligence{" "}
          <small>RESEARCH WORKSPACE</small>
        </div>
        <p className="pred-nav-label">WORKSPACE</p>
        {(
          [
            { id: "predict", name: "Prediction engine", icon: Dna },
            { id: "benchmark", name: "Benchmark", icon: FlaskConical },
            { id: "data", name: "Data & evidence", icon: Database },
            { id: "history", name: "Run history", icon: History },
          ] as const
        ).map((item) => (
          <button
            key={item.id}
            className={`pred-nav ${page === item.id ? "active" : ""}`}
            onClick={() => navigate(item.id)}
          >
            <item.icon size={18} />
            {item.name}
            {page === item.id && <span className="pred-dot" />}
          </button>
        ))}
        <div className="pred-sidebar-note">
          <ShieldCheck size={22} />
          <b>Target-specific by design</b>
          <p>
            Experimental labels, independent tests, and traceable predictions.
          </p>
          <button onClick={() => navigate("data")}>
            Explore the evidence <ArrowRight size={14} />
          </button>
        </div>
        <button className="pred-nav" onClick={() => setSettings(true)}>
          <Settings2 size={17} /> Connection settings
        </button>
        <a className="pred-legacy" href="?legacy=1">
          Open previous workspace ↗
        </a>
      </aside>
      <main className="pred-main">
        <header className="pred-topbar">
          <span>
            Workspace <span>/</span>{" "}
            {page === "predict"
              ? "Prediction engine"
              : page === "data"
                ? "Data & evidence"
                : page === "history"
                  ? "Run history"
                  : "Benchmark"}
          </span>
          <span className="pred-badge">
            <span className="pred-dot" /> SUPERVISED PREDICTION
          </span>
        </header>
        <div className="pred-content">
          <div className="pred-heading">
            <div>
              <p className="pred-eyebrow">AEROFOLD ANTIBODY INTELLIGENCE</p>
              <h1>
                {page === "predict"
                  ? "Find the variants worth making."
                  : page === "benchmark"
                    ? "Measure the model. Trust the evidence."
                    : page === "data"
                      ? "Every prediction starts with evidence."
                      : "Your modeling history."}
              </h1>
              <p className="pred-muted">
                {page === "predict"
                  ? "Predict assay performance for every candidate, with uncertainty and a clear view of where the model applies."
                  : page === "benchmark"
                    ? "Compare assay-trained predictors on unseen experimental sequences and inspect their ranking performance."
                    : page === "data"
                      ? "Separate targets, explicit assay units, published variable domains, and traceable source licenses."
                      : "Reopen saved predictions, benchmark results, and their model records."}
              </p>
            </div>
            <button
              className="pred-button secondary"
              onClick={() => setUpload(true)}
            >
              <Upload size={16} /> Add assay data
            </button>
          </div>
          {error && (
            <div role="alert" className="pred-error">
              {error}
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {(page === "predict" || page === "benchmark") && (
            <div className="pred-layout">
              <section className="pred-panel pred-controls">
                <h3>
                  {page === "benchmark"
                    ? "Run a held-out benchmark"
                    : "Configure your prediction"}
                </h3>
                <p className="pred-muted small">
                  Each model learns one target and one assay.
                </p>
                <fieldset disabled={busy}>
                  <label>
                    Target & experimental assay
                    <select
                      value={taskId}
                      onChange={(e) => setTaskId(e.target.value)}
                    >
                      {catalog?.tasks.map((t) => (
                        <option key={t.id} value={t.id} disabled={!t.ready}>
                          {t.name}
                          {!t.ready ? " · insufficient data" : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                  {task && (
                    <div className="pred-assay-info">
                      <b>
                        {task.unique_sequences.toLocaleString()} unique
                        sequences
                      </b>
                      <span>
                        {task.unit} · {task.maximize ? "higher" : "lower"} is
                        better
                      </span>
                      <span>
                        {task.category === "expression"
                          ? "Expression is a developability proxy, not structural stability."
                          : task.target}
                      </span>
                    </div>
                  )}
                  <label>
                    Model representation
                    <select
                      value={
                        page === "benchmark" ? benchmarkChoice : representation
                      }
                      onChange={(e) =>
                        page === "benchmark"
                          ? setBenchmarkChoice(e.target.value)
                          : setRepresentation(e.target.value)
                      }
                    >
                      <option value="ablang2">
                        AbLang-2 + sequence ensemble
                      </option>
                      <option value="sequence">
                        Sequence ensemble · offline / fast
                      </option>
                      {page === "benchmark" && (
                        <>
                          <option value="esm2-only">
                            ESM-2-only · 8M + ridge
                          </option>
                          <option value="antiberty-only">
                            AntiBERTy-only + ridge
                          </option>
                          <option value="compare">
                            Compare all three predictors
                          </option>
                        </>
                      )}
                    </select>
                  </label>
                  {page === "benchmark" && benchmarkChoice === "compare" && (
                    <label>
                      Ensemble representation
                      <select
                        value={representation}
                        onChange={(e) => setRepresentation(e.target.value)}
                      >
                        <option value="ablang2">
                          AbLang-2 + sequence ensemble
                        </option>
                        <option value="sequence">
                          Sequence ensemble · offline / fast
                        </option>
                      </select>
                    </label>
                  )}
                  <p className="pred-help">
                    {page === "benchmark" &&
                    ["esm2-only", "antiberty-only", "compare"].includes(
                      benchmarkChoice,
                    )
                      ? "Frozen foundation embeddings with assay-trained ridge heads and separate calibration. First-time download and encoding may take several minutes. Comparisons use identical splits and labeled budgets."
                      : (page === "benchmark"
                            ? benchmarkChoice
                            : representation) === "ablang2"
                        ? "Pretrained antibody embeddings plus assay-trained prediction heads. First-time encoding may take several minutes."
                        : "Additive, interaction, local-kernel and tree models. No foundation-model embeddings."}
                  </p>
                  <div className="pred-two">
                    <label>
                      Labeled-data budget
                      <select
                        value={limit}
                        onChange={(e) => setLimit(Number(e.target.value))}
                      >
                        {[128, 256, 768, 2048, 4096, 12000].map((n) => (
                          <option key={n} value={n}>
                            {n.toLocaleString()}
                          </option>
                        ))}
                        {page !== "benchmark" && (
                          <option value={0}>All available</option>
                        )}
                      </select>
                    </label>
                    <label>
                      Random seed
                      <input
                        type="number"
                        min={0}
                        max={2147483636}
                        value={seed}
                        onChange={(e) => setSeed(Number(e.target.value))}
                      />
                    </label>
                  </div>
                  {page === "benchmark" ? (
                    <>
                      <label>
                        Evaluation split
                        <select
                          value={split}
                          onChange={(e) => setSplit(e.target.value)}
                        >
                          <option value="random">
                            Unseen variants · random split
                          </option>
                          <option value="mutation_extrapolation">
                            Higher mutation count · stress test
                          </option>
                        </select>
                      </label>
                      <label>
                        Repeated splits
                        <select
                          value={repeats}
                          onChange={(e) => setRepeats(Number(e.target.value))}
                        >
                          {[1, 3, 5, 10].map((n) => (
                            <option key={n} value={n}>
                              {n} {n === 1 ? "seed" : "seeds"}
                            </option>
                          ))}
                        </select>
                      </label>
                      <p className="pred-help">
                        Three training budgets, up to 512 held-out sequences per
                        seed. Exact sequence duplicates never cross the split.
                      </p>
                    </>
                  ) : (
                    <>
                      <label className="pred-file">
                        <Upload size={20} />
                        <b>{file?.name || "Upload candidate sequences"}</b>
                        <span>
                          CSV · heavy_sequence, light_sequence, variant_id
                        </span>
                        <input
                          aria-label="Candidate sequence CSV"
                          type="file"
                          accept=".csv"
                          onChange={(e) => setFile(e.target.files?.[0] || null)}
                        />
                      </label>
                      {file ? (
                        <button
                          className="pred-text-button"
                          onClick={() => setFile(null)}
                        >
                          Clear file · use public held-out demo
                        </button>
                      ) : (
                        <p className="pred-help">
                          No file? Run a demo on public sequences excluded from
                          model development.
                        </p>
                      )}
                    </>
                  )}
                  <button
                    className="pred-button primary pred-run"
                    disabled={!task?.ready || busy}
                    onClick={start}
                  >
                    {busy ? (
                      <Loader2 className="spin" size={17} />
                    ) : (
                      <FlaskConical size={17} />
                    )}
                    {page === "benchmark"
                      ? "Run benchmark"
                      : "Predict & rank variants"}
                    <ArrowRight size={17} />
                  </button>
                </fieldset>
                <div className="pred-method-note">
                  <ShieldCheck size={15} />
                  {page === "benchmark"
                    ? "Validation selects prediction heads. Separate calibration data set the prediction intervals."
                    : "Validation chooses the model blend. Separate calibration data set the prediction intervals."}
                </div>
              </section>
              <div className="pred-results">
                {busy && (
                  <div className="pred-panel pred-progress" role="status">
                    <Loader2 className="spin" size={22} />
                    <div>
                      <b>{job?.progress || "Preparing model job…"}</b>
                      <p className="pred-muted small">
                        You can leave this page; the server keeps the job
                        running. Results are saved automatically.
                      </p>
                    </div>
                  </div>
                )}
                {page === "predict" && prediction ? (
                  <>
                    <div className="pred-result-title">
                      <div>
                        <p className="pred-eyebrow">PREDICTION COMPLETE</p>
                        <h2>{prediction.task.name}</h2>
                        <p className="pred-muted small">
                          {prediction.labeled_used?.toLocaleString()} /{" "}
                          {prediction.labeled_available?.toLocaleString()}{" "}
                          available labels used · {prediction.variants?.length}{" "}
                          candidates · {number(prediction.seconds, 1)} s
                        </p>
                      </div>
                      <button
                        className="pred-button secondary"
                        onClick={() => exportResult("csv")}
                      >
                        <ArrowDownToLine size={16} /> Export rankings
                      </button>
                    </div>
                    <div className="pred-stat-grid">
                      <Stat
                        label="HELD-OUT SPEARMAN ρ"
                        value={number(prediction.evaluation?.spearman)}
                        detail={
                          prediction.mode === "held_out_demo"
                            ? "Correlation with unseen assay measurements"
                            : "Upload predictions are not a benchmark"
                        }
                      />
                      <Stat
                        label="TOP-5% ENRICHMENT"
                        value={
                          prediction.evaluation
                            ? `${number(prediction.evaluation.top_enrichment, 2)}×`
                            : "—"
                        }
                        detail="Top 24 ranking vs expected random ranking"
                      />
                      <Stat
                        label="90% INTERVAL COVERAGE"
                        value={percent(prediction.evaluation?.coverage)}
                        detail="Measured coverage on the held-out demo"
                      />
                    </div>
                    <section className="pred-panel">
                      <div className="pred-section-title">
                        <div>
                          <p className="pred-eyebrow">CANDIDATE RANKING</p>
                          <h3>
                            Predicted {prediction.task.category} performance
                          </h3>
                        </div>
                        <div className="pred-search">
                          <Search size={15} />
                          <input
                            aria-label="Search variants"
                            value={query}
                            onChange={(e) => {
                              setQuery(e.target.value);
                              setTablePage(0);
                            }}
                            placeholder="Find variant…"
                          />
                        </div>
                      </div>
                      <p className="pred-help">
                        {prediction.task.unit} ·{" "}
                        {prediction.task.maximize ? "Highest" : "Lowest"}{" "}
                        predicted scores first. Sequences outside the training
                        neighborhood appear last. Click a row to inspect
                        sequences.
                      </p>
                      <div className="pred-table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Rank / variant</th>
                              <th>Prediction</th>
                              <th>90% interval</th>
                              <th>Nearest identity</th>
                              <th>Support</th>
                            </tr>
                          </thead>
                          <tbody>
                            {variants
                              .slice(tablePage * 15, (tablePage + 1) * 15)
                              .map((v) => (
                                <tr
                                  key={v.variant_id}
                                  tabIndex={0}
                                  onClick={() => setInspect(v)}
                                  onKeyDown={(e) =>
                                    e.key === "Enter" && setInspect(v)
                                  }
                                >
                                  <td>
                                    <span className="pred-rank">{v.rank}</span>
                                    <b>{v.variant_id}</b>
                                    {v.used_as_labeled_data && (
                                      <small>Included in labeled data</small>
                                    )}
                                  </td>
                                  <td className="pred-score">
                                    {number(v.prediction)}
                                  </td>
                                  <td>
                                    {number(v.lower, 2)} – {number(v.upper, 2)}
                                  </td>
                                  <td>
                                    {percent(v.nearest_training_identity)}
                                  </td>
                                  <td>
                                    <span
                                      className={`pred-chip ${v.support.startsWith("outside") ? "amber" : ""}`}
                                    >
                                      {v.support.startsWith("outside")
                                        ? "Extrapolation"
                                        : "Near training data"}
                                    </span>
                                  </td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                      <div className="pred-pagination">
                        <span>{variants.length} variants</span>
                        <button
                          disabled={tablePage === 0}
                          aria-label="Previous page"
                          onClick={() => setTablePage(tablePage - 1)}
                        >
                          <ChevronLeft size={16} />
                        </button>
                        <span>
                          {tablePage + 1} /{" "}
                          {Math.max(1, Math.ceil(variants.length / 15))}
                        </span>
                        <button
                          disabled={(tablePage + 1) * 15 >= variants.length}
                          aria-label="Next page"
                          onClick={() => setTablePage(tablePage + 1)}
                        >
                          <ChevronRight size={16} />
                        </button>
                      </div>
                    </section>
                    {prediction.evaluation && (
                      <section className="pred-panel">
                        <p className="pred-eyebrow">OUT-OF-SAMPLE CHECK</p>
                        <h3>Predicted vs measured</h3>
                        <div className="pred-chart">
                          <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart
                              margin={{
                                top: 16,
                                right: 25,
                                bottom: 28,
                                left: 10,
                              }}
                            >
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis
                                type="number"
                                dataKey="observed_score"
                                name="Measured"
                                domain={["auto", "auto"]}
                                label={{
                                  value: "Measured assay score",
                                  position: "bottom",
                                  offset: 10,
                                }}
                              />
                              <YAxis
                                type="number"
                                dataKey="prediction"
                                name="Predicted"
                                domain={["auto", "auto"]}
                              />
                              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                              <Scatter
                                name="Held-out sequences"
                                data={prediction.variants}
                                fill="#28745e"
                                fillOpacity={0.45}
                              />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </div>
                        <p className="pred-help">
                          Each point is an unseen sequence. RMSE:{" "}
                          {number(prediction.evaluation.rmse)}{" "}
                          {prediction.task.unit}. This is one split, not
                          evidence of transfer to a new target.
                        </p>
                      </section>
                    )}
                    {prediction.model_card && (
                      <ModelCard
                        card={prediction.model_card}
                        onExport={() => exportResult("json")}
                      />
                    )}
                  </>
                ) : page === "benchmark" &&
                  benchmark &&
                  last &&
                  summary &&
                  summaryModel ? (
                  <>
                    <div className="pred-result-title">
                      <div>
                        <p className="pred-eyebrow">INDEPENDENT TEST RESULTS</p>
                        <h2>{benchmark.task.name}</h2>
                        <p className="pred-muted small">
                          {last.repeats} {last.repeats === 1 ? "seed" : "seeds"}{" "}
                          · {summary.test_count} test sequences / seed ·{" "}
                          {number(benchmark.seconds, 1)} s
                        </p>
                      </div>
                      <button
                        className="pred-button secondary"
                        onClick={() => exportResult("json")}
                      >
                        <ArrowDownToLine size={16} /> Export full audit
                      </button>
                    </div>
                    <section className="pred-panel">
                      <div className="pred-section-title">
                        <div>
                          <p className="pred-eyebrow">
                            SUPERVISED LEARNING CURVE
                          </p>
                          <h3>More measurements. Measurable performance.</h3>
                        </div>
                        <select
                          aria-label="Benchmark metric"
                          value={metric}
                          onChange={(e) =>
                            setMetric(e.target.value as typeof metric)
                          }
                        >
                          <option value="top_recall">Top-5% recall</option>
                          <option value="spearman">Spearman correlation</option>
                          <option value="rmse">RMSE · lower is better</option>
                        </select>
                      </div>
                      <div className="pred-chart large">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            data={benchmark.curves?.map((c) => ({
                              labels: c.training_size,
                              ...Object.fromEntries(
                                displayedModels.map((model) => [
                                  model.key,
                                  c[model.key]?.[metric],
                                ]),
                              ),
                              additive: c.additive?.[metric],
                              random:
                                metric === "top_recall"
                                  ? c[summaryModel.key]?.random_recall
                                  : undefined,
                            }))}
                            margin={{ top: 24, right: 20, left: 5, bottom: 25 }}
                          >
                            <CartesianGrid vertical={false} stroke="#e6ebe4" />
                            <XAxis
                              dataKey="labels"
                              label={{
                                value:
                                  "Labeled sequences available for development",
                                position: "bottom",
                                offset: 10,
                              }}
                            />
                            <YAxis
                              tickFormatter={(v) =>
                                metric === "top_recall"
                                  ? percent(v)
                                  : number(v, 2)
                              }
                            />
                            <Tooltip
                              formatter={(v) =>
                                metric === "top_recall"
                                  ? percent(Number(v))
                                  : number(Number(v))
                              }
                            />
                            <Legend
                              verticalAlign="top"
                              height={displayedModels.length > 1 ? 72 : 40}
                            />
                            {displayedModels.map((model) => (
                              <Line
                                key={model.key}
                                name={model.name}
                                type="linear"
                                dataKey={model.key}
                                stroke={model.color}
                                strokeWidth={3}
                                dot={{ r: 5 }}
                              />
                            ))}
                            {last.additive && (
                              <Line
                                name="Additive regression"
                                type="linear"
                                dataKey="additive"
                                stroke="#758bab"
                                strokeWidth={2}
                                dot={{ r: 4 }}
                              />
                            )}
                            {metric === "top_recall" && (
                              <Line
                                name="Expected random ranking"
                                dataKey="random"
                                stroke="#c6904a"
                                strokeWidth={2}
                                strokeDasharray="5 5"
                                dot={false}
                              />
                            )}
                            {metric === "spearman" && (
                              <ReferenceLine
                                y={0}
                                stroke="#c6904a"
                                strokeDasharray="5 5"
                              />
                            )}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <p className="pred-help">
                        Same hidden test sequences across budgets within each
                        seed. Curves show seed means; no test labels select
                        models. Random recall is the exact fixed-budget
                        expectation.
                      </p>
                    </section>
                    {displayedModels.length > 1 && (
                      <label>
                        Summary predictor
                        <select
                          value={summaryModel.key}
                          onChange={(e) =>
                            setSummaryPredictor(e.target.value as Predictor)
                          }
                        >
                          {displayedModels.map((model) => (
                            <option key={model.key} value={model.key}>
                              {model.name}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    <div className="pred-stat-grid">
                      <Stat
                        label="FINAL ENRICHMENT"
                        value={`${number(summary.top_enrichment, 2)}×`}
                        detail={`${summaryModel.name} · expected random: 1×`}
                      />
                      <Stat
                        label="SPEARMAN CORRELATION"
                        value={number(summary.spearman)}
                        detail={
                          last.additive
                            ? `Additive baseline: ${number(last.additive.spearman)}`
                            : summaryModel.name
                        }
                      />
                      <Stat
                        label="90% INTERVAL COVERAGE"
                        value={percent(summary.coverage)}
                        detail={`RMSE: ${number(summary.rmse)} ${benchmark.task.unit}`}
                      />
                    </div>
                    <section className="pred-panel">
                      <h3>What this benchmark establishes</h3>
                      <p className="pred-muted">{benchmark.protocol}</p>
                      {recallRange && (
                        <p className="pred-help">
                          {summaryModel.name} · final top-binder recall range
                          across seeds: {percent(recallRange[0])}–
                          {percent(recallRange[1])}. This range is descriptive,
                          not a confidence interval.
                        </p>
                      )}
                      {Object.entries(benchmark.encoder_provenance || {}).map(
                        ([key, encoder]) => (
                          <p className="pred-help" key={key}>
                            {
                              benchmarkModels.find((model) => model.key === key)
                                ?.name
                            }
                            : {encoder.model}. Frozen embeddings + supervised
                            ridge; no sequence descriptors or ensemble blending.
                            Checkpoint revision: {encoder.revision}.
                          </p>
                        ),
                      )}
                      <ul className="pred-limitations">
                        {benchmark.limitations?.map((text) => (
                          <li key={text}>{text}</li>
                        ))}
                      </ul>
                    </section>
                  </>
                ) : (
                  <section className="pred-panel pred-empty">
                    <div className="pred-orbit">
                      <Dna size={46} />
                    </div>
                    <p className="pred-eyebrow">
                      {page === "benchmark"
                        ? "EVIDENCE BEFORE CLAIMS"
                        : "FROM SEQUENCE TO ASSAY PREDICTION"}
                    </p>
                    <h2>
                      {page === "benchmark"
                        ? "Give the predictor an honest test."
                        : "A prediction for every variant."}
                    </h2>
                    <p>
                      {page === "benchmark"
                        ? "Run repeated held-out evaluations, inspect the strong baseline, and see where the model succeeds or fails."
                        : "Choose a target-specific assay and upload candidates, or try a ready-to-run public-data demonstration."}
                    </p>
                    <div className="pred-empty-steps">
                      <span>
                        <Database size={17} /> Experimental data
                      </span>
                      <ArrowRight size={15} />
                      <span>
                        <Dna size={17} /> Assay-trained ensemble
                      </span>
                      <ArrowRight size={15} />
                      <span>
                        <ShieldCheck size={17} /> Calibrated predictions
                      </span>
                    </div>
                  </section>
                )}
              </div>
            </div>
          )}
          {page === "data" && (
            <>
              <div className="pred-stat-grid">
                <Stat
                  label="PACKAGED ASSAY RECORDS"
                  value={(
                    catalog?.tasks
                      .filter((t) => !t.custom)
                      .reduce((sum, t) => sum + t.unique_sequences, 0) || 0
                  ).toLocaleString()}
                  detail="Sequence–assay pairs; sequences can occur across tasks"
                />
                <Stat
                  label="READY TASKS"
                  value={String(
                    catalog?.tasks.filter((t) => t.ready).length || 0,
                  )}
                  detail={`${catalog?.tasks.length || 0} task definitions, with small tasks kept as references`}
                />
                <Stat
                  label="SOURCE CATALOG"
                  value={String(catalog?.sources.length || 0)}
                  detail="Admission and license status recorded for each source"
                />
              </div>
              <section className="pred-panel">
                <p className="pred-eyebrow">PUBLIC EVIDENCE LIBRARY</p>
                <h3>One assay. One interpretable prediction.</h3>
                <p className="pred-muted">
                  Reviewed task adapters normalize documented units and merge
                  exact-sequence replicates by their median. Binding labels from
                  different targets are never pooled.
                </p>
                <div className="pred-table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Assay / target</th>
                        <th>Unique sequences</th>
                        <th>Native prediction units</th>
                        <th>License / status</th>
                        <th>Evidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {catalog?.tasks.map((t) => (
                        <tr key={t.id}>
                          <td>
                            <b>{t.name}</b>
                            <small>{t.target}</small>
                          </td>
                          <td>{t.unique_sequences.toLocaleString()}</td>
                          <td>{t.unit}</td>
                          <td>
                            {t.license}
                            <small>
                              {t.ready
                                ? "Ready for modeling"
                                : "Too small for independent model validation"}
                            </small>
                          </td>
                          <td>
                            {t.publication_url && (
                              <a
                                href={t.publication_url}
                                target="_blank"
                                rel="noreferrer"
                              >
                                Publication ↗
                              </a>
                            )}
                            {t.ready && (
                              <button
                                className="pred-text-button"
                                onClick={() => {
                                  setTaskId(t.id);
                                  navigate("predict");
                                }}
                              >
                                Use assay →
                              </button>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="pred-help">
                  FLAb snapshot: {catalog?.source_commit}. Catalog data do not
                  imply universal binding accuracy. Noncommercial,
                  unspecified-license, gated, and unreviewed sources are
                  excluded from model training. Full source and processing
                  records are available from the API and result audit.
                </p>
              </section>
            </>
          )}
          {page === "history" && (
            <section className="pred-panel">
              <h3>Saved model jobs</h3>
              {!history.length && (
                <p className="pred-muted">
                  Run a prediction or benchmark to create your first result.
                </p>
              )}
              <div className="pred-history">
                {history.map((j) => (
                  <button key={j.id} onClick={() => openJob(j.id)}>
                    <span>
                      {j.status === "completed" ? (
                        <CheckCircle2 size={20} />
                      ) : (
                        <History size={20} />
                      )}
                    </span>
                    <div>
                      <b>
                        {j.kind === "prediction"
                          ? "Variant prediction"
                          : "Held-out benchmark"}
                      </b>
                      <small>
                        {new Date(j.created_at).toLocaleString()} ·{" "}
                        {j.id.slice(0, 8)}
                      </small>
                    </div>
                    <span className="pred-chip">{j.status}</span>
                    <ArrowRight size={17} />
                  </button>
                ))}
              </div>
            </section>
          )}
          <footer className="pred-footer">
            <span>
              <Dna size={14} /> AeroFold-Ab · Research prediction engine
            </span>
            <span>
              Published measurements. Explicit limits. Reproducible tests.
            </span>
          </footer>
        </div>
      </main>
      {inspect && (
        <div className="pred-modal-backdrop">
          <section
            className="pred-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Variant details"
          >
            <button
              className="pred-close"
              aria-label="Close variant"
              onClick={() => setInspect(null)}
            >
              <X size={19} />
            </button>
            <p className="pred-eyebrow">VARIANT DETAILS</p>
            <h2>{inspect.variant_id}</h2>
            <p>
              Predicted score: <b>{number(inspect.prediction)}</b> · interval{" "}
              {number(inspect.lower)}–{number(inspect.upper)}
            </p>
            <p className="pred-muted">
              {inspect.support.startsWith("outside")
                ? "Outside the model's sequence neighborhood. This prediction needs additional target-specific validation."
                : "Within the model's positional sequence neighborhood; this does not guarantee binding."}
            </p>
            {inspect.observed_score !== null && (
              <p>
                Published assay measurement: {number(inspect.observed_score)}
                {inspect.used_as_labeled_data
                  ? " · included in model development/calibration"
                  : " · excluded from labeled model data"}
              </p>
            )}
            <label>
              Heavy variable domain<code>{inspect.heavy_sequence}</code>
            </label>
            {inspect.light_sequence && (
              <label>
                Light variable domain<code>{inspect.light_sequence}</code>
              </label>
            )}
          </section>
        </div>
      )}
      {upload && (
        <div className="pred-modal-backdrop">
          <form className="pred-modal" onSubmit={importTraining}>
            <button
              type="button"
              className="pred-close"
              aria-label="Close upload"
              onClick={() => setUpload(false)}
            >
              <X size={19} />
            </button>
            <p className="pred-eyebrow">YOUR TARGET. YOUR ASSAY.</p>
            <h2>Add experimental training data</h2>
            <p className="pred-muted">
              At least 60 unique full-domain sequence pairs. CSV columns:
              heavy_sequence, optional light_sequence and variant_id, and score.
              Use one target and one assay protocol per upload.
            </p>
            <label>
              Assay name
              <input
                name="name"
                required
                maxLength={120}
                placeholder="Program A · SPR binding"
              />
            </label>
            <label>
              Target and scaffold family
              <input
                name="target"
                required
                maxLength={200}
                placeholder="Antigen / antibody family"
              />
            </label>
            <div className="pred-two">
              <label>
                Score units
                <input
                  name="unit"
                  required
                  maxLength={80}
                  placeholder="e.g. pKd (M), Tm °C"
                />
              </label>
              <label>
                Preferred direction
                <select name="maximize">
                  <option value="true">Higher is better</option>
                  <option value="false">Lower is better</option>
                </select>
              </label>
            </div>
            <label>
              Assay category
              <select name="category">
                <option value="binding">Binding</option>
                <option value="expression">Expression</option>
                <option value="thermostability">Thermostability</option>
                <option value="other">Other</option>
              </select>
            </label>
            <label>
              Measured sequence CSV
              <input name="file" type="file" accept=".csv" required />
            </label>
            <p className="pred-help">
              Scores are kept in the units you declare; no automatic conversion
              is applied to custom data. Replicate sequence pairs are aggregated
              before splitting.
            </p>
            <button
              className="pred-button primary"
              disabled={submitting}
              type="submit"
            >
              {submitting ? "Importing…" : "Add target-specific assay"}
              <ArrowRight size={17} />
            </button>
          </form>
        </div>
      )}
      {settings && (
        <div className="pred-modal-backdrop">
          <section
            className="pred-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Connection settings"
          >
            <button
              className="pred-close"
              aria-label="Close settings"
              onClick={() => setSettings(false)}
            >
              <X size={19} />
            </button>
            <h2>Connection settings</h2>
            <label>
              API bearer token
              <input
                type="password"
                autoComplete="off"
                value={auth}
                onChange={(e) => setAuth(e.target.value)}
              />
            </label>
            <p className="pred-help">
              Only needed if AEROFOLD_API_TOKEN is configured on the server.
              Stored in this browser session.
            </p>
            <button
              className="pred-button primary"
              onClick={() => {
                sessionStorage.setItem("aerofold-token", auth);
                setSettings(false);
                setError("");
                refresh().catch((e) => setError(e.message));
              }}
            >
              Save connection
            </button>
          </section>
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="pred-panel pred-stat">
      <p className="pred-eyebrow">{label}</p>
      <strong>{value}</strong>
      <p>{detail}</p>
    </div>
  );
}
function ModelCard({ card, onExport }: { card: Card; onExport: () => void }) {
  return (
    <section className="pred-panel">
      <div className="pred-section-title">
        <div>
          <p className="pred-eyebrow">MODEL RECORD</p>
          <h3>{card.foundation_model || "Offline sequence ensemble"}</h3>
        </div>
        <button className="pred-button secondary" onClick={onExport}>
          <BookOpen size={16} /> Full audit
        </button>
      </div>
      <p className="pred-muted small">
        {card.training.fitting} development sequences ·{" "}
        {card.training.calibration} separate calibration sequences. Weights are
        selected on internal validation before refitting.
      </p>
      <div className="pred-heads">
        {Object.entries(card.heads).map(([name, head]) => (
          <div key={name}>
            <span>{name.replaceAll("_", " ")}</span>
            <b>{percent(head.weight)}</b>
            <div>
              <i style={{ width: `${100 * head.weight}%` }} />
            </div>
            <small>Validation RMSE {number(head.rmse)}</small>
          </div>
        ))}
      </div>
      <p className="pred-help">
        {card.uncertainty}. Coverage can fail outside the training distribution;
        antibody pretraining alone does not establish antigen-specific affinity.
      </p>
    </section>
  );
}
