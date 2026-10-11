import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  Beaker,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Dna,
  ExternalLink,
  FlaskConical,
  History,
  Layers3,
  LayoutDashboard,
  Loader2,
  Plus,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { api, download, token } from "./api";
import Scatter from "./Scatter";
import SequenceMap from "./SequenceMap";
const BenchmarkView = lazy(() => import("./components/BenchmarkView"));
import type { Config, Dataset, Result, Run, Variant } from "./types";

const DEFAULT_CONFIG: Config = {
  embedding: "descriptor",
  exploration: 0.35,
  diversity: 0.55,
  min_stability: 60,
  confidence_beta: 1,
  max_abs_charge: 16,
  max_hydrophobic_patch: 0.9,
  max_apr: 0.9,
  seed: 42,
  batch_size: 24,
};
type Page = "overview" | "library" | "history" | "benchmark";
type Modal = "export" | "settings" | "import" | "methods" | "auth" | null;

export default function App() {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [config, setConfig] = useState<Config>(DEFAULT_CONFIG);
  const [page, setPage] = useState<Page>(
    window.location.hash === "#benchmark" ? "benchmark" : "overview",
  );
  const [modal, setModal] = useState<Modal>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("selected");
  const [tablePage, setTablePage] = useState(0);
  const [view, setView] = useState("variants");
  const [inspect, setInspect] = useState<Variant | null>(null);
  const [saved, setSaved] = useState<{ id: string; name: string }[]>([]);
  const [auth, setAuth] = useState(token());
  const result = run?.result || null;
  const working = busy || run?.status === "running" || run?.status === "queued";
  const changed =
    !!result && JSON.stringify(config) !== JSON.stringify(result.config);
  const selected = useMemo(
    () =>
      result?.variants
        .filter((v) => v.selected)
        .sort((a, b) => a.selection_rank! - b.selection_rank!) || [],
    [result],
  );

  async function startRun(
    data: Dataset = dataset!,
    parameters: Config = config,
  ) {
    if (!data) return;
    setError("");
    setBusy(true);
    setInspect(null);
    try {
      const job = await api<{ id: string }>(`/datasets/${data.id}/runs`, {
        method: "POST",
        body: JSON.stringify(parameters),
      });
      setRun(await api<Run>(`/runs/${job.id}`));
      setPage("overview");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function openDataset(data: Dataset, autoRun = true) {
    setDataset(data);
    setRun(null);
    setInspect(null);
    setTablePage(0);
    setConfig(DEFAULT_CONFIG);
    localStorage.setItem("aerofold-dataset", data.id);
    const latest = data.runs[0];
    if (latest) {
      const previous = await api<Run>(`/runs/${latest.id}`);
      setRun(previous);
      setConfig(previous.config);
    } else if (autoRun && data.observation_count >= 6) {
      await startRun(data, DEFAULT_CONFIG);
    }
    setSaved(await api("/datasets"));
  }

  async function initialize() {
    setBusy(true);
    setError("");
    try {
      const datasets = await api<{ id: string; name: string }[]>("/datasets");
      const remembered = localStorage.getItem("aerofold-dataset");
      const id =
        datasets.find((d) => d.id === remembered)?.id || datasets[0]?.id;
      const data = id
        ? await api<Dataset>(`/datasets/${id}`)
        : await api<Dataset>("/datasets/demo", { method: "POST" });
      await openDataset(data);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void initialize();
  }, []);
  useEffect(() => {
    if (!run || !["running", "queued"].includes(run.status)) return;
    let cancelled = false;
    const timeout = setTimeout(async () => {
      try {
        const updated = await api<Run>(`/runs/${run.id}`);
        if (cancelled) return;
        setRun(updated);
        if (updated.status === "completed") {
          const data = await api<Dataset>(`/datasets/${updated.dataset_id}`);
          if (!cancelled) {
            setDataset(data);
            setNotice(
              "24 diverse variants selected. Your next experiment is ready.",
            );
          }
        }
      } catch (e) {
        if (!cancelled) {
          setError((e as Error).message);
          setRun({
            ...run,
            status: "failed",
            error: "Connection interrupted. Reconnect to check the saved job.",
          });
        }
      }
    }, 1000);
    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [run]);
  useEffect(() => {
    if (notice) {
      const t = setTimeout(() => setNotice(""), 6000);
      return () => clearTimeout(t);
    }
  }, [notice]);
  useEffect(() => {
    setTablePage(0);
  }, [query, filter]);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setModal(null);
        setInspect(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  useEffect(() => {
    if (!modal && !inspect) return;
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const dialog = document.querySelector<HTMLElement>('[role="dialog"]');
    const selector =
      'button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), summary, [tabindex="0"]';
    dialog?.querySelector<HTMLElement>(selector)?.focus();
    const trap = (event: KeyboardEvent) => {
      if (event.key !== "Tab" || !dialog) return;
      const items = Array.from(
        dialog.querySelectorAll<HTMLElement>(selector),
      ).filter((item) => item.getClientRects().length > 0);
      const first = items[0],
        last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", trap);
    return () => {
      document.body.style.overflow = overflow;
      document.removeEventListener("keydown", trap);
      previous?.focus({ preventScroll: true });
    };
  }, [modal, inspect]);

  async function importFile(file: File | undefined, measurements: boolean) {
    if (!file || (measurements && !dataset)) return;
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const data = await api<Dataset>(
        measurements
          ? `/datasets/${dataset!.id}/observations/upload`
          : "/datasets/upload",
        { method: "POST", body: form },
      );
      setModal(null);
      if (measurements) {
        setDataset(data);
        if (run) setRun({ ...run, stale: true });
        setNotice(
          "Measurements saved. Run selection to learn from the new results.",
        );
      } else await openDataset(data);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function simulate() {
    if (!run || !dataset) return;
    setBusy(true);
    setError("");
    try {
      const data = await api<Dataset>(`/runs/${run.id}/simulate`, {
        method: "POST",
      });
      setDataset(data);
      setRun({ ...run, stale: true });
      setNotice(
        "24 synthetic assay results added. Fitting the next learning round…",
      );
      await startRun(data);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function exportFile(format: string) {
    if (!run) return;
    try {
      await download(run.id, format);
      setNotice("Export downloaded.");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const filtered = (result?.variants || [])
    .filter(
      (v) =>
        (filter === "all" ||
          (filter === "selected" && v.selected) ||
          (filter === "pareto" && v.pareto) ||
          (filter === "eligible" && v.eligible)) &&
        `${v.variant_id} ${v.mutations.map((m) => m.label).join(" ")}`
          .toLowerCase()
          .includes(query.toLowerCase()),
    )
    .sort((a, b) =>
      filter === "selected"
        ? a.selection_rank! - b.selection_rank!
        : b.acquisition - a.acquisition,
    );
  const tableRows = filtered.slice(tablePage * 12, (tablePage + 1) * 12);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("overview");
          }}
        >
          <div className="brand-mark">
            <Dna size={26} />
          </div>
          <div>
            AeroFold<span>ANTIBODY INTELLIGENCE</span>
          </div>
          <small>Ab</small>
        </a>
        <div className="workspace-label">
          WORKSPACE <span>01</span>
        </div>
        <button className="project-switch" onClick={() => setPage("library")}>
          <div className="project-icon">
            <FlaskConical size={18} />
          </div>
          <div>
            Antibody discovery
            <small>
              {dataset?.demo ? "Demo project" : "Experiment workspace"}
            </small>
          </div>
          <ChevronDown size={14} />
        </button>
        <nav aria-label="Main navigation">
          {(
            [
              { id: "overview", label: "Overview", Icon: LayoutDashboard },
              { id: "library", label: "Sequence library", Icon: Layers3 },
              { id: "history", label: "Learning history", Icon: History },
              { id: "benchmark", label: "Benchmark mode", Icon: FlaskConical },
            ] as const
          ).map(({ id, label, Icon }) => (
            <button
              key={id}
              onClick={() => {
                setPage(id);
                window.history.replaceState(
                  null,
                  "",
                  id === "benchmark" ? "#benchmark" : window.location.pathname,
                );
              }}
              className={page === id ? "nav-item active" : "nav-item"}
            >
              <Icon size={18} />
              {label}
              {id === "library" && dataset && (
                <small>{dataset.variant_count}</small>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-divider" />
        <div className="workspace-label">EXPERIMENT TOOLS</div>
        <button className="nav-item" onClick={() => setModal("settings")}>
          <SlidersHorizontal size={18} />
          Selection parameters
        </button>
        <button
          className="nav-item"
          onClick={() => setModal("export")}
          disabled={!result}
        >
          <ArrowDownToLine size={18} />
          Export center
        </button>
        <button className="nav-item" onClick={() => setModal("methods")}>
          <BookOpen size={18} />
          Methods & provenance
        </button>
        <button
          className="nav-item connection-mobile"
          onClick={() => setModal("auth")}
        >
          <Settings2 size={18} />
          API connection
        </button>
        <div className="sidebar-bottom">
          <div className="loop-card">
            <span className="mini-icon">
              <Activity size={16} />
            </span>
            <b>Learn. Select. Discover.</b>
            <p>Every experiment makes the next one smarter.</p>
            <div className="loop-dots">
              <i />
              <span />
              <i />
              <span />
              <i />
              <span />
              <i className="outline" />
            </div>
          </div>
          <button className="help-button" onClick={() => setModal("methods")}>
            <CircleHelp size={17} />
            About this workspace
            <ArrowUpRight size={14} />
          </button>
          <button className="profile" onClick={() => setModal("auth")}>
            <span>AF</span>
            <div>
              Research workspace
              <small>
                Local ·{" "}
                {config.embedding === "descriptor"
                  ? "CPU ready"
                  : config.embedding === "esm2"
                    ? "ESM-2 mode"
                    : "AbLang-2 mode"}
              </small>
            </div>
            <Settings2 size={15} />
          </button>
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={13} />
            <span>
              {page === "benchmark" ? "Benchmark mode" : "Antibody discovery"}
            </span>
            <span className="environment">
              {page === "benchmark"
                ? "PUBLIC DMS"
                : dataset?.demo
                  ? "SANDBOX"
                  : "LOCAL"}
            </span>
          </div>
          <div className="topbar-right">
            <span className="status-dot" />
            Sequence-first discovery
            <button
              className="icon-button"
              title="Methods and limitations"
              aria-label="Methods and limitations"
              onClick={() => setModal("methods")}
            >
              <CircleHelp size={17} />
            </button>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <div className="eyebrow">AEROFOLD-Ab / DISCOVERY WORKSPACE</div>
              <h1>
                {page === "overview"
                  ? "Better variants. Smarter experiments."
                  : page === "library"
                    ? "Your sequence landscape."
                    : page === "benchmark"
                      ? "Benchmark the learning loop."
                      : "Discovery is a learning loop."}
              </h1>
              <p>
                {page === "overview"
                  ? "Navigate the trade-off between binding, stability, and what comes next."
                  : page === "library"
                    ? "Bring your library, connect assay results, and explore the sequence space."
                    : page === "benchmark"
                      ? "Compare acquisition strategies on a public, measured antibody landscape."
                      : "Every run is a reproducible snapshot of the evidence available at that moment."}
              </p>
            </div>
            {page !== "benchmark" && (
              <button
                className="button secondary"
                onClick={() => setModal("import")}
                disabled={working}
              >
                <Upload size={15} />
                Import data
              </button>
            )}
          </div>
          {(error || run?.status === "failed") && (
            <div className="alert error" role="alert">
              <span>{error || run?.error}</span>
              <button
                onClick={() => {
                  setError("");
                  void initialize();
                }}
              >
                Reconnect
              </button>
            </div>
          )}
          {notice && (
            <div className="toast" role="status">
              <CheckCircle2 size={17} />
              {notice}
              <button
                aria-label="Dismiss notification"
                onClick={() => setNotice("")}
              >
                <X size={14} />
              </button>
            </div>
          )}
          {run?.stale && (
            <div className="alert">
              New assay results are available. This selection is historical; run
              a new selection to export a current batch.
            </div>
          )}

          {page !== "benchmark" && (
            <div className="experiment-bar">
              <div>
                <span className="experiment-icon">
                  <Dna size={18} />
                </span>
                <b>{dataset?.name || "Loading your workspace…"}</b>
                <span className="tag">
                  {dataset?.demo ? "Synthetic demo" : "Uploaded library"}
                </span>
              </div>
              <span className="experiment-meta">
                Revision {dataset?.revision || "—"}
                <span>·</span>
                {dataset?.observation_count || 0} observations
                <ShieldCheck size={14} />
              </span>
            </div>
          )}

          {page === "benchmark" && (
            <Suspense fallback={<p role="status">Loading Benchmark Mode…</p>}>
              <BenchmarkView />
            </Suspense>
          )}
          {page === "overview" && (
            <>
              <div className="metrics">
                <Metric
                  label="VARIANT LIBRARY"
                  value={dataset?.variant_count.toLocaleString() || "—"}
                  detail={`${result?.summary.observed_variants ?? Math.max(0, (dataset?.observation_count || 1) - 1)} measured variants`}
                  icon={<Layers3 size={17} />}
                />
                <Metric
                  label="PARETO FRONTIER"
                  value={result?.summary.pareto_count.toString() || "—"}
                  detail="Nondominated candidates"
                  icon={<Activity size={17} />}
                />
                <Metric
                  label="BATCH SIZE"
                  value={result ? "24" : "—"}
                  detail={
                    result
                      ? `${result.summary.eligible} candidates meet constraints`
                      : "Exactly 24 diverse variants"
                  }
                  icon={<Beaker size={17} />}
                  accent
                />
                <Metric
                  label="PREDICTED BINDING GAIN"
                  value={
                    result
                      ? `${result.summary.mean_binding - result.summary.parent_binding >= 0 ? "+" : ""}${(result.summary.mean_binding - result.summary.parent_binding).toFixed(2)}`
                      : "—"
                  }
                  detail="Batch mean vs. parent · score units"
                  icon={<ArrowUpRight size={17} />}
                />
              </div>
              <div className="run-toolbar">
                <div className="run-title">
                  <span className="live-dot" />
                  <b>Active learning</b>
                  <span className="tag">MC-EHVI + DPP</span>
                  <span className="muted">
                    {config.embedding === "descriptor"
                      ? "Offline descriptors"
                      : config.embedding === "esm2"
                        ? "Pretrained ESM-2"
                        : "Pretrained AbLang-2"}
                  </span>
                </div>
                <div className="toolbar-actions">
                  <button
                    className="text-button"
                    onClick={() => setModal("settings")}
                  >
                    <Settings2 size={15} />
                    {changed ? "Parameters changed" : "Configure"}
                  </button>
                  <button
                    className="button primary"
                    onClick={() => void startRun()}
                    disabled={
                      working || !dataset || dataset.observation_count < 6
                    }
                  >
                    {working ? (
                      <Loader2 className="spin" size={15} />
                    ) : (
                      <Sparkles size={15} />
                    )}{" "}
                    {working
                      ? "Selecting variants…"
                      : result
                        ? "Run new selection"
                        : "Select top 24"}
                  </button>
                </div>
              </div>
              {working && (
                <div className="progress-panel" role="status">
                  <div className="progress-track">
                    <i />
                  </div>
                  <span>
                    {run?.progress ||
                      "Preparing the sequence library and model…"}
                  </span>
                </div>
              )}
              {result ? (
                <>
                  <div className="analysis-grid">
                    <Scatter result={result} inspect={setInspect} />
                    <BatchCard
                      result={result}
                      stale={run!.stale}
                      onExport={() => setModal("export")}
                      inspect={setInspect}
                    />
                  </div>
                  <section className="panel candidates">
                    <div className="table-heading">
                      <div className="tabs">
                        <button
                          className={view === "variants" ? "active" : ""}
                          onClick={() => setView("variants")}
                        >
                          Variant explorer <span>{filtered.length}</span>
                        </button>
                        <button
                          className={view === "sequence" ? "active" : ""}
                          onClick={() => setView("sequence")}
                        >
                          <Dna size={15} />
                          Sequence heatmap
                        </button>
                      </div>
                      <button
                        className="text-button"
                        onClick={() => void exportFile("audit")}
                        disabled={run!.stale}
                      >
                        <ArrowDownToLine size={14} />
                        Run manifest
                      </button>
                    </div>
                    {view === "sequence" ? (
                      <SequenceMap result={result} />
                    ) : (
                      <>
                        <div className="table-controls">
                          <div className="segmented">
                            {[
                              ["selected", "Selected 24"],
                              ["all", "All variants"],
                              ["pareto", "Pareto frontier"],
                              ["eligible", "Eligible"],
                            ].map(([id, label]) => (
                              <button
                                key={id}
                                className={filter === id ? "active" : ""}
                                onClick={() => setFilter(id)}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                          <label className="search">
                            <Search size={15} />
                            <input
                              aria-label="Search variants or mutations"
                              placeholder="Search variants or mutations…"
                              value={query}
                              onChange={(e) => setQuery(e.target.value)}
                            />
                          </label>
                        </div>
                        <div className="table-scroll">
                          <table className="variant-table">
                            <thead>
                              <tr>
                                <th>VARIANT ID</th>
                                <th>CHANGES</th>
                                <th>
                                  PRED. BINDING <span>↑</span>
                                </th>
                                <th>
                                  ADJ. STABILITY <span>↑</span>
                                </th>
                                <th>ACQUISITION</th>
                                <th>STATUS</th>
                                <th>WELL</th>
                                <th />
                              </tr>
                            </thead>
                            <tbody>
                              {tableRows.map((v) => (
                                <tr
                                  key={v.variant_id}
                                  onClick={() => setInspect(v)}
                                >
                                  <td>
                                    <b className="variant-id">
                                      <i
                                        className={
                                          v.selected ? "dot teal" : "dot gray"
                                        }
                                      />
                                      {v.variant_id}
                                    </b>
                                  </td>
                                  <td>
                                    <span
                                      className="mutation-pill"
                                      title={v.mutations
                                        .map((m) => m.label)
                                        .join(", ")}
                                    >
                                      {v.mutations.length} substitutions /
                                      indels
                                    </span>
                                  </td>
                                  <td>
                                    <b>{v.binding_mean.toFixed(3)}</b>
                                    <small> ± {v.binding_std.toFixed(3)}</small>
                                  </td>
                                  <td>
                                    <b>{v.stability_mean.toFixed(1)}</b>
                                    <small>
                                      {" "}
                                      ± {v.stability_std.toFixed(1)}
                                    </small>
                                  </td>
                                  <td>
                                    <div className="score-cell">
                                      <span
                                        style={{
                                          width: `${v.acquisition * 50}px`,
                                        }}
                                      />
                                      {v.acquisition.toFixed(3)}
                                    </div>
                                  </td>
                                  <td>
                                    <span
                                      className={`status-pill ${v.selected ? "chosen" : ""}`}
                                    >
                                      {v.selected ? (
                                        <>
                                          <Check size={11} />
                                          Selected
                                        </>
                                      ) : v.measured ? (
                                        "Measured"
                                      ) : v.eligible ? (
                                        "Eligible"
                                      ) : (
                                        "Filtered"
                                      )}
                                    </span>
                                  </td>
                                  <td className="mono">{v.well || "—"}</td>
                                  <td>
                                    <button
                                      className="icon-button"
                                      aria-label={`Inspect ${v.variant_id}`}
                                      onClick={() => setInspect(v)}
                                    >
                                      <ArrowUpRight size={14} />
                                    </button>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {!tableRows.length && (
                            <div className="empty-small">
                              No variants match this search.
                            </div>
                          )}
                        </div>
                        <div className="table-footer">
                          <span>
                            {filtered.length ? tablePage * 12 + 1 : 0}–
                            {Math.min((tablePage + 1) * 12, filtered.length)} of{" "}
                            {filtered.length} variants <i>·</i> Uncertainty
                            shown as ±1σ
                          </span>
                          <div>
                            <button
                              aria-label="Previous page"
                              disabled={tablePage === 0}
                              onClick={() => setTablePage((p) => p - 1)}
                            >
                              <ChevronLeft size={15} />
                            </button>
                            <span>
                              {tablePage + 1} /{" "}
                              {Math.max(1, Math.ceil(filtered.length / 12))}
                            </span>
                            <button
                              aria-label="Next page"
                              disabled={(tablePage + 1) * 12 >= filtered.length}
                              onClick={() => setTablePage((p) => p + 1)}
                            >
                              <ChevronRight size={15} />
                            </button>
                          </div>
                        </div>
                      </>
                    )}
                  </section>
                  <div className="next-round">
                    <div className="next-round-icon">
                      <Activity size={22} />
                    </div>
                    <div>
                      <b>Close the loop. Open new possibilities.</b>
                      <p>
                        {dataset?.demo
                          ? "Add synthetic outcomes for this batch, then watch the model select a new set of 24."
                          : "Import your measured binding and stability scores to improve the next selection."}
                      </p>
                    </div>
                    <button
                      className="button secondary"
                      disabled={working || run!.stale}
                      onClick={() =>
                        dataset?.demo ? void simulate() : setModal("import")
                      }
                    >
                      {dataset?.demo
                        ? "Simulate next round"
                        : "Upload assay results"}
                      <ArrowRight size={15} />
                    </button>
                  </div>
                </>
              ) : (
                !working && (
                  <div className="empty-state">
                    <Dna size={36} />
                    <h2>Your next experiment starts here.</h2>
                    <p>
                      Upload at least six paired assay results, or load the
                      500-variant demo to explore a complete active learning
                      loop.
                    </p>
                    <button
                      className="button primary"
                      onClick={() => setModal("import")}
                    >
                      Load a library <Plus size={16} />
                    </button>
                  </div>
                )
              )}
            </>
          )}

          {page === "library" && (
            <>
              <div className="library-grid">
                <section className="panel padded">
                  <div className="eyebrow">PARENT TEMPLATE</div>
                  <h2>{dataset?.parent.variant_id || "No parent loaded"}</h2>
                  <p>
                    Sequences are variable domains. CDR regions use approximate
                    reference alignment.
                  </p>
                  {dataset?.annotations.map((a) => (
                    <div key={a.chain} className="parent-chain">
                      <div>
                        <b>{a.chain === "H" ? "Heavy chain" : "Light chain"}</b>
                        <span>
                          {a.chain === "H"
                            ? dataset.parent.heavy_sequence.length
                            : dataset.parent.light_sequence.length}{" "}
                          aa · {(a.identity * 100).toFixed(0)}% reference
                          identity
                        </span>
                      </div>
                      <code>
                        {a.chain === "H"
                          ? dataset.parent.heavy_sequence
                          : dataset.parent.light_sequence}
                      </code>
                      <div className="region-tags">
                        {a.regions.map((r) => (
                          <span key={r.name} title={r.sequence}>
                            {r.name}
                            <b>
                              {r.start}–{r.end}
                            </b>
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </section>
                <section className="panel padded">
                  <div className="eyebrow">CONNECTED EVIDENCE</div>
                  <h2>{dataset?.observation_count || 0} paired observations</h2>
                  <p>
                    Binding and stability must both be higher-is-better scores.
                    Leave measurements blank for untested candidates.
                  </p>
                  <button
                    className="button primary"
                    onClick={() => setModal("import")}
                  >
                    <Upload size={15} />
                    Import library or results
                  </button>
                  <div className="method-note">
                    <ShieldCheck size={19} />
                    <p>
                      Measurements are saved atomically. Existing assays are
                      never overwritten, and measured variants are excluded from
                      the next batch.
                    </p>
                  </div>
                  <label className="field-label">
                    Saved libraries
                    <select
                      value={dataset?.id || ""}
                      disabled={working}
                      onChange={async (e) => {
                        setBusy(true);
                        try {
                          await openDataset(
                            await api<Dataset>(`/datasets/${e.target.value}`),
                          );
                        } catch (err) {
                          setError((err as Error).message);
                        } finally {
                          setBusy(false);
                        }
                      }}
                    >
                      {saved.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name} · {d.id.slice(0, 6)}
                        </option>
                      ))}
                    </select>
                  </label>
                </section>
              </div>
            </>
          )}

          {page === "history" && (
            <section className="panel padded">
              <div className="eyebrow">EXPERIMENT LEDGER</div>
              <h2>Selection history</h2>
              <p>
                Runs retain their original parameters and predictions. New
                observations make older runs historical.
              </p>
              <div className="history-list">
                {dataset?.runs.length ? (
                  dataset.runs.map((r, i) => (
                    <button
                      className="history-row"
                      key={r.id}
                      onClick={async () => {
                        try {
                          const savedRun = await api<Run>(`/runs/${r.id}`);
                          setRun(savedRun);
                          setConfig(savedRun.config);
                          setPage("overview");
                        } catch (e) {
                          setError((e as Error).message);
                        }
                      }}
                    >
                      <span className="history-number">
                        {String(dataset.runs.length - i).padStart(2, "0")}
                      </span>
                      <div>
                        <b>Selection {r.id.slice(0, 8)}</b>
                        <small>
                          Revision {r.revision} ·{" "}
                          {new Date(r.created_at).toLocaleString()}
                        </small>
                      </div>
                      <span
                        className={`status-pill ${r.status === "completed" ? "chosen" : ""}`}
                      >
                        {r.status}
                      </span>
                      <span className="muted">
                        {r.revision === dataset.revision
                          ? "Current evidence"
                          : "Historical"}
                      </span>
                      <ArrowUpRight size={17} />
                    </button>
                  ))
                ) : (
                  <div className="empty-small">
                    Your first selection will appear here.
                  </div>
                )}
              </div>
            </section>
          )}

          <footer>
            <span>
              <Dna size={14} />
              AeroFold-Ab <b>Built for the next discovery.</b>
            </span>
            <span>
              {page === "benchmark"
                ? "Public experimental data · "
                : dataset?.demo
                  ? "Synthetic data · "
                  : ""}
              Sequence proxies require experimental validation{" "}
              <button onClick={() => setModal("methods")}>
                View methods <ExternalLink size={11} />
              </button>
            </span>
          </footer>
        </main>
      </div>

      {modal && (
        <div
          className="modal-backdrop"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setModal(null);
          }}
        >
          <section
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-title"
          >
            <button
              className="modal-close icon-button"
              aria-label="Close dialog"
              onClick={() => setModal(null)}
            >
              <X size={20} />
            </button>
            {modal === "settings" && (
              <>
                <div className="eyebrow">DESIGN THE NEXT EXPERIMENT</div>
                <h2 id="modal-title">Selection parameters</h2>
                <p>
                  Hard constraints apply before diversity selection. The batch
                  always contains exactly 24 eligible variants.
                </p>
                <label className="field-label">
                  Sequence representation
                  <select
                    value={config.embedding}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        embedding: e.target.value as Config["embedding"],
                      })
                    }
                  >
                    <option value="descriptor">
                      Offline descriptors · immediate demo
                    </option>
                    <option value="esm2">
                      ESM-2 · pretrained protein embeddings
                    </option>
                    <option value="ablang2">
                      AbLang-2 · paired antibody embeddings
                    </option>
                  </select>
                </label>
                {config.embedding === "esm2" && (
                  <div className="alert">
                    Requires optional ESM dependencies. First run downloads
                    public model weights; masked likelihood inference may take
                    several minutes. Sequences are processed locally.
                  </div>
                )}
                {config.embedding === "ablang2" && (
                  <div className="alert">
                    Requires optional AbLang-2 dependencies and complete variable
                    domains. First run downloads public model weights; sequences
                    are encoded locally and then cached.
                  </div>
                )}
                <Range
                  label="Exploration preference"
                  value={config.exploration}
                  onChange={(n) => setConfig({ ...config, exploration: n })}
                  left="Favor predicted improvement"
                  right="Favor uncertainty"
                />
                <Range
                  label="Sequence diversity"
                  value={config.diversity}
                  onChange={(n) => setConfig({ ...config, diversity: n })}
                  left="Prioritize acquisition"
                  right="Broaden sequence coverage"
                />
                <div className="fields-grid">
                  {(
                    [
                      {
                        key: "min_stability",
                        label: "Minimum adjusted stability",
                        min: 0,
                        max: 100,
                        step: 1,
                      },
                      {
                        key: "confidence_beta",
                        label: "Confidence multiplier β",
                        min: 0,
                        max: 3,
                        step: 0.1,
                      },
                      {
                        key: "max_abs_charge",
                        label: "Maximum |chain charge|",
                        min: 0,
                        max: 100,
                        step: 1,
                      },
                      {
                        key: "max_hydrophobic_patch",
                        label: "Max. hydrophobic patch",
                        min: 0,
                        max: 1,
                        step: 0.05,
                      },
                      {
                        key: "max_apr",
                        label: "Max. aggregation proxy",
                        min: 0,
                        max: 1,
                        step: 0.05,
                      },
                      {
                        key: "seed",
                        label: "Random seed",
                        min: 0,
                        max: 2147483647,
                        step: 1,
                      },
                    ] as const
                  ).map((f) => (
                    <label className="field-label" key={f.key}>
                      {f.label}
                      <input
                        type="number"
                        min={f.min}
                        max={f.max}
                        step={f.step}
                        value={config[f.key]}
                        onChange={(e) => {
                          if (
                            e.target.value !== "" &&
                            Number.isFinite(e.target.valueAsNumber)
                          )
                            setConfig({
                              ...config,
                              [f.key]: e.target.valueAsNumber,
                            });
                        }}
                      />
                    </label>
                  ))}
                </div>
                <div className="method-note">
                  <ShieldCheck size={19} />
                  <p>
                    Stability gate: adjusted mean − β × epistemic σ ≥ threshold.
                    If fewer than 24 pass, the run stops with an explanation.
                  </p>
                </div>
                <button
                  className="button primary full"
                  disabled={working || !dataset}
                  onClick={() => {
                    setModal(null);
                    void startRun();
                  }}
                >
                  <Sparkles size={16} />
                  Apply & select 24
                </button>
              </>
            )}
            {modal === "import" && (
              <>
                <div className="eyebrow">CONNECT YOUR EXPERIMENT</div>
                <h2 id="modal-title">Bring your sequences to life.</h2>
                <p>
                  Import a FASTA or CSV library, then add measured results as
                  your experiments progress.
                </p>
                <label className="upload-zone">
                  <Upload size={26} />
                  <b>Import a sequence library</b>
                  <span>FASTA or CSV · up to 2,000 variants · 5 MB</span>
                  <input
                    type="file"
                    accept=".csv,.fasta,.fa,.faa"
                    disabled={working}
                    onChange={(e) =>
                      void importFile(e.target.files?.[0], false)
                    }
                  />
                </label>
                <details>
                  <summary>View input formats</summary>
                  <p>
                    CSV:{" "}
                    <code>
                      variant_id,role,heavy_sequence,light_sequence,binding,stability
                    </code>
                    . Mark one row <code>role=parent</code>. For single chains
                    use <code>sequence,chain_type</code> (H or L). Each domain
                    must be 70–180 canonical residues.
                  </p>
                  <p>
                    FASTA: <code>&gt;parent/H</code>, <code>&gt;parent/L</code>,{" "}
                    <code>&gt;AB-0001/H</code>, <code>&gt;AB-0001/L</code>.
                    Single heavy-chain headers may omit /H. FASTA needs a
                    separate results CSV before fitting.
                  </p>
                </details>
                <label className="upload-zone compact">
                  <Activity size={24} />
                  <b>Add assay results to this library</b>
                  <span>CSV columns: variant_id, binding, stability</span>
                  <input
                    type="file"
                    accept=".csv"
                    disabled={working || !dataset}
                    onChange={(e) => void importFile(e.target.files?.[0], true)}
                  />
                </label>
                <div className="or-divider">OR START EXPLORING</div>
                <button
                  className="button secondary full"
                  disabled={working}
                  onClick={async () => {
                    setBusy(true);
                    setError("");
                    try {
                      const data = await api<Dataset>("/datasets/demo", {
                        method: "POST",
                      });
                      setModal(null);
                      await openDataset(data);
                    } catch (e) {
                      setError((e as Error).message);
                    } finally {
                      setBusy(false);
                    }
                  }}
                >
                  <FlaskConical size={16} />
                  Load a fresh 500-variant demo
                </button>
                {working && <p role="status">Importing and validating…</p>}
              </>
            )}
            {modal === "export" && (
              <>
                <div className="eyebrow">FROM MODEL TO BENCH</div>
                <h2 id="modal-title">Your next 24, ready to go.</h2>
                <p>
                  A01–A12 and B01–B12 on a standard 96-well plate. Every export
                  records the selection run.
                </p>
                {!result ? (
                  <div className="alert">
                    Complete a selection to enable exports.
                  </div>
                ) : (
                  <>
                    <div className="export-summary">
                      <CheckCircle2 size={21} />
                      <b>{selected.length} unique variants</b>
                      <span>{run?.id.slice(0, 8)}</span>
                    </div>
                    {[
                      [
                        "robot",
                        "Plate manifest",
                        "24 rows with wells, paired sequences, and predictions.",
                      ],
                      [
                        "synthesis",
                        "Protein synthesis order",
                        "One row per chain on separate H/L plates; amino-acid sequences.",
                      ],
                      [
                        "results",
                        "Assay results template",
                        "Fill in measured scores, then upload to close the loop.",
                      ],
                      [
                        "audit",
                        "Complete run manifest",
                        "JSON with parameters, training IDs, uncertainty, and provenance.",
                      ],
                    ].map(([format, title, description]) => (
                      <button
                        key={format}
                        className="export-option"
                        disabled={run?.stale}
                        onClick={() => void exportFile(format)}
                      >
                        <div>
                          <b>{title}</b>
                          <small>{description}</small>
                        </div>
                        <ArrowDownToLine size={19} />
                      </button>
                    ))}
                    <p className="fine-print">
                      Plate manifests are generic import tables, not executable
                      robot protocols. Protein orders require vendor-specific
                      DNA design and codon optimization before gene synthesis.
                    </p>
                    {run?.stale && (
                      <div className="alert">
                        This run is historical. Run a new selection before
                        exporting.
                      </div>
                    )}
                  </>
                )}
              </>
            )}
            {modal === "methods" && (
              <>
                <div className="eyebrow">TRANSPARENT BY DESIGN</div>
                <h2 id="modal-title">Evidence behind the selection.</h2>
                <div className="methods">
                  <h3>01 / Sequence representation</h3>
                  <p>
                    {result?.provenance.embedding ||
                      "Offline descriptors or optional pretrained ESM-2 / AbLang-2."}{" "}
                    Offline and AbLang-2 modes do not report natural sequence
                    likelihood. ESM mode masks each residue to compute
                    pseudo-perplexity.
                  </p>
                  <h3>02 / Learn two objectives</h3>
                  <p>
                    A joint PyTorch Gaussian process fits measured binding and
                    stability with learned covariance, returning means and latent
                    epistemic uncertainty. Both scores are maximized. Stability uses a
                    0–100 assay scale with explicit liability penalties. Raw Kd
                    must be transformed before upload.
                  </p>
                  <h3>03 / Explore the Pareto frontier</h3>
                  <p>
                    Marginal Monte Carlo expected hypervolume improvement uses
                    256 samples and the observed frontier. An uncertainty bonus
                    decays with training size. This is a practical marginal-EHVI
                    heuristic, not joint qNEHVI.
                  </p>
                  <h3>04 / Select a diverse batch</h3>
                  <p>
                    A quality-weighted greedy determinantal point process
                    operates on a frontier-enriched candidate pool. Stability
                    lower bounds, per-chain charge, hydrophobicity, and
                    aggregation proxies are hard filters.
                  </p>
                  <h3>Scope & validation</h3>
                  {(
                    result?.warnings || [
                      "Sequence scores are experimental prioritization tools, not structural guarantees.",
                    ]
                  ).map((w) => (
                    <p key={w} className="method-warning">
                      {w}
                    </p>
                  ))}
                  <p>
                    No predicted affinity is credible without relevant assay
                    observations. Independent-objective GPs do not model
                    cross-objective covariance or systematic assay bias.
                  </p>
                </div>
                <a
                  className="button secondary full"
                  href="/docs"
                  target="_blank"
                  rel="noreferrer"
                >
                  Open API documentation
                  <ExternalLink size={15} />
                </a>
              </>
            )}
            {modal === "auth" && (
              <>
                <div className="eyebrow">WORKSPACE ACCESS</div>
                <h2 id="modal-title">API connection</h2>
                <p>
                  The local workspace needs no token by default. If your server
                  requires one, enter its bearer token here. It is kept in this
                  browser session only.
                </p>
                <label className="field-label">
                  Bearer token
                  <input
                    type="password"
                    autoComplete="off"
                    value={auth}
                    onChange={(e) => setAuth(e.target.value)}
                  />
                </label>
                <button
                  className="button primary full"
                  onClick={() => {
                    sessionStorage.setItem("aerofold-token", auth);
                    setModal(null);
                    void initialize();
                  }}
                >
                  Save & reconnect
                  <ArrowRight size={15} />
                </button>
              </>
            )}
            {error && (
              <div className="alert error" role="alert">
                {error}
              </div>
            )}
          </section>
        </div>
      )}

      {inspect && (
        <div
          className="modal-backdrop"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setInspect(null);
          }}
        >
          <section
            className="modal variant-detail"
            role="dialog"
            aria-modal="true"
            aria-labelledby="variant-title"
          >
            <button
              className="modal-close icon-button"
              aria-label="Close variant"
              onClick={() => setInspect(null)}
            >
              <X size={20} />
            </button>
            <span className="eyebrow">VARIANT INSPECTOR</span>
            <h2 id="variant-title">{inspect.variant_id}</h2>
            <span className="status-pill chosen">
              {inspect.selected
                ? `Selected · well ${inspect.well}`
                : inspect.measured
                  ? "Measured"
                  : inspect.eligible
                    ? "Eligible candidate"
                    : "Filtered candidate"}
            </span>
            <div className="detail-scores">
              <div>
                <small>Binding score</small>
                <b>{inspect.binding_mean.toFixed(3)}</b>
                <span>± {inspect.binding_std.toFixed(3)} epistemic σ</span>
              </div>
              <div>
                <small>Adjusted stability</small>
                <b>{inspect.stability_mean.toFixed(1)}</b>
                <span>± {inspect.stability_std.toFixed(1)} epistemic σ</span>
              </div>
            </div>
            <h3>Sequence changes</h3>
            <div className="mutations-list">
              {inspect.mutations.map((m) => (
                <span key={m.label} title={m.region}>
                  {m.label}
                </span>
              ))}
            </div>
            {[
              ["Heavy chain", inspect.heavy_sequence],
              ["Light chain", inspect.light_sequence],
            ]
              .filter(([, seq]) => seq)
              .map(([label, seq]) => (
                <div className="parent-chain" key={label}>
                  <b>{label}</b>
                  <code>{seq}</code>
                </div>
              ))}
            <div className="detail-facts">
              <span>
                Stability lower bound<b>{inspect.stability_lcb.toFixed(2)}</b>
              </span>
              <span>
                Net charge proxy<b>{inspect.net_charge.toFixed(1)}</b>
              </span>
              <span>
                Hydrophobic patch<b>{inspect.hydrophobic_patch.toFixed(3)}</b>
              </span>
              <span>
                Aggregation proxy<b>{inspect.apr.toFixed(3)}</b>
              </span>
              <span>
                Developability penalty
                <b>{inspect.developability_penalty.toFixed(3)}</b>
              </span>
              <span>
                Masked pseudo-perplexity
                <b>
                  {inspect.pseudo_perplexity?.toFixed(2) ||
                    "Not available in offline mode"}
                </b>
              </span>
            </div>
            {inspect.exclusions.length > 0 && (
              <div className="alert">
                Excluded: {inspect.exclusions.join(", ")}
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  detail,
  icon,
  accent = false,
}: {
  label: string;
  value: string;
  detail: string;
  icon: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <div className={`metric ${accent ? "accent" : ""}`}>
      <div className="metric-label">
        {label}
        {icon}
      </div>
      <div className="metric-value">
        {value}
        {accent && <span>next to test</span>}
      </div>
      <div className="metric-detail">
        {accent && <span className="status-dot" />}
        {detail}
      </div>
    </div>
  );
}
function Range({
  label,
  value,
  onChange,
  left,
  right,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  left: string;
  right: string;
}) {
  return (
    <label className="range-field">
      <span>
        {label}
        <b>{Math.round(value * 100)}%</b>
      </span>
      <input
        type="range"
        min="0"
        max="1"
        step=".05"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <small>
        <span>{left}</span>
        <span>{right}</span>
      </small>
    </label>
  );
}
function BatchCard({
  result,
  stale,
  onExport,
  inspect,
}: {
  result: Result;
  stale: boolean;
  onExport: () => void;
  inspect: (v: Variant) => void;
}) {
  const selected = result.variants
    .filter((v) => v.selected)
    .sort((a, b) => a.selection_rank! - b.selection_rank!);
  return (
    <section className="panel batch-card">
      <div className="batch-heading">
        <span className="eyebrow">YOUR NEXT EXPERIMENT</span>
        <span className="tag">{stale ? "Historical" : "Ready to test"}</span>
      </div>
      <div className="batch-number">
        24
        <span>
          possibilities.
          <br />
          One focused batch.
        </span>
      </div>
      <p>
        High-potential variants, selected for improvement and sequence
        diversity.
      </p>
      <div className="plate">
        <div className="plate-header">
          <span>96-WELL PLATE</span>
          <span>A01–B12</span>
        </div>
        <div className="plate-grid">
          <span />
          {Array.from({ length: 12 }, (_, i) => (
            <small key={i}>{i + 1}</small>
          ))}
          {Array.from({ length: 8 }, (_, row) => (
            <div className="plate-row" key={row}>
              <small>{"ABCDEFGH"[row]}</small>
              {Array.from({ length: 12 }, (_, col) => {
                const v = selected[row * 12 + col];
                return (
                  <button
                    key={col}
                    aria-label={
                      v
                        ? `Well ${v.well}: ${v.variant_id}`
                        : `Empty well ${"ABCDEFGH"[row]}${col + 1}`
                    }
                    title={v?.variant_id || "Empty well"}
                    disabled={!v}
                    className={v ? "well occupied" : "well"}
                    onClick={() => v && inspect(v)}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <div className="batch-stat">
        <span>Mean adjusted stability</span>
        <b>
          {result.summary.mean_stability.toFixed(1)}
          <small> / 100</small>
        </b>
      </div>
      <div className="batch-stat">
        <span>Minimum confidence bound</span>
        <b>{result.summary.min_stability_lcb.toFixed(1)}</b>
      </div>
      <button
        className="button primary full"
        disabled={stale}
        onClick={onExport}
      >
        <ArrowDownToLine size={16} />
        Export selected batch
        <ArrowRight size={16} />
      </button>
      <div className="batch-foot">
        <ShieldCheck size={12} />
        All 24 pass configured constraints
      </div>
    </section>
  );
}
