# AeroFold-Ab

An end-to-end antibody active-learning workbench: **FastAPI + React/TypeScript + PyTorch**, with optional pretrained AbLang-2 or ESM-2 embeddings, a joint two-objective Gaussian-process surrogate, multi-objective acquisition, and exactly **24 diverse, feasible candidates** per successful run.

The default demo starts immediately without model downloads. It contains **500 unique paired-chain variants**, 48 measured seed variants, and a measured parent. Outcomes for the remaining variants stay hidden until you simulate the selected experiment.

## Run locally

Use **Python 3.11–3.12** and **Node.js 20.19+ or 22+**. A first installation needs internet access. Windows may have an older `python` earlier on PATH; check `python --version` before creating the environment.

```powershell
# Windows PowerShell, from the project directory
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm.cmd ci
npm.cmd run build
cd ..
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
# On CPU-only Linux, install the CPU wheel first to avoid CUDA downloads:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. FastAPI serves the compiled dashboard and API on the same origin. Interactive API documentation is at **http://127.0.0.1:8000/docs**. A new browser workspace automatically loads the demo and selects a batch. Previously saved libraries resume from SQLite.

For frontend development, keep the API running and run `npm run dev` in `frontend/`. Vite serves **http://127.0.0.1:5173** and proxies `/api` to port 8000. Its API-docs link is available on the backend port. Rebuild the frontend and restart the API if the production build did not exist when the API started.

### Try a full learning loop

1. Inspect the predicted binding–stability plot, uncertainty bars, and frontier.
2. Open **Sequence heatmap** or inspect a variant to see parent-relative substitutions and indels.
3. Adjust exploration, diversity, and hard constraints in **Selection parameters**.
4. Export the 24-well manifest or blank assay-results template.
5. Click **Simulate next round** in the demo. Exactly the selected 24 synthetic outcomes are revealed, the dataset revision advances, and a fresh batch excludes all measured sequences.
6. Use **Learning history** to inspect prior configurations and predictions. Historical runs cannot export a current plate or simulate stale results.

## Real sequence and assay inputs

All input sequences are **variable domains**, 70–180 canonical amino acids per chain. Heavy-only, light-only, and paired-chain libraries are supported. Each variant must have the same chain types as its parent. Duplicate sequence pairs and duplicate IDs are rejected, including a duplicate of the parent. IDs must begin with a letter/number and contain only letters, numbers, `_`, `-`, or `.`. Limits: 24–2,000 candidates and 5 MB per upload.

### CSV library

```csv
variant_id,role,heavy_sequence,light_sequence,binding,stability
parent,parent,EVQL...,DIQM...,7.0,87.0
AB-0001,variant,EVQL...,DIQM...,7.2,83.0
AB-0002,variant,EVQL...,DIQM...,,
```

Ellipses above are illustrative; use full sequences. Mark exactly one parent with `role=parent` or `variant_id=parent`. For a single chain, `sequence` and `chain_type` (`H` or `L`, default `H`) may replace the separate chain columns. Provide **both** assay values for measured rows and leave **both blank** for untested candidates. The checked-in `data/demo_library.csv` is a complete input example.

### FASTA library

```text
>parent/H
FULL_HEAVY_VARIABLE_DOMAIN
>parent/L
FULL_LIGHT_VARIABLE_DOMAIN
>AB-0001/H
VARIANT_HEAVY_VARIABLE_DOMAIN
>AB-0001/L
VARIANT_LIGHT_VARIABLE_DOMAIN
```

Use real canonical sequences, not the placeholder text above. A single heavy-chain library can omit `/H`. FASTA does not include assay results; upload measurements separately before fitting.

### Assay results

```csv
variant_id,binding,stability
AB-0002,7.42,82.5
AB-0003,7.18,85.1
```

At least **six paired observations** are required, including the parent if measured. Sequences alone cannot identify antigen-specific binding. Results are appended atomically; unknown IDs, duplicate IDs, non-finite values, partially populated results, and overwriting existing assays are rejected. For corrected assays, import a new library with the corrected measurement set.

**Units matter:** both objectives are maximized. Binding is an assay-specific higher-is-better score. For positive Kd in molar units, use `−log10(Kd)` before import; do not upload raw Kd as binding. Stability is a calibrated **0–100 higher-is-better assay score**, not raw melting temperature or ΔG. Convert your assay using a fixed, documented scale before upload. Keep units and assay conditions consistent across rounds. The synthetic demo scores have no physical interpretation.

## Scientific implementation

### Annotation and representation

- `sequences.py`: deterministic Needleman–Wunsch alignment, exact parent-relative substitutions/indels, and approximate CDR transfer from explicit heavy/light reference segments. Parent reference identity below 60% is rejected. Coordinates are **one-based ungapped parent positions, not IMGT or Kabat numbering**. Verify unusual germlines/loop boundaries with a validated numbering pipeline before experimental use. An ambiguous repetitive alignment can have more than one equally scoring interpretation.
- `features.py`: offline representation is a fixed seeded projection of position-specific amino-acid one-hot features, plus sequence descriptors (102 dimensions). It is deliberately labeled **not a pLM**; it never invents natural sequence likelihood.
- Optional ESM mode uses `facebook/esm2_t6_8M_UR50D` through Hugging Face Transformers. Residue hidden states are mean-pooled per chain (excluding special tokens), then chain embeddings are concatenated. Each residue is masked in turn and scored against its true token. Pseudo-perplexity is `exp(mean masked NLL)` across residues. This is a true masked pseudo-likelihood calculation, not unmasked reconstruction confidence.
- Optional AbLang-2 mode uses the pretrained `ablang2-paired` antibody language model on complete VH/VL variable domains (or one complete chain). Its official sequence-coding representation is cached per exact chain pair. This path does not calculate pseudo-perplexity. It requires full domains; the fragment-only Tite-Seq benchmark cannot be passed to it without a verified original scaffold.
- ESM inference is local. Downloaded weights and content-addressed sequence feature caches can be reused. Only public weights are downloaded; antibody sequences are not sent to Hugging Face. The frontend uses optional Google Fonts with system fallbacks.

### Surrogate and constraints

`surrogate.py` fits a joint two-output intrinsic-coregionalization Gaussian process in double-precision PyTorch. A learned task correlation models binding/stability covariance; a linear-plus-RBF sequence kernel models additive effects and residual interactions, with separate assay-noise terms. Inputs and targets are normalized using training observations only. Predictions contain means, a 2×2 latent objective covariance and **latent epistemic σ**, excluding observation noise; they are not calibrated clinical or biophysical confidence intervals.

Exact-GP training is capped at 512 observations to bound cubic cost. Larger observation sets are deterministically sampled at evenly spaced indices in library order; all observed candidates remain excluded. The exact training IDs and total observation count are recorded in every run. Use a sparse GP implementation for larger production studies.

The stability surrogate predicts the measured score. A known sequence-based penalty is then subtracted:

```text
penalty = 3 max(hydrophobic_patch − 0.65, 0)
        + 2 APR
        + 0.1 max(charge_patch − 3, 0)
        + 2 max(log(pseudo_perplexity / parent_pseudo_perplexity), 0) [ESM only]

adjusted_stability = raw_stability_mean − penalty
stability_LCB = adjusted_stability − beta × stability_epistemic_sigma
```

The numerical penalty weights are transparent heuristics on the stated 0–100 score scale. They need calibration on your assays. Hydrophobic patch = maximum seven-residue hydrophobic fraction; APR = the maximum seven-residue hydrophobic fraction downweighted by local absolute charge. Net charge uses K/R = +1, D/E = −1, H = +0.1. Charge patches use K/R/D/E. Windows do not span the heavy/light boundary. Each chain must satisfy the absolute-charge limit; cancellation between chains cannot bypass it.

The hard filter requires unmeasured status, a stability LCB above the configured floor, and bounded charge/hydrophobic/APR liabilities. The constraints are **never relaxed to fill a plate**. A run with fewer than 24 feasible candidates fails with a useful message. Raw means, adjusted means, penalties, uncertainties, and exclusion reasons remain inspectable.

### Multi-objective acquisition and diversity

`optimization.py` implements maximization Pareto dominance and exact two-dimensional hypervolume. Marginal Monte Carlo EHVI uses 256 correlated objective-posterior samples, an observed-outcome normalized reference point of `(-0.15, -0.15)`, and common random numbers. The observed frontier uses the same developability adjustment as predictions.

```text
exploration_weight = exploration_preference / (1 + n_training / 100)
acquisition = (1 − exploration_weight) × percentile(EHVI)
            + exploration_weight × percentile(normalized epistemic uncertainty)
```

An acquisition shortlist of at least 96 eligible candidates is enriched with the eligible posterior-mean and upper-confidence Pareto frontiers. A quality-weighted RBF determinantal point process greedily selects exactly 24 unique indices. Diversity=0 returns the acquisition ranking; increasing diversity emphasizes separation in embedding space. The selected set can include dominated candidates for exploration and diversity, particularly when the frontier has fewer than 24 members.

This is **marginal MC-EHVI plus a DPP batch heuristic**, not joint qEHVI/qNEHVI and not a proof of the globally optimal wet-lab batch. The surrogate captures covariance between the two objectives at each sequence; the acquisition does not model full cross-candidate batch posterior dependence. Sequential hypervolume fantasies, assay-replicate modeling, uncertainty calibration, and held-out validation are appropriate extensions for higher-stakes deployment.

### Enable AbLang-2 antibody embeddings

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ablang2.txt
```

Select **AbLang-2** in the dashboard for libraries containing complete antibody variable domains. The first run downloads the official pretrained weights; subsequent runs reuse cached weights and sequence embeddings. The package encodes sequences locally. Missing dependencies or model load failures fail the run explicitly. The benchmark remains on its published CDR fragments and uses a mutation-aware sequence kernel instead of applying AbLang-2 to invalid fragment context.

### Enable real ESM-2

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-esm.txt
$env:HF_HOME = Join-Path (Get-Location) 'runtime\huggingface'
.\.venv\Scripts\python.exe -m scripts.verify_esm
```

Then select **ESM-2** in the dashboard and run selection. The first model load downloads weights. Full masked likelihoods for hundreds of paired sequences require many forward passes and may take minutes to hours on CPU; GPU is used when PyTorch reports CUDA available. Identical chains are cached. Set `AEROFOLD_ESM_LOCAL_ONLY=1` after downloading weights for disconnected use. Missing dependencies or model load failures cause explicit failed runs; they never silently substitute descriptor features.

## Exports

| Export | Contents |
| --- | --- |
| Plate manifest | Exactly 24 rows, `A01`–`A12`, `B01`–`B12`, IDs, paired amino-acid sequences, prediction means/σ, and run ID |
| Protein synthesis order | One row per present chain, unique order names, separate H/L plates, wells, amino-acid sequences and `sequence_type=protein` |
| Assay template | 24 selected IDs and blank binding/stability fields for re-import |
| Run manifest | JSON including all predictions, selected IDs, training IDs, configuration, revision, provenance and warnings |

The plate CSV is a **generic robot import manifest**, not an executable Hamilton/Opentrons/Tecan method. Source plates, liquid classes, concentrations, transfer volumes, and instrument-specific instructions are deliberately not invented. Protein synthesis export does not invent DNA sequences; obtain vendor-specific codon optimization, cloning context and QC before gene ordering.

## Service operations

- SQLite WAL transactions persist libraries, measurements, revisions and run snapshots under `runtime/` (override `AEROFOLD_DATA_DIR`). Back up the database consistently, including WAL state or using SQLite's backup API.
- One embedded worker executes jobs; one additional job may queue. Excess submissions return HTTP 429. Shutdown waits for active work; on a hard restart, unfinished jobs become failed with an explanation.
- **Run one Uvicorn process (`--workers 1`)**. For multiple API replicas, move persistence to a shared database and job execution to a durable queue; the in-process queue/recovery is intentionally single-process.
- Optional `AEROFOLD_API_TOKEN` protects all experiment endpoints. Enter the token using **API connection**; it is held in session storage, not persistent local storage. Use a strong secret, HTTPS and authenticated reverse proxy before remote exposure. This is a single-tenant workspace, not a multi-tenant identity/access-control system.
- Uploaded filenames never become server paths. SQL uses bound parameters, model input schemas reject extra fields and non-finite scores, and sequence IDs cannot inject spreadsheet formulas. Uploads and JSON bodies are bounded; place rate and request-size limits at the reverse proxy as defense in depth.
- ESM feature caches and the database contain sequences or derived proprietary information; treat the data volume as sensitive project data. Audit exports contain sequences and assay scores.

### Docker

```bash
docker compose up --build
```

Open port 8000 on localhost. The multi-stage image builds the frontend and runs as a non-root user. It persists `/app/runtime` in a named volume and exposes a health check. The included image uses CPU Torch and offline descriptors; ESM is optional. Docker configuration is supplied for deployment convenience; local validation is described below.

## Tests and reproducibility

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q --basetemp=runtime/pytest
.\.venv\Scripts\python.exe -m ruff check backend scripts tests
cd frontend
npm.cmd run build
```

Test coverage includes parsing, invalid and duplicate inputs, indel tracking, exact hypervolume identities, DPP cardinality and feasibility, GP uncertainty away from observed data, transactional ingestion, API authentication, exports, stale-run protection, hidden-outcome isolation, and sequential learning. The optional `scripts.verify_esm` performs a real pretrained model and cache smoke test. It is intentionally separate from offline unit tests.

The reproducible generator is `python -m scripts.generate_demo`. `data/demo_library.csv` has only seed labels. `data/demo_oracle.csv` contains all 500 **synthetic** outcomes for verification; never upload it as your candidate library when demonstrating hidden-outcome learning. The pipeline module never imports the oracle. `data/demo_library.fasta` provides paired-chain sequences without measurements.

`requirements.txt` specifies supported dependency ranges; `requirements-lock.txt` records the exact core/development/optional-ESM environment validated on Python 3.12/Windows. `frontend/package-lock.json` locks frontend packages. ESM weights are pinned to checkpoint revision `c731040fcd8d73dceaa04b0a8e6329b345b0f5df`. Runtime behavior may differ across hardware; seeds fix acquisition samples and demo generation, not every GPU kernel. Run manifests record runtime versions, a dataset hash, and the exact training observation values.

## Benchmark Mode: public antibody DMS

Open **Benchmark mode** in the sidebar and click **Run benchmark**. The default run starts both agents with the same 20 random observations, then selects five batches of 24. Controls support 1–12 rounds, batches of 12/24/48, seeds, exploration, diversity, and a confidence-bound feasibility floor. These benchmark controls do not alter the lab selector's exactly-24 constraint. No model or dataset download is needed at runtime.

The dashboard streams completed rounds, compares recall, Spearman correlation and hypervolume, and exports the complete audit JSON. It remembers the latest job in this browser; re-entering Benchmark Mode restores its results. Jobs and partial rounds are persisted in SQLite. Interrupted jobs are marked failed on server restart, retaining completed rounds. The shared bounded compute queue accepts at most two jobs, including wet-lab selections; use one API process.

The packaged **512-variant 4-4-20 Tite-Seq subset** contains genuine measurements from [Adams et al., eLife (2016)](https://doi.org/10.7554/eLife.23156), sourced from the [authors' repository](https://github.com/jbkinney/16_titeseq). Binding is pKd (higher is better). The `stability_score` column is a transformed **expression proxy, not measured structural stability**. The parent and variant sequences contain only the two assayed 10-residue CDR fragments; they are not full heavy chains and should not be uploaded to the full-domain workbench. See [data/benchmarks/README.md](data/benchmarks/README.md) for the schema, exact preprocessing and attribution.

### Run through the API

With the backend running, use PowerShell:

```powershell
$catalog = Invoke-RestMethod http://127.0.0.1:8000/api/benchmark/datasets
$parameters = @{ dataset_id='titeseq_4_4_20'; rounds=5; batch_size=24; seed_size=20; seed=42; exploration=0.35 } | ConvertTo-Json
$job = Invoke-RestMethod http://127.0.0.1:8000/api/benchmark/run -Method Post -ContentType 'application/json' -Body $parameters
do {
    Start-Sleep -Seconds 1
    $job = Invoke-RestMethod "http://127.0.0.1:8000/api/benchmark/runs/$($job.id)"
    Write-Host $job.progress
} while ($job.status -in @('queued', 'running'))
$job.result.summary
$job | ConvertTo-Json -Depth 30 | Set-Content benchmark-audit.json
```

If authentication is configured, pass `-Headers @{Authorization="Bearer $env:AEROFOLD_API_TOKEN"}` to each request. `POST /api/benchmark/run` returns HTTP 202 with a job ID and `poll_url`; poll `GET /api/benchmark/runs/{id}` for `queued`, `running`, `completed` or `failed`. `result.rounds` contains paired metrics as each round completes. A completed job may have `result.status = stopped_early` and a `stop_reason` when either agent cannot fill the next feasible batch. Invalid parameters/budgets return 422, unknown jobs 404, and a full compute queue 429.

### Evaluation protocol

- Both agents fit the shared joint two-output PyTorch exact GP on their own revealed observations. AeroFold uses correlated-output MC-EHVI and greedy DPP modules; random selection samples uniformly without replacement among its own model-feasible candidates. Their fits run concurrently, with separate observed histories and seeded RNG streams.
- The benchmark uses exact position/residue mutation contrasts and CDR-specific mutation counts from the two assayed fragments. It does **not** claim AbLang-2 or ESM embeddings or apply full-chain developability heuristics to incomplete fragments. Both methods receive identical representations and modeling choices.
- Binding posterior means are clipped to the dataset's documented pKd assay range [5, 9.5] before policy scoring and retrospective rank evaluation; scores outside that range are not distinguishable measurements. The Gaussian uncertainty remains an approximation to the censored assay.
- Unseen oracle labels never enter GP fitting, acquisition, feasibility, normalization of model targets, or selection. Feasibility is predicted expression mean minus β times latent uncertainty, using each agent's observations. This does not guarantee measured feasibility. Both batches are planned before either is revealed. If either has too few candidates, both stop without relaxing the floor or consuming unequal budgets.
- Round 0 is the shared seed set. Later metrics include the newly revealed batch and refitted models. Recall is the percentage of the library's absolute top 5% binders found, with all ties at the cutoff included. The denominator includes binders that the learned feasibility gate might exclude. Enrichment is `(top hits / observed count) / (top binders / library size)`, not a comparison against the other policy.
- Spearman uses average tie ranks and the **common still-unmeasured intersection** for both agents in each round. Evaluation IDs/counts are exported. The cohort changes between rounds; correlations across rounds are not a fixed-test-set learning curve. Constant/empty cohorts produce JSON null rather than fabricated zero correlation.
- Cumulative hypervolume maximizes revealed binding and expression, including every observed seed/batch outcome. Fixed per-objective min/range scaling from the packaged oracle and reference `[-0.05,-0.05]` are used **only for retrospective reporting**, never selection. Hypervolume is dimensionless and includes observed points below the selection floor.
- Signed differences and active/random recall ratios are reported without assuming active learning wins. The ratio is null when random recall is zero. A single run on one subset is an algorithm diagnostic, not evidence of general superiority. In a local ten-seed check (seeds 0–9, five rounds, 20 seed assays, batches of 24, default controls), active selection won top-binder recall in 8/10 runs, with mean +18.1 percentage points. Mean correlation difference was −0.062 and mean hypervolume difference was +0.007; the same landscape and ten seeds do not establish generalization. The UI reports each actual outcome.

Source checksums, the pinned upstream commit, preprocessing, configuration, selected/observed IDs, evaluation IDs, software versions and normalization constants are included in audits. Repackage explicitly with `python -m scripts.package_benchmark` (network required if sources are not cached); the application always uses the local checksum-verified CSV.

## Project map

```text
backend/
  models.py         Validated sequences, measurements, selection constraints
  sequences.py      FASTA/CSV parsing, alignment, CDRs and mutation tracking
  features.py       Local/mutation descriptors, optional AbLang-2 and ESM-2
  surrogate.py      Joint PyTorch exact GP and latent objective covariance
  optimization.py   Pareto, hypervolume, MC-EHVI and greedy DPP
  pipeline.py       Reproducible model → constraint → acquisition → batch flow
  learning.py       Shared objective fitting and acquisition scoring
  benchmarks.py     Public DMS oracle evaluator and paired learning simulation
  storage.py        Transactional revisioned SQLite persistence
  main.py           FastAPI, background jobs, auth, exports and frontend serving
  demo.py           Seeded synthetic library and isolated assay oracle
frontend/src/       React dashboard, SVG scatter, sequence heatmap, Recharts benchmark
data/               Ready-to-upload 500-variant examples
data/benchmarks/    Public antibody DMS subset, source manifest and processing notes
tests/              Scientific invariants and full API workflow tests
scripts/            Demo generation and optional real-ESM verification
```

## Method references and scope

- [ESM implementation and model documentation](https://huggingface.co/docs/transformers/model_doc/esm)
- [Official ESM repository](https://github.com/facebookresearch/esm)
- [Official AbLang-2 repository](https://github.com/oxpig/AbLang2)
- [BoTorch multi-objective optimization tutorials](https://botorch.org/docs/tutorials/)
- [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/)

This is a functional, deployment-oriented **research workbench**, not an experimentally validated therapeutic discovery system. Neither naturalness nor sequence liabilities guarantee folding, binding, manufacturability or clinical performance. Real assay calibration, validated antibody numbering, empirical uncertainty assessment, structural/biophysical follow-up and operational review remain necessary before relying on it in a production therapeutic program.
