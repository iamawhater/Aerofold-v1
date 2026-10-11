# AeroFold-Ab — antibody prediction engine

A general platform with **separate supervised models for each target and assay**. The default workspace predicts and ranks every candidate; it no longer runs an active-learning acquisition loop. The previous workbench and saved runs remain accessible through **Open previous workspace** (`?legacy=1`).

The engine combines pretrained **paired AbLang-2 antibody embeddings** with additive mutation effects, second- and third-order interaction kernels, a local sequence kernel, and Extra Trees. Validation data choose regularization and nonnegative ensemble weights. Separate calibration measurements produce split-conformal intervals. A model's contribution can be zero when validation does not support it.

This is a tested research prediction platform, not a universally validated affinity oracle or a claim of state-of-the-art performance. Predicting a new antigen requires measurements for that antigen.

## Start locally

Python 3.11+ and Node 20+. PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
cd frontend
npm ci
npm run build
cd ..
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open **http://127.0.0.1:8000**. API docs: **http://127.0.0.1:8000/docs**. Restart an already-running backend after this update; a stale backend returns 404 for new endpoints. For frontend development, `npm run dev` in `frontend/` serves port 5173 and proxies the API to port 8000.

AbLang-2 is included in the base requirements. First use downloads approximately 179 MB of weights through the official package and requires network access and write access to the virtual environment. Embeddings are cached in `runtime/ablang2-cache`. **Sequence ensemble · offline / fast** needs no model download and is identified explicitly in each result. There is no silent fallback. ESM remains available in the previous workbench.

Docker prefetches weights during its build so its non-root runtime does not write into site-packages. Docker configuration is supplied; this update was verified locally on Windows, not with a Docker image build.

## Use the app

1. Choose **Prediction engine**, then the target and assay.
2. Choose AbLang-2 plus the sequence ensemble, or the offline ensemble. Set a labeled-data budget. **All available** removes that cap; calibration data remain separate from fitting data.
3. Upload candidate sequences, or leave the file blank for a held-out public-data demonstration. Demo query labels are excluded from fitting, validation and calibration.
4. Inspect rankings, native assay units, intervals, sequence support, and model weights. Download rankings as CSV or the full audit as JSON.
5. Use **Benchmark** for repeated hidden-test evaluations, or **Add assay data** to create your own target-specific model.

Candidate CSV columns: `variant_id,heavy_sequence,light_sequence`. Use complete canonical variable domains, 70–180 residues per chain; light domains are optional for VHHs. Add `score` for experimental training CSVs and specify target, units and direction. Custom scores are **not automatically transformed**. Keep one antigen, scaffold family and assay protocol per upload. Exact sequence-pair replicates are grouped by median before any split.

Limits: 5 MB HTTP uploads; 2,000 prediction candidates; 12,000 training-upload rows; at least 60 unique training examples. The public held-out demo evaluates at most 512 queries. Full benchmark availability also depends on having sufficient outer-test examples.

Published affinity tasks use **pKd = −log10(Kd in molar)**; higher is better. Expression ratio and melting temperature remain distinct tasks. Expression is not structural stability. Scores from different antigens are not combined into a universal antibody-quality score.

## Public evidence

`data/public/` packages **74,219 task-specific sequence–assay records across 17 task definitions**, with **10 tasks** large enough for modeling. Counts are not globally unique antibodies. The source catalog records 165 upstream entries including exclusions. This is a curated subset, not all internet data.

| Task family | Records | Assay |
|---|---:|---|
| 4-4-20 / fluorescein | 2,807 binding + 2,803 expression | Tite-Seq pKd; yeast expression |
| CR9114 / influenza | 32,392 H1 + 32,767 H3 | Separate antigen-binding tasks |
| D44 / lysozyme | 2,048 | Binding pKd |
| Trastuzumab variants / HER2 | 422 | Binding pKd |
| IgDesign / seven antigens | 319 across seven tasks | SPR pKd; TSLP is large enough for modeling |
| NbThermo / nanobodies | 661 across four protocols | Melting temperature; three usable protocols |

Unlike the legacy 512-row fragment benchmark, the new 4-4-20 task contains complete published heavy/light domains supplied by FLAb and supports real AbLang-2 encoding.

Sources: [FLAb](https://github.com/Graylab/FLAb), [Tite-Seq](https://doi.org/10.7554/eLife.23156), [CR9114](https://doi.org/10.7554/eLife.71393), [D44](https://doi.org/10.1371/journal.pcbi.1007207), [IgDesign](https://github.com/AbSciBio/igdesign), [NbThermo](https://doi.org/10.1093/database/baad021), and [AbLang-2](https://github.com/oxpig/AbLang2).

FLAb snapshot: `e05e3f0135783ffb7de412df6a1467e7a0031911`. `data/public/catalog.json` records source/publication links, source and CSV hashes, transformations, exclusions, replicate counts, and original licenses. MIT and Clear BSD notices are retained in `data/public/licenses/`; CC BY attribution and license links are in the manifest. Restricted, unspecified-license, gated and unreviewed sources are excluded. Ginkgo GDPa1 was catalogued as requiring an account and acceptance of terms; it was not downloaded.

Reproduce the reviewed tasks:

```powershell
.\.venv\Scripts\python.exe -m scripts.sync_prediction_data --download
```

Explicit adapters use unit-bearing raw columns rather than pooling ambiguous `fitness` columns. Source framework reconstructions are inherited from FLAb; the app does not fabricate them or independently verify them experimentally.

## Model and uncertainty

Sparse chain-separated residue contrasts use a reference learned from fitting sequences. Candidate heads: ridge regression; Nyström degree-2, degree-3 and RBF kernels; Extra Trees; optional standardized AbLang-2 ridge. Nyström bases are capped at 256 components for CPU usability. AbLang-2 is frozen, not affinity-fine-tuned.

Roughly 60% of the labeled budget fits candidates, 20% selects hyperparameters and weights, and 20% calibrates intervals. After model selection, fitting and validation data are combined to refit the selected specifications. Calibration labels never influence predictions or ensemble weights. Finite-sample conformal quantiles use independent absolute residuals. Nominal coverage assumes exchangeability and does not guarantee conditional coverage or coverage on new scaffolds.

Head disagreement is a diagnostic, not calibrated Bayesian epistemic uncertainty. Queries with nearest positional identity below 90%, or unsupported chain layouts, are flagged and placed after supported predictions. This is an applicability heuristic, not an experimentally validated success probability. Position features are not IMGT-aligned.

## Honest benchmarking

The benchmark is a **supervised learning curve** with up to three training budgets. Test sequences are fixed across budgets within a seed. Exact sequence duplicates never cross partitions. Frozen label-free embeddings may be shared; supervised labels and fitted statistics are isolated.

- **Random split:** 20% of sequence pairs are held out before budget selection. Measures within-assay interpolation.
- **Mutation extrapolation:** whole higher-mutation-count strata are withheld relative to the unlabeled library consensus. This is a stress test, not a new-scaffold or new-antigen test. Insufficient strata cause an explicit error.
- Baselines: validation-tuned additive regression and constant mean. Expected random recall is exactly `k/n_test`; random enrichment is 1.
- Metrics: Spearman rho, RMSE, MAE, tie-inclusive top-5% recall, top-24 enrichment, interval coverage and width. For tiny test sets, k is capped at test size; selecting the whole test set cannot demonstrate ranking quality.
- Repeated seeds show means and descriptive recall ranges, not confidence intervals or independent biological replicates.
- Audits contain per-seed metrics, configurations, provenance, test IDs, split hashes, calibration IDs and model weights.

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_prediction --task titeseq_4420_binding --representation ablang2 --budget 768 --seeds 3 --output data/evaluations/my-benchmark.json
```

Add `--split mutation_extrapolation` for the stress test. CLI reports also record dependency versions and source-code hashes. Completed local evaluations are in `data/evaluations/`. Simple baselines can outperform the ensemble on individual tasks, splits or budgets.

### ESM-2-only and AntiBERTy-only benchmark predictors

Install the optional encoders, then compare all three predictors in one report:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark-models.txt
.\.venv\Scripts\python.exe -m scripts.benchmark_prediction --predictors ensemble esm2-only antiberty-only --budget 768 --seeds 3 --output data/evaluations/foundation-comparison.json
```

For just the standalone predictors, use `--predictors esm2-only antiberty-only`. The existing script default remains `--predictors ensemble`; `--representation` selects the representation of that ensemble only.

In the website, open **Benchmark** and use **Model representation** to select **ESM-2-only**, **AntiBERTy-only**, or **Compare all three predictors**. Comparison mode also lets you choose the ensemble representation. Click **Run benchmark**: separate model curves appear in the chart, **Summary predictor** switches the metrics cards, and **Export full audit** includes all predictors and checkpoint provenance. Saved older ensemble results remain viewable. These options apply to benchmarks; the production prediction controls are unchanged.

The same API accepts `predictors: ["ensemble", "esm2-only", "antiberty-only"]`, optional `embedding_batch_size` (1–128, default 16), and `local_only` (default false). For a single standalone model, send just its name in `predictors`. The existing default is the ensemble alone. `representation` affects only the ensemble, and `local_only` affects only ESM-2/AntiBERTy checkpoint downloads. Jobs retain the existing progress polling, history, and JSON export endpoints.

Each standalone model uses **only its frozen foundation embeddings** with a supervised `StandardScaler + Ridge` head, validation-selected regularization and independent 90% conformal calibration. It adds no mutation descriptors, masked likelihoods, or ensemble blending. Both average final-layer residue states, exclude special/padding tokens, and concatenate the heavy and light chain vectors. Missing chains use zero-filled blocks. They use the exact same seeds, outer test identities, nested labeled budgets and internal calibration splits as the ensemble.

Checkpoints are pinned: **ESM-2 t6 / 8M / UR50D**, `facebook/esm2_t6_8M_UR50D` at `c731040fcd8d73dceaa04b0a8e6329b345b0f5df`; **AntiBERTy**, `jeffruffolo/AntiBERTy` at `d582006e0c7dc6917d26fa60740a357e8df7b466`, using official package `antiberty==1.0.0`. These are supervised frozen-embedding baselines, not zero-shot affinity predictions; the ESM comparator is the 8M checkpoint, not a larger 650M model. See [official ESM](https://github.com/facebookresearch/esm) and [official AntiBERTy](https://github.com/jeffreyruffolo/AntiBERTy).

Weights cache under `runtime/huggingface/hub`; embeddings under `runtime/benchmark-embedding-cache`. Use `--embedding-batch-size 8` to reduce memory, or `--local-only` to require cached ESM/AntiBERTy checkpoints. Missing dependencies or checkpoints fail explicitly, without substituted predictors. Reports add separately named metric series, predictor cards, fitting/calibration identities, checkpoint revisions, actual weight/config SHA-256 hashes, pooling rules, and dependency versions. First-time AntiBERTy encoding downloads roughly 100 MB of weights.

## API and operations

| Endpoint | Purpose |
|---|---|
| `GET /api/prediction/tasks` | Catalog, provenance and custom assays |
| `POST /api/prediction/tasks` | Multipart training CSV; name, target, unit, maximize, category |
| `POST /api/prediction/run` | JSON config; optional query sequence records |
| `POST /api/prediction/upload` | Candidate CSV plus JSON config form field |
| `POST /api/prediction/benchmark/run` | Submit hidden-test benchmark |
| `GET /api/prediction/jobs` | Saved job history |
| `GET /api/prediction/jobs/{id}` | Progress, errors and results |
| `GET /api/prediction/jobs/{id}/export?format=csv` | Prediction rankings |
| `GET /api/prediction/jobs/{id}/export?format=json` | Full audit |

Example:

```powershell
$config = @{task_id='titeseq_4420_binding'; representation='ablang2'; train_limit=768; repeats=3; seed=42; split='random'; test_limit=512; top_k=24} | ConvertTo-Json
$job = Invoke-RestMethod http://127.0.0.1:8000/api/prediction/benchmark/run -Method Post -ContentType 'application/json' -Body $config
do {
    Start-Sleep -Seconds 3
    $job = Invoke-RestMethod "http://127.0.0.1:8000/api/prediction/jobs/$($job.id)"
    $job.progress
} while ($job.status -in @('queued','running'))
$job.result.curves
```

SQLite persistence and the shared bounded executor allow one running job and one queued job. Interrupted jobs are marked failed after a server restart. Use **one Uvicorn worker**. `AEROFOLD_DATA_DIR` changes runtime storage. Set `AEROFOLD_API_TOKEN` before serving beyond a trusted local environment; enter it in Connection settings. This embedded executor is intended for single-host research deployments, not distributed production.

## Verification and remaining scientific work

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp runtime/pytest-local
.\.venv\Scripts\python.exe -m ruff check backend tests scripts
cd frontend
npm run build
```

Tests cover data integrity, unit conversion, deduplication, split independence, hidden-test label isolation, calibration-label independence, custom imports, API authentication, persistence, exports, and legacy regression behavior.

Stronger real-world claims require scaffold/lineage-held-out and prospective wet-lab validation; antigen-conditioned models with matched labels; IMGT-aware alignment; replicate- and censor-aware likelihoods; paired multi-property measurements and a validated joint covariance model; and controlled comparisons against larger antibody encoders or supervised fine-tuning. This implementation does not claim those validations or universal superiority. Censored assay endpoints are currently point labels; pretraining overlap is unknown.

Earlier active-learning architecture, ESM settings, synthetic demos and legacy APIs are documented in `docs/LEGACY_WORKSPACE.md`. Historical PDF assessments describe the earlier version, not this prediction engine.
