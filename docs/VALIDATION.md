# Validation record

Validated locally with Python 3.12 on Windows and Node.js 22.16.0. Exact package versions are in `requirements-lock.txt` and `frontend/package-lock.json`.

- **28 pytest tests passed.** Covers scientific identities, GP uncertainty, exact-24 selection, strict constraints, duplicates, non-finite scores, request limits (including chunked bodies), transactional ingestion, authentication, export shape and stale-run rejection. Benchmark tests also verify packaged-source checksum integrity, tied/undefined correlations, deterministic replay, equal budgets, monotonic cumulative metrics, hidden-oracle isolation, strict early stopping, random feasibility, configurable batches, persisted job recovery and bounded queues.
- **Ruff passed** for backend, scripts and tests.
- **React/TypeScript production build passed.** npm dependency audit reported zero vulnerabilities during installation.
- **Real pretrained ESM-2 check passed**, using the pinned `facebook/esm2_t6_8M_UR50D` checkpoint. A heavy-chain domain produced a `(1, 320)` embedding, masked pseudo-perplexity approximately `10.0365`, and identical cached results. The check also passed with local-only weights.
- **Browser workflow checked:** initial demo selection; parent-relative heavy-chain mutation heatmap; variant inspection; simulated next round; export dialog and successful HTTP 200 plate-manifest request. The simulated round increased measured candidate count from 48 to 72 and produced 24 fresh selections. Both narrow and desktop layouts were inspected. The browser automation download-event observer timed out, so the browser's saved-file location was not independently confirmed; CSV response content/cardinality is covered by API tests.

Not validated here: Docker image execution (Docker unavailable), hosted CI execution, large concurrent load, a full 500-variant ESM masked-likelihood run, GPU behavior, standardized antibody numbering, real-assay calibration, or experimental structural preservation. The complete ESM library run is intentionally not part of the fast offline test suite.

The local tests emit one dependency deprecation warning: Starlette's TestClient plans to replace its `httpx` integration with `httpx2`. It does not affect the passing tests or application runtime.

## Benchmark Mode validation (2026-09-28)

The public 512-variant Tite-Seq dataset was packaged from three pinned upstream replicate files and checksum-verified. The five-round default simulation was exercised through the browser with 20 shared seed assays and 24 picks per round, producing 140 observations per agent. Final top-binder recall was **7.692% AeroFold / 26.923% random**, Spearman rho **0.38944 / 0.41519**, and normalized hypervolume **0.79660 / 0.94673**. Random won on this seed; the application retains the negative result. This benchmark uses fragment positional descriptors and an expression proxy, not pretrained full-chain embeddings or folding-stability measurements.

Browser checks covered background progress, all three chart choices, a second run with one batch of 12 (32 assays per agent), null recall-ratio handling when the random agent finds no top binders, recovery of persisted results after a server restart and page reload, and the final five-round results table. Browser console showed no warnings or errors. The final production build passes TypeScript and Vite, with Benchmark Mode loaded as a separate bundle. A screenshot is saved as `docs/benchmark-dashboard.png`.
