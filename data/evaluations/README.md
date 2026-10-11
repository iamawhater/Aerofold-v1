# Local supervised evaluation — 29 September 2026

These are actual runs of `python -m scripts.benchmark_prediction`, not synthetic oracle scores. Each uses 768 labeled development/calibration examples, top-24 ranking, and seeds 42–44. Smaller-budget curves, split identities, model weights, package versions and source checksums are in the adjacent JSON files.

`foundation_only_smoke.json` was added on 30 September to verify the ESM-2-only and AntiBERTy-only integrations using actual pretrained weights. It uses 128 labeled examples, one seed and 64 hidden test sequences, and is excluded from the 768-example comparison below. Its small test set is an integration check, not evidence for choosing the best predictor.

| Task / representation / split | Test rows per seed | Ensemble rho | Additive rho | Ensemble enrichment | Additive enrichment | 90% interval coverage |
|---|---:|---:|---:|---:|---:|---:|
| Tite-Seq / AbLang-2 ensemble / random | 512 | 0.591 | 0.561 | 9.85× | 8.21× | 90.4% |
| Tite-Seq / AbLang-2 ensemble / mutation extrapolation | 302 | 0.278 | 0.279 | 5.24× | 4.46× | 93.6% |
| CR9114 H1 / sequence ensemble / random | 512 | 0.953 | 0.907 | 8.21× | 1.37× | 89.9% |
| D44 / sequence ensemble / random | 410 | 0.399 | 0.395 | 1.59× | 1.59× | 90.2% |

Enrichment is the prevalence of true top-5% binders among the 24 predictions divided by their prevalence in the entire held-out test set. Expected random enrichment is 1. Top-score cutoff ties are included. Averages describe these seeds and assay libraries, not independent biological replicates or confidence intervals.

For Tite-Seq random splits, mean top-binder recall is 46.15% versus 4.69% expected random recall. Additive regression reaches 38.46%. Ensemble RMSE is 1.011 pKd versus 1.079 for additive regression. The frozen AbLang-2 head receives validation-derived weights of 31.8%, 16.1%, and 5.7%; other sequence heads remain important. This comparison does **not** isolate the causal benefit of AbLang-2; a matched sequence-only ensemble ablation would be needed for that claim.

The mutation-extrapolation test is materially harder. Its correlation is only 0.278 and the ensemble's RMSE (1.032) is worse than the constant-mean predictor (0.876), despite better top-binder enrichment. D44's ensemble offers essentially no improvement over additive regression at the final budget. These limitations are part of the result, not reasons to hide those tasks.

The benchmark specification and model-selection rule were fixed before these reports. Test scores were not used to select hyperparameters or ensemble weights. The evaluation is still an internal retrospective check, not an independent external leaderboard or prospective wet-lab validation. Random within-library splits can be optimistic; foundation-model pretraining overlap and assay censoring remain unresolved.
