# Reviewed antibody assay tasks

These CSVs are transformed extracts from [FLAb](https://github.com/Graylab/FLAb), pinned to commit `e05e3f0135783ffb7de412df6a1467e7a0031911`.

`catalog.json` supplies the original paper title, DOI, source URL, source hash, processed CSV hash, original license/link, exact score column, transformation, and data-quality exclusions for every task. Consult the original publications when interpreting an assay. Full-domain sequence reconstructions are inherited from FLAb; this app does not independently resequence or experimentally verify them.

Changes made here: canonical full-domain filtering; explicit `−log10(Kd[M])` or `9−log10(Kd[nM])` transformations for raw affinities; median aggregation of identical heavy/light sequence pairs; and separation of nanobody melting temperatures by protocol. No label-dependent subsampling is used during packaging. Model compute budgets are separate, seeded sampling steps disclosed in results.

Credit the authors listed under each task's `citation` and the FLAb authors, Michael Chungyoun and Jeffrey J. Gray. Source articles under CC BY 4.0 retain their copyright; the manifest links to license terms. IgDesign retains the MIT notice in `licenses/IgDesign-MIT.txt`. The HER2 unlocking dataset retains the Clear BSD notice in `licenses/Unlocking-Clear-BSD.txt`. No author or institution endorsement is implied.

The total of 74,219 counts task-specific sequence/assay observations, not globally distinct antibodies. Full sequences are available for the new Tite-Seq task. H3's upstream `fitness` column is used because an adjacent H3 metadata column is mislabeled as H9; the task remains CR9114/H3. Expression is not thermodynamic stability. Separate temperature protocols are not pooled.

The source catalog also records studies not admitted because of unreviewed semantics, limited size, incomplete domains, licensing, or access requirements. Catalog inclusion does not mean those data trained a model.

Reproduce with `python -m scripts.sync_prediction_data --download` from the project root. The app checks processed-file integrity before use. Scores at detection limits are currently point labels; future models should treat them as censored.
