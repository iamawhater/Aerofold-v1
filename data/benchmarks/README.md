# 4-4-20 antibody Tite-Seq benchmark

`titeseq_4_4_20.csv` contains 512 **experimental**, not synthetic, antibody variants from Adams RM, Mora T, Walczak AM, Kinney JB (2016), *Measuring the sequence-affinity landscape of antibodies with massively parallel titration curves*, [eLife 5:e23156](https://doi.org/10.7554/eLife.23156).

Source: [jbkinney/16_titeseq](https://github.com/jbkinney/16_titeseq), pinned commit `54130c8fc6081bbfa8301c41e1afcfa6a5d79590`, `data/replicate_1.csv`, `replicate_2.csv`, `replicate_3.csv`. Per-file URLs and SHA256 hashes are in `titeseq_4_4_20.json`. Public factual measurements are attributed here; no upstream code is redistributed. The associated eLife article is CC BY 4.0; the source repository has no separate license file. Retain source attribution and provenance.

| Column | Meaning |
| --- | --- |
| `variant_id` | Stable local ID after deterministic subset selection |
| `parent_sequence` | `TFSDYWMNWVGSYYGMDYWG`, concatenated wild-type assayed CDR1H/CDR3H fragments |
| `variant_sequence` | Corresponding concatenated 20-aa fragments; **not a full heavy chain** |
| `variant_string` | Substitutions with one-based, fragment-local coordinates (`CDR1H:S3G`, etc.) |
| `true_binding` | Median experimental pKd, clipped to assay range [5, 9.5]; maximize |
| `stability_score` | `100 * relative_expression / (1 + relative_expression)`; expression proxy only, maximize |
| `relative_expression` | Expression divided by replicate wild-type median, then aggregated |
| `replicate_count` | Number of available experimental replicates, at least two |
| `stability_kind` | Identifies the expression-based proxy |

Packaging filters canonical 20-aa pairs, positive finite Kd and finite nonnegative expression. It groups synonymous records within each replicate using medians, normalizes expression to that replicate's wild type, requires two or more replicates, and takes across-replicate medians. Wild type is excluded from the candidate pool. It retains the 512 lowest SHA256 ranks of `aerofold-titeseq-v1:` plus the amino acid pair among 2,794 qualifying variants, then orders sequences lexically for IDs. Subset selection is independent of outcome values after quality filtering. Affinity values clipped at assay limits remain censored measurements; they are not precise affinities beyond those limits.

Regenerate from the project root using `python -m scripts.package_benchmark`. Source files are cached in `runtime/benchmark-source/`; only the processed subset and manifest are packaged. Runtime simulation does not use the network. The loader rejects a CSV whose checksum differs from the manifest. The full local oracle is intended for benchmark evaluation; policy functions only receive revealed labels.

This dataset does not contain folding-stability measurements or full variable-domain sequences. The benchmark uses exact fragment mutation features and a joint binding-expression GP, avoiding invented framework sequence or pLM context. It does not evaluate the optional full-domain AbLang-2 or ESM mode. Results cannot establish therapeutic or wet-lab performance.
