# Final evaluation report

This report is based on the committed evaluation artifacts in the repository and reflects the final production run of the benchmark pipeline.

## 1. Scope and provenance

- Benchmark queries: 15
- Systems evaluated: 6 (Fuzzy, BM25, Linear Weighted, Weighted Sum, Portal, Semantic)
- Corpus snapshot used for the run: data/raw/ogd_metadata_20260306_183841.json
- Ground-truth source: evaluation/ground_truth_final.json
- Reproducibility metadata: evaluation/results/reproducibility_report.json
- Evaluation metric family: MAP, P@5, nDCG@10, and MRR

## 2. Aggregate results

| System | MAP | P@5 | nDCG@10 | MRR |
|---|---:|---:|---:|---:|
| Semantic | 0.543 | 0.347 | 0.574 | 0.600 |
| BM25 | 0.497 | 0.320 | 0.536 | 0.550 |
| Fuzzy | 0.431 | 0.280 | 0.459 | 0.411 |
| Linear Weighted | 0.400 | 0.280 | 0.424 | 0.355 |
| Weighted Sum | 0.284 | 0.240 | 0.349 | 0.279 |
| Portal | 0.284 | 0.240 | 0.349 | 0.279 |

## 3. Statistical significance

The reported pairwise comparison file contains 60 comparisons across the four metrics. In the committed output table, none of the reported comparisons reached statistical significance after Holm correction.

This wording is deliberately cautious: the benchmark is a frozen-snapshot evaluation over 15 queries, and the current evidence supports a descriptive statement of no statistically significant pairwise differences rather than a stronger claim of practical equivalence.

## 4. Sensitivity analysis

The rule-weight sensitivity analysis reported identical aggregate outcomes across the tested configurations:

- Baseline: MAP 0.483, P@5 0.307, nDCG@10 0.533, MRR 0.533
- Metadata Focus: MAP 0.483, P@5 0.307, nDCG@10 0.533, MRR 0.533
- Similarity Focus: MAP 0.442, P@5 0.293, nDCG@10 0.500, MRR 0.480
- Recency Focus: MAP 0.483, P@5 0.307, nDCG@10 0.533, MRR 0.533
- Balanced Conservative: MAP 0.483, P@5 0.307, nDCG@10 0.533, MRR 0.533

## 5. Reproducibility notes

The reproducibility report records the frozen corpus hash, the ground-truth hash, the number of queries, and the sentence-transformer model used for the semantic baseline. The lightweight verifier in the reproducible runner checks that the current files still match those recorded hashes before the main evaluation is launched.

- Corpus file hash: a827392f6c788749a71ba3e19103f126e6bafb47ee07329f0301950bc15edfe1
- Ground-truth file hash: 385603b162df43d35f6884906d9faebd11063e6bdc3900fd220736e6d406e41f
- Semantic baseline model: paraphrase-multilingual-MiniLM-L12-v2
- Python version recorded in the reproducibility report: 3.11.9
