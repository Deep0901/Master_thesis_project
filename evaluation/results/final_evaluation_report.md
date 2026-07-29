# Final evaluation report

This report is based on the committed evaluation artifacts in the repository and reflects the final production run of the benchmark pipeline.

## 1. Scope and provenance

- Benchmark queries: 15
- Systems evaluated: 6 (Fuzzy, BM25, Linear Weighted, Weighted Sum, Portal, Semantic)
- Corpus snapshot used for the run: data/raw/ogd_metadata_20260306_183841.json
- Ground-truth source: evaluation/ground_truth_final.json
- Reproducibility metadata: evaluation/results/reproducibility_report.json
- Evaluation metric family: MAP, P@5, nDCG@10, and MRR

## 2. Research questions and terminology

The report uses the following research questions consistently:

- RQ1: How can fuzzy logic be used to effectively retrieve Open Government Datasets within OGD portals?
- RQ2: How can multiple metadata-based relevance criteria be integrated into a unified and interpretable ranking mechanism?
- RQ3: To what extent does a fuzzy logic-based retrieval system support Human-Centered Information Retrieval principles?
- RQ4: How effective are metadata-driven explanations in improving user understanding and trust in ranking decisions?

The report uses the term "resource availability" consistently for the fourth fuzzy input; no alternative labels such as "organizational quality" or "dataset quality" are used in this report.

## 3. Evaluation protocol and evidence

The quantitative benchmark uses a frozen local corpus snapshot and a fixed ground-truth file. The repository records the exact corpus hash, ground-truth hash, model name, and reproduction command needed to regenerate the published artifacts.

The evaluation protocol in the repository is documented as a single-assessor workflow for relevance grading. This report does not claim inter-annotator agreement or adjudication because those claims are not supported by the repository artifacts.

User-study evidence is captured through a 10-item questionnaire recorded in evaluation/results/user_study_audit.md. The repository includes the questionnaire instrument and its audit summary. No anonymized item-level response matrix is present in the checked repository snapshot, so the report avoids unsupported claims about causal effects or aggregate participant-level results beyond the documented instrument.

## 4. Aggregate results

| System | MAP | P@5 | nDCG@10 | MRR |
|---|---:|---:|---:|---:|
| Semantic | 0.543 | 0.347 | 0.574 | 0.600 |
| BM25 | 0.497 | 0.320 | 0.536 | 0.550 |
| Fuzzy | 0.431 | 0.280 | 0.459 | 0.411 |
| Linear Weighted | 0.400 | 0.280 | 0.424 | 0.355 |
| Weighted Sum | 0.284 | 0.240 | 0.349 | 0.279 |
| Portal | 0.284 | 0.240 | 0.349 | 0.279 |

## 5. Statistical significance and ablation

The committed output contains the full statistical evidence required for the report:

- Wilcoxon statistic, original p-value, Holm-corrected p-value, and significance flag are recorded in evaluation/results/pairwise_statistics.csv.
- Bootstrap confidence intervals for pairwise mean differences are recorded in evaluation/results/bootstrap_confidence_intervals.csv.
- Win/loss/tie counts are recorded in evaluation/results/win_loss_matrix.csv.
- The ablation comparison and interpretation summary are recorded in evaluation/results/ablation_interpretation_summary.md.

The current evidence supports a cautious descriptive interpretation: the benchmark is a frozen-snapshot evaluation over 15 queries, and none of the reported pairwise comparisons reached significance after Holm correction.

## 6. Sensitivity analysis

The rule-weight sensitivity analysis reported identical aggregate outcomes across the tested configurations:

- Baseline: MAP 0.483, P@5 0.307, nDCG@10 0.533, MRR 0.533
- Metadata Focus: MAP 0.483, P@5 0.307, nDCG@10 0.533, MRR 0.533
- Similarity Focus: MAP 0.442, P@5 0.293, nDCG@10 0.500, MRR 0.480
- Recency Focus: MAP 0.483, P@5 0.307, nDCG@10 0.533, MRR 0.533
- Balanced Conservative: MAP 0.483, P@5 0.307, nDCG@10 0.533, MRR 0.533

## 7. Reproducibility notes

The reproducibility report records the frozen corpus hash, the ground-truth hash, the number of queries, and the sentence-transformer model used for the semantic baseline. The lightweight verifier in the reproducible runner checks that the current files still match those recorded hashes before the main evaluation is launched.

- Corpus file hash: a827392f6c788749a71ba3e19103f126e6bafb47ee07329f0301950bc15edfe1
- Ground-truth file hash: 385603b162df43d35f6884906d9faebd11063e6bdc3900fd220736e6d406e41f
- Semantic baseline model: paraphrase-multilingual-MiniLM-L12-v2
- Python version recorded in the reproducibility report: 3.11.9

Reproduction command:

```powershell
python evaluation/run_reproducible_experiment.py
```
