# Fuzzy Ranking Ablation Summary

## Code Changes

- Added `LinearWeightedBaselineAdapter` to `evaluation/experiment_runner.py`.
- Reused the fuzzy framework's query processing, metadata completeness, similarity calculator, normalized recency score, and normalized resource score.
- Used the same default feature weights as `FuzzyHCIRRanker.rank`: recency=1.0, completeness=1.0, resources=1.0, similarity=1.0.
- Removed only the fuzzy inference layer for the ablation score.
- Registered the baseline in the complete benchmark with Portal, BM25, Weighted Sum, Linear Weighted, and Fuzzy systems.
- Added ablation CSV and interpretation summary outputs.

## Mean Metric Comparison

| System | MAP | nDCG@10 | P@5 | MRR |
|---|---:|---:|---:|---:|
| Portal | 0.2840 | 0.3489 | 0.2400 | 0.2792 |
| BM25 | 0.4972 | 0.5362 | 0.3200 | 0.5500 |
| Weighted Sum | 0.2840 | 0.3489 | 0.2400 | 0.2792 |
| Linear Weighted | 0.3995 | 0.4239 | 0.2800 | 0.3550 |
| Fuzzy | 0.4306 | 0.4586 | 0.2800 | 0.4111 |

## Fuzzy Layer Effect

- MAP: Fuzzy is higher than Linear Weighted by 0.0311.
- nDCG@10: Fuzzy is higher than Linear Weighted by 0.0348.
- P@5: Fuzzy is equal to Linear Weighted.
- MRR: Fuzzy is higher than Linear Weighted by 0.0561.
- nDCG@10 win/loss/tie count for Fuzzy versus Linear Weighted: 4/2/9.

Interpretation: Linear Weighted isolates the contribution of the normalized feature set without fuzzy rules. Any metric difference between Fuzzy and Linear Weighted is therefore attributable to the fuzzy inference layer plus the fuzzy framework's final blend with the linear factor score, not to candidate data, feature extraction, or feature weights.
