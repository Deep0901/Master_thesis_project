# Rule Weight Sensitivity Analysis

| Configuration | Description | Metadata | Similarity | Recency | Resources | MAP | P@5 | nDCG@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Original rule weights (all weights = 1.0) | 1.00 | 1.00 | 1.00 | 1.00 | 0.8191 | 0.8000 | 0.8613 | 0.8667 |
| Metadata Focus | Emphasize metadata quality and resource rules | 1.30 | 1.00 | 1.00 | 1.20 | 0.8191 | 0.8000 | 0.8613 | 0.8667 |
| Similarity Focus | Increase emphasis on thematic-similarity rules | 1.00 | 1.30 | 1.00 | 1.00 | 0.8163 | 0.7867 | 0.8606 | 0.8667 |
| Recency Focus | Increase emphasis on recency rules | 1.00 | 1.00 | 1.30 | 1.00 | 0.8191 | 0.8000 | 0.8613 | 0.8667 |
| Balanced Conservative | Reduce all rule weights for a more conservative aggregation | 0.80 | 0.80 | 0.80 | 0.80 | 0.8191 | 0.8000 | 0.8613 | 0.8667 |
