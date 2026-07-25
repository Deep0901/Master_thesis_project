# Rule Weight Sensitivity Analysis

| Configuration | Description | Metadata | Similarity | Recency | Resources | MAP | P@5 | nDCG@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Original rule weights (all weights = 1.0) | 1.00 | 1.00 | 1.00 | 1.00 | 0.4828 | 0.3067 | 0.5330 | 0.5333 |
| Metadata Focus | Emphasize metadata quality and resource rules | 1.30 | 1.00 | 1.00 | 1.20 | 0.4828 | 0.3067 | 0.5330 | 0.5333 |
| Similarity Focus | Increase emphasis on thematic-similarity rules | 1.00 | 1.30 | 1.00 | 1.00 | 0.4422 | 0.2933 | 0.5004 | 0.4800 |
| Recency Focus | Increase emphasis on recency rules | 1.00 | 1.00 | 1.30 | 1.00 | 0.4828 | 0.3067 | 0.5330 | 0.5333 |
| Balanced Conservative | Reduce all rule weights for a more conservative aggregation | 0.80 | 0.80 | 0.80 | 0.80 | 0.4828 | 0.3067 | 0.5330 | 0.5333 |
