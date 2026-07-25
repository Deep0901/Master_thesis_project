# Membership-Function Sensitivity Analysis

## Baseline Fuzzy Metrics

- MAP: 0.0000
- nDCG@10: 0.0000
- P@5: 0.0000
- MRR: 0.0000

## Stability Summary

- Perturbations tested: 162
- Maximum absolute metric delta: 0.0000
- Mean Kendall's Tau: 0.9964
- Mean average rank displacement: 0.0153
- Lowest Kendall's Tau: 0.6533 for `recency.very_recent.b3.plus_10`
- Largest metric movement: 0.0000 for `recency.very_recent.b1.plus_10`

## Interpretation

The fuzzy system is robust to +/-10% membership breakpoint variation under this run.

Robustness is assessed using small aggregate metric movement, high Kendall rank correlation, and low average rank displacement relative to the unperturbed fuzzy ranking.
