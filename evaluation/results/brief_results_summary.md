# Brief Results Summary

- Runs: 15 benchmark queries, 5 systems compared (Fuzzy, BM25, Linear Weighted, Weighted Sum, Portal).
- Key metrics: MAP, P@5, nDCG@10, MRR.
- Observations:
  - `Fuzzy` and `BM25` show identical aggregated metrics in this run (means and bootstrap CIs equal), indicated by `identical_vectors` notes in pairwise statistics.
  - Linear Weighted and Weighted Sum perform slightly lower on mean MAP and nDCG; differences are not statistically significant after Holm correction (all corrected p-values = 1.0).
  - Bootstrapped CIs are reported in `bootstrap_confidence_intervals.csv` and system means in `system_summary.csv`.
- Reproducibility:
  - Frozen snapshot: `data/raw/ogd_metadata_20260306_183841.json` (sha256 in reproducibility report).
  - Pinned semantic model: `paraphrase-multilingual-MiniLM-L12-v2` (seed=42) — recorded in reproducibility report; semantic baseline excluded due to runtime imports unless dependencies are installed.

Files of interest:
- `evaluation/results/system_summary.csv`
- `evaluation/results/pairwise_statistics.csv`
- `evaluation/results/bootstrap_confidence_intervals.csv`
- `evaluation/results/reproducibility_report.json`

Next recommended actions:
- If you want the semantic baseline included, install full dependencies (`transformers`, `huggingface-hub==0.13.3`) and re-run the experiment.
- Prepare Appendix A audit artifacts (export annotated CSVs, provenance logs) — I can do this next if you want.
