# Human-Centered Information Retrieval for Swiss Open Government Data Using Fuzzy Logic-Based Ranking

**Author:** Deep Shukla

**Supervisor:** Janick Spycher

**Professor / Examiner:** Edy Portmann

**Institution:** University of Neuchâtel

---

## Overview

This repository contains the final implementation and evaluation artifacts for the Master's thesis "Human-Centered Information Retrieval for Swiss Open Government Data Using Fuzzy Logic-Based Ranking". The work implements, evaluates, and documents a retrieval framework that integrates lexical retrieval (BM25-style), metadata-aware scoring, and a Mamdani fuzzy inference reranker to produce explainable dataset rankings for the Swiss Open Government Data (OGD) portal.

Key outcomes:
- The codebase implements a complete production retrieval path alongside a Streamlit-based prototype UI.  
- The ranking pipeline combines lexical similarity, metadata-derived features (recency, completeness, resource availability) and a calibrated Mamdani inference engine.  
- The evaluation uses a frozen corpus, a 15-query benchmark, and an authoritative final ground-truth to generate reproducible results.  
- The implementation produces human-readable explanations that map fuzzy memberships and top contributing rules to plain-language statements.

This repository reflects the completed thesis deliverables and evaluation artifacts; it is written in the past tense to describe completed work and results.

---

## Quick Reproduction

From a fresh clone, run the following sequence from the repository root to reproduce the published evaluation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python evaluation/run_reproducible_experiment.py
```

This uses the frozen metadata snapshot, the frozen benchmark queries, and the final ground truth, and it regenerates the evaluation artifacts reported in the thesis. No live CKAN access is required when running the reproducible entry point.

---

## Research Questions

- RQ1: How can fuzzy logic be used to effectively retrieve Open Government Datasets within OGD portals?
- RQ2: How can multiple metadata-based relevance criteria be integrated into a unified and interpretable ranking mechanism?
- RQ3: To what extent does a fuzzy logic-based retrieval system support Human-Centered Information Retrieval principles?
- RQ4: How effective are metadata-driven explanations in improving user understanding and trust in ranking decisions?

---

## Architecture

The implementation is organized into a production retrieval pipeline (used for evaluation and reproducible benchmarking) and optional prototype components (UI, LLM-based normalization, visual explanations).

Production retrieval pipeline (core):
- Query parsing and normalization (`code/query_processing`)  
- Candidate retrieval against the frozen corpus (implemented in `evaluation/experiment_runner.py`)  
- Lexical similarity calculator / BM25-style scoring (implemented in `code/ranking/fuzzy_ranker.py`)  
- Metadata scorers (recency, completeness, resource availability) (implemented in `code/ranking/fuzzy_ranker.py`)  
- Mamdani fuzzy inference engine (fuzzification → rule evaluation → aggregation → defuzzification) (`code/fuzzy_system/inference_engine.py`)  
- Explanation generator (`code/ranking/explanation_generator.py`)  

Optional / prototype components:
- Streamlit prototype UI with visual explanations (`code/prototype`)  
- LLM-based query normalizer (`code/query_processing/llm_normalizer.py`) — optional and disabled by default  
- Semantic baseline adapters that depend on external embedding models (cached in `evaluation/embeddings_cache`) — optional for reproduction when model files are available

---

## Repository Structure (generated from this workspace)

The following tree lists the primary files and directories present at the time of writing. Paths are workspace-relative and reflect the actual repository contents.

- code/
  - config.py
  - main.py
  - data_collection/
    - ckan_api_client.py
    - comprehensive_collector.py
    - metadata_collector.py
  - fuzzy_system/
    - calibrated_variables.py
    - inference_engine.py
    - linguistic_variables.py
    - membership_functions.py
    - fuzzy_rules.py
    - production_engine.py
    - __init__.py
  - query_processing/
    - query_parser.py
    - llm_normalizer.py
    - __init__.py
  - ranking/
    - fuzzy_ranker.py
    - explanation_generator.py
    - ai_semantic_baseline.py
    - baseline_keyword.py
    - __init__.py
  - visualization/
    - membership_plots.py
  - prototype/
    - app.py
    - swiss_ogd_portal.py
    - portal_analysis_app.py
    - analytics_dashboard.py
    - feedback_logger.py
  - tests/
    - test_fuzzy_engine.py
    - test_query_and_ranking.py
    - test_integration.py

- evaluation/
  - experiment_runner.py
  - run_reproducible_experiment.py
  - evaluation_framework.py
  - build_ground_truth.py
  - benchmark_queries_v2.json
  - benchmark_queries.json
  - ground_truth_final.json
  - ground_truth_manual.json
  - data/
    - pooled_candidates.csv
  - scripts/
    - export_user_study_audit.py
    - normalize_pooled_grades.py
  - results/
    - final_evaluation_report.md
    - system_summary.csv
    - query_metrics.csv
    - bootstrap_confidence_intervals.csv
    - pairwise_statistics.csv
    - reproducibility_report.md
    - user_study_audit.md
    - mob04_trace.json

- data/
  - raw/
    - ogd_metadata_20260306_183841.json
    - ogd_metadata_20260306_183841.csv
    - ogd_representative_sample.json

- analytics/
  - fuzzy_calibration_live.json
  - dynamic_calibration.py
  - statistical_analysis.py
  - statistical_analysis_report.json

- requirements.txt
- tools/
  - trace_single_query.py

- images/
  - (evaluation and visualization figures used in the thesis)

This README intentionally omits obsolete files and duplicate prototype paths. The list above maps to the actual files present in this workspace.

---

## Installation

Supported Python: 3.11 (the code was developed and tested on Python 3.11). Create and activate a virtual environment before installing dependencies.

Recommended steps (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:
- requirements.txt pins key packages used for evaluation (e.g., numpy, scipy, pandas, sentence-transformers for the mandatory semantic baseline).
- The semantic baseline depends on external embedding models and torch. Embedding files used for the evaluation are cached under `evaluation/embeddings_cache/`; the reproducible runner uses this local cache and requires `sentence-transformers` to be installed.

---

## Running

Run the Streamlit prototype (optional UI/prototype):

```powershell
# From repository root (Windows)
.\.venv\Scripts\Activate.ps1
streamlit run code/prototype/app.py
```

Run the reproducible evaluation (produces `evaluation/results` artifacts):

```powershell
.\.venv\Scripts\Activate.ps1
python evaluation/run_reproducible_experiment.py
```

Run unit and evaluation tests:

```powershell
.\.venv\Scripts\Activate.ps1
pytest -q --import-mode=importlib
# or run focused tests
pytest evaluation/tests -q --import-mode=importlib
```

Run the single-query trace (example used to capture inference internals for MOB-04):

```powershell
.\.venv\Scripts\Activate.ps1
python tools/trace_single_query.py
# This writes JSON to evaluation/results/mob04_trace.json
```

---

## Evaluation Summary

The evaluation was executed on a frozen snapshot of opendata.swiss metadata (snapshot date: 2026-03-06). The evaluation suite includes:

- 15 benchmark queries (`evaluation/benchmark_queries_v2.json`).
- Five evaluated retrieval systems (see "Systems Evaluated" below).
- Final ground-truth judgments (`evaluation/ground_truth_final.json`).
- A formative user study (10 Master's students) documented in `evaluation/results/user_study_audit.md`.

Metrics and statistical analysis produced in `evaluation/results` include:

- MAP (Mean Average Precision)
- P@5 (Precision at 5)
- nDCG@10 (nDCG at 10)
- MRR (Mean Reciprocal Rank)
- Paired Wilcoxon signed-rank tests with Holm-Bonferroni correction for multiple comparisons
- Bootstrap confidence intervals for mean differences
- Rule-weight and membership-function sensitivity analyses (results in `evaluation/results/sensitivity_*.csv`)

Key claim in the thesis: the fuzzy HCIR framework achieved competitive retrieval performance across standard IR metrics while providing transparent, interpretable explanations that support human-centered decision-making; results and statistics are available in `evaluation/results` and the final evaluation report `evaluation/results/final_evaluation_report.md`.

---

## Systems Evaluated (final implementation)

The evaluation compares the following systems (implemented in this repository):

- portal_default — Frozen-corpus proxy of the opendata.swiss portal ranking (baseline)
- keyword_bm25 — BM25-style lexical baseline implemented in `code/ranking/baseline_keyword.py`
- Weighted Sum — Simple weighted-sum metadata baseline implemented in `evaluation/experiment_runner.py`
- Linear Weighted — Linear weighted baseline implemented in `evaluation/experiment_runner.py`
- Fuzzy HCIR — Proposed Mamdani fuzzy inference reranker implemented in `code/fuzzy_system/` and `code/ranking/fuzzy_ranker.py`
- Semantic — Real semantic baseline using sentence-transformers with cached embeddings available in `evaluation/embeddings_cache`; implementation in `code/ranking/ai_semantic_baseline.py`.
- The semantic baseline uses the `paraphrase-multilingual-MiniLM-L12-v2` sentence-transformers model with seed `42`; the repository does not pin a specific Hugging Face commit revision in code.

All systems are evaluated against the same frozen corpus to ensure a reproducible comparison.

---

## Reproducibility

This repository includes the exact artifacts and runner used for the thesis benchmark. The evaluation is deterministic and local: it does not require live CKAN API access when you run the reproducible entry point.

### Key assets used by the benchmark

- Frozen corpus: `data/raw/ogd_metadata_20260306_183841.json`
- Final ground truth: `evaluation/ground_truth_final.json`
- Benchmark queries: `evaluation/benchmark_queries_v2.json`
- Reproducible evaluation helper: `evaluation/run_reproducible_experiment.py`
- Core evaluation engine: `evaluation/experiment_runner.py`
- Evaluation output directory: `evaluation/results/`

### What is verified

- The benchmark uses the local frozen corpus snapshot, not live CKAN package search.
- `evaluation/run_reproducible_experiment.py` checks the corpus and ground truth files before executing the evaluation.
- All retrievers in the benchmark read from the frozen corpus implementation and do not make live CKAN API calls.
- The final ground truth file is the authoritative label source for the reported metrics.
- Determinism is supported by fixed seeds in the bootstrap statistics and semantic baseline provider, and by the fixed evaluation reference date in `evaluation/experiment_runner.py`.
- The semantic baseline is included in the benchmark and uses cached embeddings from `evaluation/embeddings_cache`; requirements.txt installs sentence-transformers and torch so the full benchmark can run as-is.

### Reproducibility command (fresh clone)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python evaluation/run_reproducible_experiment.py
```

### Expected outputs

The evaluation writes its results into `evaluation/results/`.
Expected files include:

- `evaluation/results/query_metrics.csv`
- `evaluation/results/system_summary.csv`
- `evaluation/results/pairwise_statistics.csv`
- `evaluation/results/bootstrap_confidence_intervals.csv`
- `evaluation/results/win_loss_matrix.csv`
- `evaluation/results/experiment_results.json`
- `evaluation/results/reproducibility_report.json`
- `evaluation/results/reproducibility_report.md`
- `evaluation/results/publication_tables.md`
- `evaluation/results/figures/` (generated plot PNGs)

If the semantic baseline is included, the run also makes use of sentence-transformers and the cached embedding artifacts under `evaluation/embeddings_cache/`.

---

## User Study and Explainability

A formative user study with 10 Master's students evaluated the interpretability and usefulness of metadata-driven explanations. The study materials and audit are available at `evaluation/results/user_study_audit.md` and the aggregated feedback is summarized in `evaluation/results/brief_results_summary.md`.

The explanation generator maps:
- Dominant fuzzy terms (e.g., exact_match, very_recent, complete) to short natural-language statements.
- Top contributing rules from the Mamdani engine into human-readable reasons.  
These explanations were judged by participants to improve transparency and helped users understand why datasets are ranked in a particular order.

---

## Removing Obsolete Materials

This README reflects the final thesis deliverables. It intentionally omits project timeline, proposal milestones, placeholder URLs, and planning artifacts. The repository retains some prototype and migration backups for traceability, but the active entry points and evaluation scripts listed above are the authoritative references for reproduction and evidence.

---

## Contact

For questions about the implementation or reproducibility artifacts, contact Deep Shukla via university channels.

---

*This README describes the completed Master's thesis artifacts and evaluation as present in the repository.*