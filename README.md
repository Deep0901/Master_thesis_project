# Swiss OGD Fuzzy Retrieval System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Master Thesis Project: "Improving Access to Swiss Open Government Data through Fuzzy Human-Centered Information Retrieval"

**Author:** Deep Shukla  
**Institution:** University of Fribourg, Human-IST Institute  
**Examiner:** Prof. Dr. Edy Portmann  
**Supervisor:** Janick Spycher

---

## 📖 Overview

This repository contains the implementation of a fuzzy logic-based retrieval system for Swiss Open Government Data (OGD). The system aims to improve the discoverability and accessibility of datasets on opendata.swiss by:

1. **Handling vague queries** through fuzzy linguistic variables
2. **Incorporating metadata quality** in ranking decisions
3. **Providing transparent explanations** for ranking results
4. **Supporting multilingual queries** (German, French, Italian, English)

## 🎯 Research Questions

| RQ | Question |
|----|----------|
| RQ1 | How can fuzzy logic model vagueness in OGD metadata for query-specific ranking? |
| RQ2 | How does fuzzy ranking compare to keyword-based retrieval? |
| RQ3 | How does explainability influence user trust in rankings? |
| RQ4 | How does fuzzy ranking compare to AI-driven semantic approaches? |

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Query Parser   │────▶│  Fuzzy Ranker   │────▶│  Explanation    │
│  (Multilingual) │     │  (Mamdani)      │     │  Generator      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         ▼                      ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  LLM Normalizer │     │   Rule Base     │     │  Streamlit UI   │
│  (Optional)     │     │   (20 Rules)    │     │  (Prototype)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 📂 Project Structure

```
thesis/
├── code/                     # Main application code
│   ├── config.py
│   ├── main.py
│   ├── fuzzy_system/         # Fuzzy inference engine
│   │   ├── calibrated_variables.py
│   │   ├── linguistic_variables.py
│   │   ├── membership_functions.py
│   │   ├── fuzzy_rules.py
│   │   ├── inference_engine.py
│   │   └── production_engine.py
│   ├── query_processing/     # Query parsing & normalization
│   │   ├── query_parser.py
│   │   └── llm_normalizer.py
│   ├── ranking/              # Ranking implementations
│   │   ├── fuzzy_ranker.py
│   │   ├── baseline_keyword.py
│   │   ├── ai_semantic_baseline.py
│   │   └── explanation_generator.py
│   ├── data_collection/      # CKAN API integration
│   │   ├── ckan_api_client.py
│   │   ├── comprehensive_collector.py
│   │   └── metadata_collector.py
│   ├── prototype/            # Web application
│   │   ├── app.py
│   │   ├── swiss_ogd_portal.py
│   │   ├── portal_analysis_app.py
│   │   └── analytics_dashboard.py
│   ├── visualization/        # Data visualization utilities
│   │   └── membership_plots.py
│   └── tests/                # Unit and integration tests
│       ├── test_fuzzy_engine.py
│       └── test_integration.py
├── analytics/                # Statistical analysis
│   ├── statistical_analysis.py
│   └── statistical_analysis_report.json
├── data/                     # Datasets and cache
│   ├── raw/                  # Original datasets
│   │   ├── ogd_metadata_20260306_183841.csv
│   │   ├── ogd_metadata_20260306_183841.json
│   │   └── ogd_representative_sample.json
│   └── processed/            # Processed datasets
├── evaluation/               # Evaluation materials and results
│   ├── annotation_tool.py
│   ├── evaluation_framework.py
│   ├── experiment_runner.py
│   ├── benchmark_queries.json
│   ├── benchmark_queries_v2.json
│   ├── experiment_results.json
│   ├── ground_truth_auto.json
│   ├── ground_truth/         # Ground truth data
│   ├── raters/               # Rater data
│   ├── results/              # Evaluation results
│   └── test_queries/         # Test queries
│       └── benchmark_queries.json
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/username/swiss-ogd-fuzzy.git
cd swiss-ogd-fuzzy

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Prototype

```bash
streamlit run code/prototype/app.py
```

Then open http://localhost:8501 in your browser.

### Running Tests

```bash
pytest tests/ -v
```

## 🔧 Configuration

Create a `.env` file for optional features:

```env
# OpenAI API (for LLM query normalization)
OPENAI_API_KEY=your-api-key

# Debug mode
DEBUG_MODE=false

# Enable LLM normalization
ENABLE_LLM=false
```

## 📊 Fuzzy System

### Input Variables

| Variable | Range | Terms |
|----------|-------|-------|
| Recency | 0-730 days | very_recent, recent, moderate, old, very_old |
| Completeness | 0-1 | complete, mostly_complete, partial, sparse, empty |
| Thematic Similarity | 0-1 | exact_match, highly_relevant, relevant, somewhat_relevant, not_relevant |
| Resource Availability | 0-20 | comprehensive, good, limited, minimal |

### Output Variable

| Variable | Range | Terms |
|----------|-------|-------|
| Relevance Score | 0-100 | excellent, good, moderate, low, very_low |

### Sample Rules

```
IF similarity IS exact_match AND recency IS very_recent 
   AND completeness IS complete
THEN relevance IS excellent

IF similarity IS highly_relevant AND recency IS recent
THEN relevance IS good
```

## 📈 Evaluation

The system is evaluated through:

1. **Benchmark Queries**: 15 queries across Environment & Mobility domains
2. **Baseline Comparisons**: TF-IDF keyword search, AI semantic search
3. **User Study**: 15-20 participants, within-subjects design
4. **Metrics**: Precision@K, NDCG, SUS, Trust scale

## 🗓️ Timeline

| Phase | Period | Deliverables |
|-------|--------|--------------|
| 1 | Mar-Apr 2026 | Literature review, conceptual design |
| 2 | May-Jun 2026 | Prototype implementation |
| 3 | Jul 2026 | Evaluation & user study |
| 4 | Aug 2026 | Thesis writing |
| 5 | Sep 2026 | Final submission |

## 📚 Key References

- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*
- Portmann, E. (2013). Fuzzy methods for human-centric information systems
- Hevner, A. R. (2004). Design science in information systems research
- Peffers, K. (2007). A design science research methodology

## 📝 License

This project is part of a Master's thesis at the University of Fribourg.

## 🤝 Acknowledgments

- Prof. Dr. Edy Portmann (Examiner)
- Janick Spycher (Supervisor)
- Human-IST Institute, University of Fribourg
- Swiss Federal Archives (opendata.swiss)

---

*For questions or feedback, please contact the author through the University of Fribourg.*
