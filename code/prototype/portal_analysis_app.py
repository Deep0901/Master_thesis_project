"""
Portal Analysis Prototype

A thesis-aligned Streamlit prototype for the chapter on portal analysis and
problem identification. The app focuses on the search limitations identified
in the document and contrasts them with a fuzzy, explainable ranking view.
"""

from __future__ import annotations

import html
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import streamlit as st

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

try:
    from query_processing.query_parser import QueryParser
except Exception:  # pragma: no cover - fallback for partial environments
    QueryParser = None

try:
    from fuzzy_system import create_inference_engine
except Exception:  # pragma: no cover - fallback for partial environments
    create_inference_engine = None

try:
    from ranking.explanation_generator import create_explanation_generator
except Exception:  # pragma: no cover - fallback for partial environments
    create_explanation_generator = None

st.set_page_config(
    page_title="Swiss OGD Portal Analysis Prototype",
    page_icon="CH",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --swiss-red: #D6232D;
    --swiss-red-dark: #A91C23;
    --ink: #1F2937;
    --muted: #6B7280;
    --panel: #FFFFFF;
    --panel-alt: #F8FAFC;
    --border: #E5E7EB;
    --success: #0F766E;
    --warning: #B45309;
    --danger: #B91C1C;
    --info: #1D4ED8;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(214, 35, 45, 0.09), transparent 28%),
        linear-gradient(180deg, #FEFEFE 0%, #F4F6F8 100%);
    color: var(--ink);
}

.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1250px;
}

.hero {
    background: linear-gradient(135deg, var(--swiss-red) 0%, var(--swiss-red-dark) 100%);
    color: white;
    border-radius: 22px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 20px 45px rgba(155, 28, 34, 0.18);
    margin-bottom: 1rem;
}

.hero h1 {
    color: white !important;
    margin: 0 0 0.35rem 0;
    font-size: 2rem;
    line-height: 1.15;
    letter-spacing: -0.02em;
}

.hero p {
    margin: 0.2rem 0 0 0;
    color: rgba(255, 255, 255, 0.92);
    font-size: 0.98rem;
}

.subtle-panel {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(229, 231, 235, 0.9);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
}

.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem 1rem 0.9rem 1rem;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
    margin-bottom: 0.85rem;
}

.card-compact {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.85rem;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
}

.card-header {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
}

.result-title {
    font-size: 1.02rem;
    font-weight: 700;
    line-height: 1.3;
    margin: 0;
    color: var(--ink);
}

.result-meta {
    color: var(--muted);
    font-size: 0.88rem;
    margin-top: 0.25rem;
}

.score-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.45rem 0.7rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.82rem;
    white-space: nowrap;
}

.score-excellent { background: rgba(15, 118, 110, 0.12); color: var(--success); }
.score-good { background: rgba(29, 78, 216, 0.12); color: var(--info); }
.score-moderate { background: rgba(180, 83, 9, 0.12); color: var(--warning); }
.score-low { background: rgba(185, 28, 28, 0.12); color: var(--danger); }

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.45rem;
}

.badge {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    background: var(--panel-alt);
    color: var(--ink);
    border: 1px solid var(--border);
    padding: 0.2rem 0.55rem;
    font-size: 0.78rem;
    font-weight: 600;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}

.chip {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.18rem 0.55rem;
    font-size: 0.78rem;
    font-weight: 600;
    background: #F1F5F9;
    color: #334155;
    border: 1px solid #E2E8F0;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 800;
    margin: 0.3rem 0 0.7rem 0;
    color: var(--ink);
    letter-spacing: -0.01em;
}

.section-note {
    color: var(--muted);
    font-size: 0.92rem;
    margin-top: -0.15rem;
    margin-bottom: 1rem;
}

.limit-card {
    height: 100%;
    background: linear-gradient(180deg, white 0%, #FCFCFD 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 0.95rem;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
}

.limit-card h4 {
    margin: 0 0 0.4rem 0;
    font-size: 1rem;
}

.limit-tag {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.18rem 0.5rem;
    background: rgba(214, 35, 45, 0.08);
    color: var(--swiss-red-dark);
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}

.small-muted {
    color: var(--muted);
    font-size: 0.87rem;
}

.callout {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-radius: 16px;
    padding: 0.9rem 1rem;
    color: #7C2D12;
}

.profile-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 0.9rem 1rem;
}

hr {
    border-color: rgba(229, 231, 235, 0.9) !important;
}
</style>
"""

PORTAL_STATS = {
    "Total datasets": "~14,254",
    "Publishing organizations": "157",
    "Thematic categories": "14",
    "Supported languages": "DE, FR, IT, EN",
}

THEME_DISTRIBUTION = [
    {"Category": "Territory and Environment", "Datasets": 3847, "Share": 27.0},
    {"Category": "Economy", "Datasets": 2134, "Share": 15.0},
    {"Category": "Population", "Datasets": 1854, "Share": 13.0},
    {"Category": "Mobility and Transport", "Datasets": 1567, "Share": 11.0},
    {"Category": "Administration", "Datasets": 1283, "Share": 9.0},
    {"Category": "Health", "Datasets": 892, "Share": 6.3},
    {"Category": "Education", "Datasets": 743, "Share": 5.2},
    {"Category": "Agriculture", "Datasets": 621, "Share": 4.4},
    {"Category": "Construction", "Datasets": 456, "Share": 3.2},
    {"Category": "Politics", "Datasets": 398, "Share": 2.8},
]

MULTILINGUAL_QUALITY = [
    {"Language": "German", "Avg. description length": 847, "Complete translations (%)": 78},
    {"Language": "French", "Avg. description length": 623, "Complete translations (%)": 65},
    {"Language": "Italian", "Avg. description length": 412, "Complete translations (%)": 41},
    {"Language": "English", "Avg. description length": 534, "Complete translations (%)": 52},
]

LIMITATIONS = [
    {
        "id": "L1",
        "title": "Lexical keyword matching",
        "problem": "The portal relies on string matching without semantic understanding.",
        "impact": "Users must guess the exact metadata wording used by publishers.",
        "example": "'car accidents' and 'traffic collisions' behave like unrelated searches.",
    },
    {
        "id": "L2",
        "title": "Vocabulary mismatch",
        "problem": "User vocabulary often differs from administrative or technical metadata terms.",
        "impact": "Relevant datasets remain hidden behind jargon or local naming conventions.",
        "example": "'air pollution measurements' may miss a dataset titled 'NABEL'.",
    },
    {
        "id": "L3",
        "title": "Multilingual inconsistency",
        "problem": "Metadata completeness varies across German, French, Italian, and English.",
        "impact": "Non-German users are disadvantaged in discovery and comparison.",
        "example": "French and Italian records frequently have shorter or incomplete descriptions.",
    },
    {
        "id": "L4",
        "title": "Metadata quality variation",
        "problem": "Documentation completeness differs sharply between publishers.",
        "impact": "Well-described datasets and sparse records are ranked without quality context.",
        "example": "Description length and resource counts vary widely between organizations.",
    },
    {
        "id": "L5",
        "title": "No interpretation of vague queries",
        "problem": "Terms like recent, comprehensive, detailed, and reliable are treated literally.",
        "impact": "The system cannot capture user intent or quality preferences.",
        "example": "'recent environmental data' becomes a keyword search for recent + environmental + data.",
    },
    {
        "id": "L6",
        "title": "Binary relevance model",
        "problem": "Results are shown as matching or not matching, without graded relevance.",
        "impact": "Users cannot inspect why a ranking order was chosen.",
        "example": "The first result is not necessarily the best fit for the search intent.",
    },
]

QUERY_EXAMPLES = [
    ("Keyword", "air quality Zurich"),
    ("Vague intent", "recent environmental data with complete metadata"),
    ("Multilingual", "données climatiques récentes"),
    ("Vocabulary mismatch", "air pollution measurements"),
]

DEMO_DATASETS: List[Dict[str, Any]] = [
    {
        "id": "air-quality-zurich-2024",
        "title": {
            "de": "Luftqualitätsmessungen Zürich 2024",
            "fr": "Mesures de qualité de l'air Zurich 2024",
            "it": "Misure della qualità dell'aria Zurigo 2024",
            "en": "Air Quality Measurements Zurich 2024",
        },
        "description": {
            "de": "Tägliche Messungen von PM2.5, NO2 und O3 in der Region Zürich.",
            "en": "Daily air quality measurements including PM2.5, NO2 and O3 in Zurich.",
        },
        "tags": ["air quality", "pollution", "environment", "zurich"],
        "groups": ["environment"],
        "resources": [{"format": "CSV"}, {"format": "JSON"}, {"format": "API"}],
        "organization": {"name": "City of Zurich"},
        "license_id": "cc-by-4.0",
        "days_since_modified": 3,
    },
    {
        "id": "nabel-air-pollution",
        "title": {
            "de": "NABEL - Nationales Beobachtungsnetz für Luftfremdstoffe",
            "en": "NABEL - National Air Pollution Monitoring Network",
        },
        "description": {
            "de": "Langzeitmessungen zu Luftschadstoffen an Referenzstandorten in der Schweiz.",
        },
        "tags": ["air pollution", "nabel", "monitoring", "environment"],
        "groups": ["environment"],
        "resources": [{"format": "CSV"}],
        "organization": {"name": "BAFU"},
        "license_id": "cc-by-4.0",
        "days_since_modified": 112,
    },
    {
        "id": "traffic-volume-swiss-2023",
        "title": {
            "de": "Verkehrsaufkommen Statistik 2023",
            "fr": "Statistiques du trafic 2023",
            "en": "Traffic Volume Statistics 2023",
        },
        "description": {
            "de": "Jährliche Verkehrsdaten für Nationalstrassen und Hauptachsen.",
            "en": "Annual traffic volume data for Swiss national highways and main roads.",
        },
        "tags": ["traffic", "transport", "mobility", "highways"],
        "groups": ["mobility"],
        "resources": [{"format": "CSV"}, {"format": "PDF"}],
        "organization": {"name": "ASTRA"},
        "license_id": "cc-by-4.0",
        "days_since_modified": 45,
    },
    {
        "id": "public-transport-punctuality",
        "title": {
            "de": "Pünktlichkeit öffentlicher Verkehr",
            "fr": "Ponctualité des transports publics",
            "en": "Public Transport Punctuality",
        },
        "description": {
            "de": "Statistiken zur Pünktlichkeit von Bahn und Bus in der Schweiz.",
            "fr": "Statistiques sur la ponctualité des trains et des bus en Suisse.",
            "en": "Statistics on train and bus punctuality across Switzerland.",
        },
        "tags": ["public transport", "trains", "mobility", "punctuality"],
        "groups": ["mobility"],
        "resources": [{"format": "CSV"}],
        "organization": {"name": "SBB"},
        "license_id": "cc-by-4.0",
        "days_since_modified": 180,
    },
    {
        "id": "water-quality-rivers-2024",
        "title": {
            "de": "Wasserqualität Flüsse 2024",
            "en": "Water Quality Rivers 2024",
        },
        "description": {
            "de": "Messwerte zur Wasserqualität an wichtigen Schweizer Flüssen.",
            "en": "Water quality measurements for major Swiss rivers.",
        },
        "tags": ["water quality", "rivers", "environment", "pollution"],
        "groups": ["environment"],
        "resources": [{"format": "CSV"}, {"format": "GeoJSON"}],
        "organization": {"name": "BAFU"},
        "license_id": "cc-by-4.0",
        "days_since_modified": 10,
    },
    {
        "id": "population-statistics-swiss",
        "title": {
            "de": "Bevölkerungsstatistik Schweiz",
            "fr": "Statistiques de la population Suisse",
            "it": "Statistiche della popolazione Svizzera",
            "en": "Population Statistics Switzerland",
        },
        "description": {
            "de": "Demografische Kennzahlen zu Altersstruktur, Haushalten und Migration.",
            "fr": "Indicateurs démographiques sur l'âge, les ménages et la migration.",
            "it": "Indicatori demografici su età, famiglie e migrazione.",
            "en": "Demographic indicators on age structure, households and migration.",
        },
        "tags": ["population", "demographics", "statistics"],
        "groups": ["population"],
        "resources": [{"format": "CSV"}, {"format": "XLSX"}],
        "organization": {"name": "FSO"},
        "license_id": "cc-by-4.0",
        "days_since_modified": 28,
    },
]

THEME_KEYWORDS = {
    "environment": [
        "environment",
        "environmental",
        "air quality",
        "air pollution",
        "luftqualitat",
        "wasserqualitat",
        "water quality",
        "pollution",
        "climate",
        "climatic",
        "noise",
        "larm",
    ],
    "mobility": [
        "transport",
        "traffic",
        "mobility",
        "verkehr",
        "public transport",
        "rail",
        "road",
        "trains",
        "bus",
    ],
    "population": [
        "population",
        "demographic",
        "demographics",
        "bevolkerung",
        "statistics",
        "migration",
    ],
    "health": ["health", "gesundheit", "sante", "salute", "hospital"],
    "economy": ["economy", "economy", "wirtschaft", "finance", "employment"],
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "with",
    "for",
    "of",
    "in",
    "on",
    "to",
    "de",
    "des",
    "du",
    "le",
    "la",
    "les",
    "et",
    "und",
    "oder",
    "mit",
    "für",
    "fur",
    "di",
    "da",
    "con",
    "per",
    "recent",
    "recently",
    "new",
    "old",
    "complete",
    "comprehensive",
    "detailed",
    "reliable",
    "data",
    "dataset",
    "datasets",
}


@dataclass
class RankedResult:
    dataset: Dict[str, Any]
    score: float
    rank: int
    explanation: str
    input_scores: Dict[str, float]


@st.cache_data(show_spinner=False)
def load_demo_datasets() -> List[Dict[str, Any]]:
    return DEMO_DATASETS


@st.cache_resource(show_spinner=False)
def get_components(defuzzification_method: str) -> Tuple[Any, Any, Any]:
    parser = QueryParser() if QueryParser is not None else None
    engine = create_inference_engine(defuzzification_method) if create_inference_engine is not None else None
    explainer = create_explanation_generator() if create_explanation_generator is not None else None
    return parser, engine, explainer


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = text.replace("\u2019", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_query_terms(query: str) -> List[str]:
    terms = [term for term in normalize_text(query).split() if len(term) > 2 and term not in STOPWORDS]
    return terms


def extract_text(dataset: Dict[str, Any]) -> str:
    parts: List[str] = []
    title = dataset.get("title", "")
    description = dataset.get("description", "")

    if isinstance(title, dict):
        parts.extend(str(value) for value in title.values())
    else:
        parts.append(str(title))

    if isinstance(description, dict):
        parts.extend(str(value) for value in description.values())
    else:
        parts.append(str(description))

    parts.extend(dataset.get("tags", []))
    parts.extend(dataset.get("groups", []))

    organization = dataset.get("organization", {}) or {}
    if isinstance(organization, dict):
        parts.extend(str(value) for value in organization.values())
    else:
        parts.append(str(organization))

    for resource in dataset.get("resources", []):
        if isinstance(resource, dict):
            parts.extend(str(value) for value in resource.values())
        else:
            parts.append(str(resource))

    return " ".join(parts)


def dataset_title(dataset: Dict[str, Any], preferred_language: Optional[str] = None) -> str:
    title = dataset.get("title", "")
    if isinstance(title, dict):
        if preferred_language and title.get(preferred_language):
            return str(title[preferred_language])
        for language in ("en", "de", "fr", "it"):
            if title.get(language):
                return str(title[language])
        return str(next(iter(title.values())))
    return str(title)


def dataset_description(dataset: Dict[str, Any], preferred_language: Optional[str] = None) -> str:
    description = dataset.get("description", "")
    if isinstance(description, dict):
        if preferred_language and description.get(preferred_language):
            return str(description[preferred_language])
        for language in ("en", "de", "fr", "it"):
            if description.get(language):
                return str(description[language])
        return str(next(iter(description.values())))
    return str(description)


def score_label(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Moderate"
    return "Low"


def score_class(score: float) -> str:
    if score >= 80:
        return "score-excellent"
    if score >= 60:
        return "score-good"
    if score >= 40:
        return "score-moderate"
    return "score-low"


def count_translations(field: Any) -> int:
    if isinstance(field, dict):
        return sum(1 for value in field.values() if str(value).strip())
    return 1 if str(field).strip() else 0


def metadata_completeness(dataset: Dict[str, Any]) -> float:
    checks = [
        bool(dataset.get("title")),
        bool(dataset.get("description")),
        len(dataset.get("tags", [])) > 0,
        len(dataset.get("resources", [])) > 0,
        bool(dataset.get("organization")),
        bool(dataset.get("license_id")),
        count_translations(dataset.get("title")) >= 2,
        count_translations(dataset.get("description")) >= 2,
    ]
    return sum(checks) / len(checks)


def recency_days(dataset: Dict[str, Any]) -> int:
    value = dataset.get("days_since_modified")
    if value is None:
        return 365
    return int(max(0, min(int(value), 730)))


def resource_count(dataset: Dict[str, Any]) -> int:
    resources = dataset.get("resources", [])
    return int(min(len(resources), 20))


def infer_themes(text: str) -> Set[str]:
    normalized = normalize_text(text)
    detected: Set[str] = set()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            detected.add(theme)
    return detected


def thematic_similarity(query: str, dataset: Dict[str, Any]) -> float:
    query_terms = extract_query_terms(query)
    corpus = normalize_text(extract_text(dataset))
    if not query_terms:
        return 0.0

    query_hits = sum(1 for term in query_terms if term in corpus)
    token_score = query_hits / len(query_terms)

    query_themes = infer_themes(query)
    dataset_themes = infer_themes(extract_text(dataset))
    if query_themes and dataset_themes:
        overlap = len(query_themes & dataset_themes) / len(query_themes | dataset_themes)
    else:
        overlap = 0.0

    title_text = normalize_text(dataset_title(dataset))
    title_bonus = 0.12 if any(term in title_text for term in query_terms) else 0.0

    return min(1.0, (token_score * 0.6) + (overlap * 0.3) + title_bonus)


def portal_score(query: str, dataset: Dict[str, Any]) -> float:
    query_terms = extract_query_terms(query)
    corpus = normalize_text(extract_text(dataset))
    if not query_terms:
        return 0.0

    exact_hits = sum(1 for term in query_terms if term in corpus)
    query_exact = normalize_text(query)
    phrase_bonus = 1.0 if query_exact and query_exact in corpus else 0.0
    title_text = normalize_text(dataset_title(dataset))
    title_bonus = 1.0 if any(term in title_text for term in query_terms) else 0.0
    recency_bonus = 1.0 - (recency_days(dataset) / 730.0)

    score = (
        (exact_hits / len(query_terms)) * 55.0
        + phrase_bonus * 18.0
        + title_bonus * 12.0
        + max(0.0, recency_bonus) * 15.0
    )
    return round(min(100.0, score), 2)


def fuzzy_inputs(query: str, dataset: Dict[str, Any]) -> Dict[str, float]:
    return {
        "recency": float(recency_days(dataset)),
        "completeness": float(metadata_completeness(dataset)),
        "thematic_similarity": float(thematic_similarity(query, dataset)),
        "resource_availability": float(resource_count(dataset)),
    }


def fuzzy_score(query: str, dataset: Dict[str, Any], engine: Any, explainer: Any) -> Tuple[float, str, Dict[str, float]]:
    inputs = fuzzy_inputs(query, dataset)
    title = dataset_title(dataset)

    if engine is not None:
        result = engine.infer(inputs)
        score = float(result.crisp_output)
        explanation = result.get_explanation(top_n=3)
        if explainer is not None:
            explanation_model = explainer.generate_explanation(
                dataset_title=title,
                relevance_score=score,
                input_scores=inputs,
            )
            explanation = explanation_model.full_explanation
        return score, explanation, inputs

    score = (
        inputs["thematic_similarity"] * 52.0
        + (1.0 - min(inputs["recency"], 730.0) / 730.0) * 22.0
        + inputs["completeness"] * 16.0
        + min(inputs["resource_availability"] / 8.0, 1.0) * 10.0
    )
    explanation = (
        f"Fallback fuzzy score derived from thematic similarity ({inputs['thematic_similarity']:.0%}), "
        f"recency ({inputs['recency']:.0f} days), completeness ({inputs['completeness']:.0%}), "
        f"and resource availability ({inputs['resource_availability']:.0f})."
    )
    return round(min(100.0, score), 2), explanation, inputs


def build_query_profile(query: str, parser: Any) -> Dict[str, Any]:
    if parser is None:
        themes = sorted(infer_themes(query))
        return {
            "language": "unknown",
            "keywords": extract_query_terms(query),
            "temporal": "any",
            "quality": "any",
            "themes": themes,
        }

    parsed = parser.parse(query)
    return {
        "language": getattr(parsed.detected_language, "value", "unknown"),
        "keywords": parsed.keywords,
        "temporal": getattr(parsed.temporal_modifier, "value", "any"),
        "quality": getattr(parsed.quality_modifier, "value", "any"),
        "themes": parsed.themes or sorted(infer_themes(query)),
    }


def rank_datasets(query: str, datasets: Sequence[Dict[str, Any]], parser: Any, engine: Any, explainer: Any) -> Tuple[List[RankedResult], List[RankedResult], Dict[str, Any]]:
    profile = build_query_profile(query, parser)
    portal_rows: List[RankedResult] = []
    fuzzy_rows: List[RankedResult] = []

    for dataset in datasets:
        portal = portal_score(query, dataset)
        fuzzy, explanation, inputs = fuzzy_score(query, dataset, engine, explainer)
        portal_rows.append(
            RankedResult(
                dataset=dataset,
                score=portal,
                rank=0,
                explanation="Literal keyword matching with a recency bias.",
                input_scores=inputs,
            )
        )
        fuzzy_rows.append(
            RankedResult(
                dataset=dataset,
                score=fuzzy,
                rank=0,
                explanation=explanation,
                input_scores=inputs,
            )
        )

    portal_rows.sort(key=lambda item: item.score, reverse=True)
    fuzzy_rows.sort(key=lambda item: item.score, reverse=True)

    for index, item in enumerate(portal_rows, start=1):
        item.rank = index
    for index, item in enumerate(fuzzy_rows, start=1):
        item.rank = index

    return portal_rows, fuzzy_rows, profile


def result_card(result: RankedResult, method_name: str) -> None:
    dataset = result.dataset
    title = html.escape(dataset_title(dataset))
    description = html.escape(dataset_description(dataset))
    organization = html.escape(str((dataset.get("organization") or {}).get("name", "Unknown")))
    tags = [html.escape(str(tag)) for tag in dataset.get("tags", [])[:4]]
    tag_html = "".join(f'<span class="chip">{tag}</span>' for tag in tags)

    resources = ", ".join(
        sorted({str(resource.get("format", "")) for resource in dataset.get("resources", []) if isinstance(resource, dict) and resource.get("format")})
    ) or "No resource formats listed"

    score_cls = score_class(result.score)
    score_text = f"{result.score:.1f}/100 · {score_label(result.score)}"

    st.markdown(
        f"""
        <div class="card">
            <div class="card-header">
                <div>
                    <div class="small-muted">{html.escape(method_name)} · Rank #{result.rank}</div>
                    <div class="result-title">{title}</div>
                    <div class="result-meta">{description}</div>
                    <div class="badge-row">
                        <span class="badge">{organization}</span>
                        <span class="badge">Updated {recency_days(dataset)} days ago</span>
                        <span class="badge">{resources}</span>
                    </div>
                    <div class="badge-row">{tag_html}</div>
                </div>
                <div class="score-pill {score_cls}">{score_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Why this ranking?", expanded=False):
        if method_name.lower().startswith("fuzzy"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Thematic similarity", f"{result.input_scores['thematic_similarity']:.0%}")
            with col2:
                st.metric("Recency", f"{result.input_scores['recency']:.0f} days")
            with col3:
                st.metric("Completeness", f"{result.input_scores['completeness']:.0%}")
            with col4:
                st.metric("Resources", f"{result.input_scores['resource_availability']:.0f}")
        else:
            st.metric("Keyword score", f"{result.score:.1f}/100")
            st.caption("This baseline reflects literal term matching and a recency bias, mirroring the limitations described in the chapter.")

        st.info(result.explanation)


def query_profile_box(profile: Dict[str, Any]) -> None:
    chips = [
        f'<span class="chip">Language: {html.escape(str(profile["language"]).upper())}</span>',
        f'<span class="chip">Temporal: {html.escape(str(profile["temporal"]))}</span>',
        f'<span class="chip">Quality: {html.escape(str(profile["quality"]))}</span>',
    ]
    if profile["themes"]:
        chips.append(f'<span class="chip">Themes: {html.escape(", ".join(profile["themes"]))}</span>')
    if profile["keywords"]:
        chips.append(f'<span class="chip">Keywords: {html.escape(", ".join(profile["keywords"]))}</span>')

    st.markdown(
        f"""
        <div class="profile-box">
            <div class="small-muted" style="margin-bottom: 0.35rem;">Query interpretation</div>
            <div class="chip-row">{''.join(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_query_input(example_query: str) -> None:
    """Update the search field from an example button callback."""
    st.session_state.query_input = example_query


def render_overview_tab() -> None:
    st.markdown('<div class="section-title">Portal analysis summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">The prototype foregrounds the evidence collected in the chapter: lexical matching, vocabulary mismatch, multilingual inconsistencies, and the lack of graded relevance.</div>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    for index, (label, value) in enumerate(PORTAL_STATS.items()):
        with metric_cols[index]:
            st.metric(label, value)

    st.markdown("---")
    st.markdown('<div class="section-title">Observed search limitations</div>', unsafe_allow_html=True)
    limitation_rows = [LIMITATIONS[:3], LIMITATIONS[3:]]
    for row in limitation_rows:
        cols = st.columns(len(row))
        for col, item in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="limit-card">
                        <div class="limit-tag">{item['id']}</div>
                        <h4>{html.escape(item['title'])}</h4>
                        <div class="small-muted"><strong>Problem:</strong> {html.escape(item['problem'])}</div>
                        <div style="height: 0.45rem;"></div>
                        <div class="small-muted"><strong>Impact:</strong> {html.escape(item['impact'])}</div>
                        <div style="height: 0.45rem;"></div>
                        <div class="small-muted"><strong>Example:</strong> {html.escape(item['example'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown('<div class="section-title">Thematic distribution</div>', unsafe_allow_html=True)
        theme_df = pd.DataFrame(THEME_DISTRIBUTION)
        st.dataframe(theme_df, width="stretch", hide_index=True)
    with c2:
        st.markdown('<div class="section-title">Multilingual metadata quality</div>', unsafe_allow_html=True)
        multilingual_df = pd.DataFrame(MULTILINGUAL_QUALITY)
        st.dataframe(multilingual_df, width="stretch", hide_index=True)

    st.caption("These tables mirror the numbers referenced in the chapter and provide the empirical baseline for the prototype.")


def render_search_tab(query: str, view_mode: str, top_n: int, parser: Any, engine: Any, explainer: Any) -> None:
    st.markdown('<div class="section-title">Search comparison demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Enter a query from the chapter examples to compare the portal-style baseline with a fuzzy, explainable ranking.</div>',
        unsafe_allow_html=True,
    )

    example_cols = st.columns(len(QUERY_EXAMPLES))
    for index, (label, example) in enumerate(QUERY_EXAMPLES):
        with example_cols[index]:
            st.button(
                label,
                width="stretch",
                key=f"example-{index}",
                on_click=set_query_input,
                args=(example,),
            )

    if not query.strip():
        st.info("Choose an example query or enter your own search phrase.")
        return

    portal_rows, fuzzy_rows, profile = rank_datasets(query, load_demo_datasets(), parser, engine, explainer)
    query_profile_box(profile)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Portal baseline</div>', unsafe_allow_html=True)
        st.caption("Literal matching with a recency bias, approximating the current portal behavior described in the chapter.")
        for result in portal_rows[:top_n]:
            result_card(result, "Portal baseline")

    with right:
        st.markdown('<div class="section-title">Fuzzy prototype</div>', unsafe_allow_html=True)
        st.caption("The same corpus ranked by thematic similarity, metadata completeness, resource availability, and recency.")
        for result in fuzzy_rows[:top_n]:
            result_card(result, "Fuzzy ranking")

    union_ids = {result.dataset["id"] for result in portal_rows[:top_n]} | {result.dataset["id"] for result in fuzzy_rows[:top_n]}
    comparison_rows: List[Dict[str, Any]] = []
    portal_lookup = {result.dataset["id"]: result for result in portal_rows}
    fuzzy_lookup = {result.dataset["id"]: result for result in fuzzy_rows}

    for dataset_id in union_ids:
        portal_item = portal_lookup.get(dataset_id)
        fuzzy_item = fuzzy_lookup.get(dataset_id)
        if portal_item is None or fuzzy_item is None:
            continue
        comparison_rows.append(
            {
                "Dataset": dataset_title(portal_item.dataset),
                "Portal rank": portal_item.rank,
                "Fuzzy rank": fuzzy_item.rank,
                "Shift": portal_item.rank - fuzzy_item.rank,
            }
        )

    if comparison_rows:
        st.markdown("---")
        st.markdown('<div class="section-title">Rank shifts</div>', unsafe_allow_html=True)
        comparison_df = pd.DataFrame(comparison_rows).sort_values(["Fuzzy rank", "Portal rank"])
        st.dataframe(comparison_df, width="stretch", hide_index=True)

    if view_mode == "Portal only":
        st.warning("Portal-only view selected. The fuzzy prototype remains available above for comparison.")
    elif view_mode == "Fuzzy only":
        st.success("Fuzzy-only view selected. The baseline is still shown to anchor the comparison.")


def render_method_tab(defuzzification_method: str, parser: Any, engine: Any) -> None:
    st.markdown('<div class="section-title">Method view</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">This view explains how the prototype turns a query and dataset metadata into a graded relevance score.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    with cols[0]:
        st.markdown('<div class="card-compact"><strong>1. Query parsing</strong><div class="small-muted">Detect language, vague predicates, and thematic hints.</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="card-compact"><strong>2. Fuzzy scoring</strong><div class="small-muted">Combine recency, completeness, similarity, and resources.</div></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="card-compact"><strong>3. Explanation</strong><div class="small-muted">Expose the decision path in human-readable form.</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Current backend status</div>', unsafe_allow_html=True)
    backend_cols = st.columns(3)
    backend_cols[0].metric("Query parser", "Ready" if parser is not None else "Fallback")
    backend_cols[1].metric("Fuzzy engine", "Ready" if engine is not None else "Fallback")
    backend_cols[2].metric("Defuzzification", defuzzification_method)

    st.markdown(
        '<div class="callout">The prototype uses the thesis backend when available. If a module is unavailable in the environment, the app falls back to a deterministic scoring path so the demo still runs.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    sample = load_demo_datasets()[0]
    sample_inputs = fuzzy_inputs("air quality Zurich", sample)
    st.markdown('<div class="section-title">Sample feature values</div>', unsafe_allow_html=True)
    sample_cols = st.columns(4)
    sample_cols[0].metric("Thematic similarity", f"{sample_inputs['thematic_similarity']:.0%}")
    sample_cols[1].metric("Recency", f"{sample_inputs['recency']:.0f} days")
    sample_cols[2].metric("Completeness", f"{sample_inputs['completeness']:.0%}")
    sample_cols[3].metric("Resources", f"{sample_inputs['resource_availability']:.0f}")

    st.caption("The values above are derived from the demo corpus and match the quality dimensions described in the chapter.")


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "query_input" not in st.session_state:
        st.session_state.query_input = "recent environmental data"

    parser, engine, explainer = get_components(st.session_state.get("defuzzification_method", "centroid"))

    st.markdown(
        """
        <div class="hero">
            <h1>Swiss OGD Portal Analysis Prototype</h1>
            <p>A thesis prototype built from the portal-analysis chapter. It demonstrates why literal search fails and how a fuzzy HCIR ranking can address vague, multilingual, and quality-aware queries.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sidebar = st.sidebar
    sidebar.markdown("### Prototype settings")
    sidebar.selectbox(
        "View mode",
        ["Compare", "Portal only", "Fuzzy only"],
        index=0,
        key="view_mode",
    )
    sidebar.slider("Top results", min_value=3, max_value=6, value=5, key="top_n")
    sidebar.selectbox(
        "Defuzzification",
        ["centroid", "bisector", "mom", "som", "lom"],
        index=0,
        key="defuzzification_method",
    )
    sidebar.markdown("---")
    sidebar.markdown("### Chapter focus")
    sidebar.caption("L1-L6 from the portal analysis are reflected in the UI and the result comparison.")
    sidebar.caption("The demo corpus mirrors the document's example queries and dataset types.")

    st.markdown("### Search query")
    st.text_input(
        "Enter a search phrase",
        key="query_input",
        placeholder="Try: recent environmental data, données climatiques récentes, air pollution measurements",
    )

    query = st.session_state.get("query_input", "")

    tabs = st.tabs(["Overview", "Search demo", "Method"])
    with tabs[0]:
        render_overview_tab()
    with tabs[1]:
        render_search_tab(query, st.session_state.view_mode, st.session_state.top_n, parser, engine, explainer)
    with tabs[2]:
        render_method_tab(st.session_state.defuzzification_method, parser, engine)

    st.markdown("---")
    st.caption("Prototype for the master thesis: Improving Access to Swiss Open Government Data through Fuzzy Human-Centered Information Retrieval.")


if __name__ == "__main__":
    main()
