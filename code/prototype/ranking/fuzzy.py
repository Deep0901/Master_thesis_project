from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .baselines import deduplicate_datasets
from .models import DatasetResult, FuzzyMembership, RankingFactors


class CalibratedFuzzyEngine:
    """Mamdani Fuzzy Inference Engine calibrated from opendata.swiss statistics."""

    RECENCY_MF = {
        "very_recent": ("trimf", [0, 0, 30]),
        "recent": ("trapmf", [7, 30, 150, 365]),
        "moderate": ("trapmf", [180, 365, 776, 1200]),
        "old": ("trapmf", [776, 1200, 2500, 3500]),
        "very_old": ("trimf", [2500, 3500, 4500]),
    }

    COMPLETENESS_MF = {
        "low": ("trimf", [0, 0, 0.55]),
        "partial": ("trimf", [0.45, 0.60, 0.72]),
        "medium": ("trapmf", [0.65, 0.72, 0.78, 0.85]),
        "high": ("trapmf", [0.75, 0.83, 0.92, 0.97]),
        "complete": ("trimf", [0.90, 0.97, 1.0]),
    }

    RESOURCES_MF = {
        "minimal": ("trimf", [0, 1, 2]),
        "limited": ("trimf", [1, 2, 4]),
        "moderate": ("trimf", [2, 4, 6]),
        "rich": ("trapmf", [4, 6, 10, 15]),
        "comprehensive": ("trimf", [10, 20, 50]),
    }

    SIMILARITY_MF = {
        "not_relevant": ("trimf", [0, 0, 0.15]),
        "somewhat_relevant": ("trimf", [0.10, 0.25, 0.40]),
        "relevant": ("trimf", [0.30, 0.50, 0.70]),
        "highly_relevant": ("trimf", [0.60, 0.78, 0.92]),
        "exact_match": ("trimf", [0.85, 0.95, 1.0]),
    }

    RELEVANCE_MF = {
        "very_low": ("trimf", [0, 0, 0.25]),
        "low": ("trimf", [0.10, 0.28, 0.45]),
        "moderate": ("trimf", [0.35, 0.50, 0.65]),
        "good": ("trimf", [0.55, 0.72, 0.88]),
        "excellent": ("trimf", [0.80, 0.92, 1.0]),
    }

    RULES = [
        ({"similarity": "exact_match", "recency": "very_recent", "completeness": "high"}, "excellent"),
        ({"similarity": "exact_match", "recency": "very_recent"}, "excellent"),
        ({"similarity": "exact_match", "completeness": "complete"}, "excellent"),
        ({"similarity": "highly_relevant", "recency": "recent", "completeness": "high"}, "excellent"),
        ({"similarity": "highly_relevant", "recency": "recent"}, "good"),
        ({"similarity": "highly_relevant", "completeness": "high"}, "good"),
        ({"similarity": "relevant", "recency": "very_recent", "completeness": "high"}, "good"),
        ({"similarity": "relevant", "resources": "comprehensive"}, "good"),
        ({"similarity": "relevant", "completeness": "medium"}, "moderate"),
        ({"similarity": "relevant", "recency": "moderate"}, "moderate"),
        ({"similarity": "somewhat_relevant", "recency": "very_recent"}, "moderate"),
        ({"similarity": "highly_relevant", "recency": "old"}, "moderate"),
        ({"recency": "very_old"}, "low"),
        ({"completeness": "low"}, "low"),
        ({"completeness": "low", "similarity": "somewhat_relevant"}, "very_low"),
        ({"similarity": "not_relevant"}, "very_low"),
        ({"resources": "minimal", "completeness": "low"}, "very_low"),
        ({"recency": "very_recent", "completeness": "complete", "resources": "rich"}, "excellent"),
        ({"recency": "recent", "completeness": "high", "resources": "moderate"}, "good"),
        ({"recency": "old", "resources": "comprehensive"}, "moderate"),
    ]

    def __init__(self, calibration_path: Optional[str] = None):
        self.universe_points = 1000

        self.recency_mf = dict(self.RECENCY_MF)
        self.completeness_mf = dict(self.COMPLETENESS_MF)
        self.resources_mf = dict(self.RESOURCES_MF)
        self.similarity_mf = dict(self.SIMILARITY_MF)

        self._try_load_calibration(calibration_path)

    def _try_load_calibration(self, calibration_path: Optional[str]) -> None:
        try:
            default_path = "analytics/fuzzy_calibration_live.json"
            path = calibration_path or default_path
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return

        mf_payload = (payload or {}).get("membership_functions", {})

        def to_mf_dict(mapping: Dict[str, List[float]]) -> Dict[str, Tuple[str, List[float]]]:
            out: Dict[str, Tuple[str, List[float]]] = {}
            for term, params in (mapping or {}).items():
                if not isinstance(params, list):
                    continue
                if len(params) == 3:
                    out[term] = ("trimf", [float(x) for x in params])
                elif len(params) == 4:
                    out[term] = ("trapmf", [float(x) for x in params])
            return out

        rec = mf_payload.get("recency")
        comp = mf_payload.get("completeness")
        res = mf_payload.get("resources")

        if isinstance(rec, dict):
            loaded = to_mf_dict(rec)
            if loaded:
                self.recency_mf = loaded

        if isinstance(comp, dict):
            loaded = to_mf_dict(comp)
            if loaded:
                self.completeness_mf = loaded

        if isinstance(res, dict):
            loaded = to_mf_dict(res)
            if loaded:
                self.resources_mf = loaded

    def _triangular_mf(self, x: float, params: List[float]) -> float:
        a, b, c = params
        if a == b and b == c:
            return 1.0
        if a == b:
            if x <= b:
                return 1.0
            if x < c:
                return (c - x) / (c - b) if c != b else 1.0
            return 0.0
        if b == c:
            if x >= b:
                return 1.0
            if x > a:
                return (x - a) / (b - a) if b != a else 1.0
            return 0.0
        if x < a or x > c:
            return 0.0
        if a <= x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        return (c - x) / (c - b) if c != b else 1.0

    def _trapezoidal_mf(self, x: float, params: List[float]) -> float:
        a, b, c, d = params
        if a == b and c == d:
            return 1.0 if a <= x <= d else 0.0
        if x < a or x > d:
            return 0.0
        if a <= x <= b:
            return 1.0 if b == a else (x - a) / (b - a)
        if b < x < c:
            return 1.0
        return 1.0 if d == c else (d - x) / (d - c)

    def _compute_membership(self, value: float, mf_type: str, params: List[float]) -> float:
        if mf_type == "trimf":
            return self._triangular_mf(value, params)
        if mf_type == "trapmf":
            return self._trapezoidal_mf(value, params)
        return 0.0

    def fuzzify(self, variable: str, value: float) -> FuzzyMembership:
        mf_dict = {
            "recency": self.recency_mf,
            "completeness": self.completeness_mf,
            "resources": self.resources_mf,
            "similarity": self.similarity_mf,
        }

        if variable not in mf_dict:
            return FuzzyMembership(variable, value, {})

        memberships: Dict[str, float] = {}
        for term, (mf_type, params) in mf_dict[variable].items():
            memberships[term] = self._compute_membership(value, mf_type, params)

        return FuzzyMembership(variable, value, memberships)

    def evaluate_rules(self, fuzzified_inputs: Dict[str, FuzzyMembership]) -> Dict[str, float]:
        output_memberships: defaultdict[str, float] = defaultdict(float)

        for antecedents, consequent in self.RULES:
            firing_strength = 1.0
            for var, term in antecedents.items():
                if var in fuzzified_inputs:
                    membership = fuzzified_inputs[var].memberships.get(term, 0.0)
                    firing_strength = min(firing_strength, membership)

            output_memberships[consequent] = max(output_memberships[consequent], firing_strength)

        return dict(output_memberships)

    def defuzzify(self, output_memberships: Dict[str, float]) -> float:
        x = np.linspace(0, 1, self.universe_points)
        aggregated = np.zeros(self.universe_points)

        for term, strength in output_memberships.items():
            if term in self.RELEVANCE_MF and strength > 0:
                mf_type, params = self.RELEVANCE_MF[term]
                for i, xi in enumerate(x):
                    mu = self._compute_membership(float(xi), mf_type, params)
                    aggregated[i] = max(aggregated[i], min(mu, strength))

        if float(np.sum(aggregated)) > 0:
            centroid = float(np.sum(x * aggregated) / np.sum(aggregated))
            return centroid

        return 0.5

    def infer(
        self,
        *,
        recency_days: float,
        completeness: float,
        resource_count: int,
        similarity: float,
    ) -> Tuple[float, Dict[str, FuzzyMembership]]:
        fuzzified = {
            "recency": self.fuzzify("recency", recency_days),
            "completeness": self.fuzzify("completeness", completeness),
            "resources": self.fuzzify("resources", resource_count),
            "similarity": self.fuzzify("similarity", similarity),
        }

        output_memberships = self.evaluate_rules(fuzzified)
        relevance = self.defuzzify(output_memberships)
        return relevance, fuzzified


class MetadataAnalyzer:
    """Analyzes dataset metadata quality (proxy completeness score)."""

    FIELD_WEIGHTS = {
        "title": 1.0,
        "description": 1.0,
        "organization": 0.8,
        "resources": 0.9,
        "tags": 0.6,
        "groups": 0.7,
        "license": 0.5,
        "metadata_modified": 0.4,
    }

    def compute_completeness(self, dataset: Dict) -> float:
        total_weight = sum(self.FIELD_WEIGHTS.values())
        achieved_weight = 0.0

        for field, weight in self.FIELD_WEIGHTS.items():
            value = dataset.get(field)

            if field == "title":
                if isinstance(value, dict):
                    achieved_weight += weight if any(value.values()) else 0
                elif value:
                    achieved_weight += weight

            elif field == "description":
                if isinstance(value, dict):
                    desc = " ".join(str(v) for v in value.values() if v)
                else:
                    desc = str(value) if value else ""

                if len(desc) >= 50:
                    achieved_weight += weight
                elif len(desc) >= 20:
                    achieved_weight += weight * 0.5

            elif field == "resources":
                resources = value or []
                if len(resources) >= 3:
                    achieved_weight += weight
                elif len(resources) >= 1:
                    achieved_weight += weight * 0.7

            elif field == "tags":
                tags = value or []
                if len(tags) >= 3:
                    achieved_weight += weight
                elif len(tags) >= 1:
                    achieved_weight += weight * 0.5

            elif field == "groups":
                if value:
                    achieved_weight += weight

            elif field == "organization":
                org = value or {}
                if isinstance(org, dict) and (org.get("name") or org.get("title")):
                    achieved_weight += weight
                elif org:
                    achieved_weight += weight * 0.5

            elif value:
                achieved_weight += weight

        return achieved_weight / total_weight if total_weight > 0 else 0.0


class MultilingualQueryProcessor:
    """Processes queries with multilingual stopwords and vague predicate detection."""

    STOPWORDS = {
        "en": {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "data",
            "dataset",
            "datasets",
            "related",
            "show",
            "statistics",
            "statistic",
        },
        "de": {
            "der",
            "die",
            "das",
            "den",
            "dem",
            "des",
            "ein",
            "eine",
            "einer",
            "einem",
            "einen",
            "und",
            "oder",
            "aber",
            "in",
            "auf",
            "an",
            "zu",
            "für",
            "von",
            "mit",
            "bei",
            "nach",
            "über",
            "unter",
            "vor",
            "durch",
            "ist",
            "sind",
            "war",
            "waren",
            "sein",
            "hat",
            "haben",
            "wird",
            "werden",
            "kann",
            "können",
            "muss",
            "müssen",
            "soll",
            "sollen",
            "ich",
            "du",
            "er",
            "sie",
            "es",
            "wir",
            "ihr",
            "daten",
            "datensatz",
            "zusammenhang",
            "statistik",
        },
        "fr": {
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "et",
            "ou",
            "mais",
            "dans",
            "sur",
            "à",
            "de",
            "pour",
            "par",
            "avec",
            "sans",
            "sous",
            "vers",
            "est",
            "sont",
            "était",
            "être",
            "avoir",
            "a",
            "ont",
            "fait",
            "je",
            "tu",
            "il",
            "elle",
            "nous",
            "vous",
            "ils",
            "elles",
            "ce",
            "cette",
            "ces",
            "qui",
            "que",
            "quoi",
            "dont",
            "où",
            "données",
            "jeu",
            "statistiques",
        },
        "it": {
            "il",
            "lo",
            "la",
            "i",
            "gli",
            "le",
            "un",
            "uno",
            "una",
            "e",
            "o",
            "ma",
            "in",
            "su",
            "a",
            "di",
            "da",
            "per",
            "con",
            "tra",
            "fra",
            "è",
            "sono",
            "era",
            "essere",
            "avere",
            "ha",
            "hanno",
            "fatto",
            "io",
            "tu",
            "lui",
            "lei",
            "noi",
            "voi",
            "loro",
            "che",
            "chi",
            "dati",
            "insieme",
            "statistiche",
        },
    }

    THEME_KEYWORDS = {
        "environment": {"environment", "umwelt", "environnement", "ambiente", "pollution", "climate", "clima"},
        "mobility": {
            "mobility",
            "transport",
            "verkehr",
            "traffic",
            "road",
            "rail",
            "transit",
            "bicycle",
            "bike",
            "biking",
            "cycling",
            "cycle",
            "velo",
            "vélo",
            "fahrrad",
            "bicicletta",
        },
        "health": {"health", "gesundheit", "santé", "salute", "hospital"},
        "education": {"education", "bildung", "éducation", "istruzione", "school"},
        "economy": {"economy", "wirtschaft", "économie", "economia", "finance", "employment"},
        "population": {"population", "bevölkerung", "demographic", "demographie"},
    }

    VAGUE_PREDICATES = {
        "recency": {
            "recent",
            "new",
            "latest",
            "fresh",
            "current",
            "updated",
            "neu",
            "aktuell",
            "neueste",
            "frisch",
            "récent",
            "nouveau",
            "actuel",
            "dernier",
            "recente",
            "nuovo",
            "attuale",
            "ultimo",
        },
        "completeness": {
            "complete",
            "comprehensive",
            "full",
            "detailed",
            "documented",
            "well-documented",
            "thorough",
            "vollständig",
            "komplett",
            "vollumfänglich",
            "dokumentiert",
            "complet",
            "détaillé",
            "documenté",
            "completo",
            "dettagliato",
            "documentato",
        },
        "quality": {
            "quality",
            "good",
            "reliable",
            "accurate",
            "verified",
            "qualität",
            "gut",
            "zuverlässig",
            "genau",
            "qualité",
            "bon",
            "fiable",
            "précis",
            "qualità",
            "buono",
            "affidabile",
            "preciso",
        },
    }

    def __init__(self):
        self.all_stopwords = set()
        for sw_set in self.STOPWORDS.values():
            self.all_stopwords.update(sw_set)

    def detect_language(self, query: str) -> str:
        words = set(query.lower().split())
        scores: Dict[str, int] = {}
        for lang, stopwords in self.STOPWORDS.items():
            scores[lang] = len(words & stopwords)
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return "en"

    def extract_keywords(self, query: str) -> List[str]:
        words = re.findall(r"\b\w+\b", query.lower())
        return [w for w in words if w not in self.all_stopwords and len(w) > 2]

    def extract_themes(self, query: str) -> List[str]:
        words = set(re.findall(r"\b\w+\b", query.lower()))
        themes: List[str] = []
        for theme, keywords in self.THEME_KEYWORDS.items():
            if words & keywords:
                themes.append(theme)
        return themes

    def detect_vague_predicates(self, query: str) -> Dict[str, bool]:
        query_lower = query.lower()
        words = set(re.findall(r"\b\w+\b", query_lower))
        detected: Dict[str, bool] = {}
        for predicate_type, terms in self.VAGUE_PREDICATES.items():
            detected[predicate_type] = bool(words & terms)
        return detected

    def process(self, query: str) -> Dict[str, Any]:
        keywords = self.extract_keywords(query)
        themes = self.extract_themes(query)

        # Optional LLM normalization: best-effort, ignore failures.
        try:
            from code.config import load_config_from_env
            from code.query_processing import create_normalizer

            config = load_config_from_env()
            if config.enable_llm_normalization:
                normalizer = create_normalizer(use_openai=(config.llm.provider == "openai"))
                normalization = normalizer.normalize(query)
                llm_tokens: List[str] = []
                for text in [
                    normalization.normalized_query,
                    normalization.english_translation,
                    " ".join(normalization.synonyms),
                    " ".join(normalization.related_terms),
                ]:
                    llm_tokens.extend(re.findall(r"\b\w+\b", str(text).lower()))
                keywords = list(dict.fromkeys(keywords + [token for token in llm_tokens if len(token) > 2]))
        except Exception:
            pass

        vague_predicates = self.detect_vague_predicates(query)

        return {
            "original": query,
            "language": self.detect_language(query),
            "keywords": keywords,
            "themes": themes,
            "vague_predicates": vague_predicates,
            "has_vague_terms": any(vague_predicates.values()),
        }


class SimilarityCalculator:
    """Calculate query-document similarity using BM25-style scoring."""

    def __init__(self):
        self._document_count = 0
        self._document_frequencies: Dict[str, int] = {}
        self._average_document_length = 0.0
        self._k1 = 1.5
        self._b = 0.75

    @staticmethod
    def _flatten_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            return " ".join(
                SimilarityCalculator._flatten_text(item)
                for item in value.values()
                if item is not None
            )
        if isinstance(value, (list, tuple, set)):
            return " ".join(SimilarityCalculator._flatten_text(item) for item in value)
        return str(value)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, datasets: List[Dict]) -> None:
        self._document_count = len(datasets)
        self._document_frequencies = {}
        total_length = 0

        for dataset in datasets:
            doc_tokens_list = self._tokenize(self._get_doc_text(dataset))
            total_length += len(doc_tokens_list)
            doc_tokens = set(doc_tokens_list)
            for token in doc_tokens:
                self._document_frequencies[token] = self._document_frequencies.get(token, 0) + 1

        self._average_document_length = total_length / self._document_count if self._document_count else 0.0

    def _idf(self, term: str) -> float:
        if self._document_count <= 0:
            return 1.0
        document_frequency = self._document_frequencies.get(term, 0)
        numerator = self._document_count - document_frequency + 0.5
        denominator = document_frequency + 0.5
        if numerator <= 0 or denominator <= 0:
            return 0.0
        return math.log((numerator / denominator) + 1.0)

    def _get_doc_text(self, dataset: Dict) -> str:
        title_text = self._flatten_text(dataset.get("title", {}))
        desc_text = self._flatten_text(dataset.get("description", {}))
        tags_text = self._flatten_text(dataset.get("tags", []))
        groups_text = self._flatten_text(dataset.get("groups", []))
        org_text = self._flatten_text(dataset.get("organization", {}))

        parts = [
            title_text,
            title_text,
            title_text,
            tags_text,
            tags_text,
            groups_text,
            groups_text,
        ]

        if title_text or tags_text or groups_text:
            if desc_text:
                parts.append(desc_text)
        else:
            parts.extend([desc_text, org_text])

        return " ".join(part for part in parts if part)

    def _apply_theme_boost(
        self,
        score: float,
        dataset: Dict,
        *,
        query_themes: Optional[List[str]] = None,
        query_terms: Optional[List[str]] = None,
    ) -> float:
        title_text = self._flatten_text(dataset.get("title", "")).lower()
        tags_text = self._flatten_text(dataset.get("tags", [])).lower()
        groups_text = self._flatten_text(dataset.get("groups", [])).lower()
        searchable_text = " ".join([title_text, tags_text, groups_text])

        if query_terms:
            high_priority = {"bicycle", "bike", "biking", "cycling", "cycle", "velo", "vélo", "fahrrad", "bicicletta"}
            medium_priority = {"mobility", "transport", "verkehr", "traffic", "transit", "transportation"}
            low_priority = {"data", "dataset", "datasets", "related", "show", "statistics", "statistic"}

            exact_matches = 0.0
            for term in query_terms:
                term_l = term.lower()
                if term_l in searchable_text:
                    if term_l in high_priority:
                        exact_matches += 1.6
                    elif term_l in medium_priority:
                        exact_matches += 1.15
                    elif term_l in low_priority:
                        exact_matches += 0.75
                    else:
                        exact_matches += 1.0

            if exact_matches:
                score = min(score * (1.0 + min(0.12 * exact_matches, 0.45)), 1.0)

        if not query_themes:
            return score

        theme_terms = {
            "environment": ["environment", "umwelt", "environnement", "ambiente", "pollution", "climate"],
            "mobility": [
                "mobility",
                "transport",
                "verkehr",
                "traffic",
                "road",
                "rail",
                "bicycle",
                "bike",
                "cycling",
                "velo",
                "vélo",
                "fahrrad",
                "transit",
                "transportation",
            ],
            "health": ["health", "gesundheit", "santé", "salute", "hospital"],
            "education": ["education", "bildung", "éducation", "istruzione", "school"],
            "economy": ["economy", "wirtschaft", "économie", "economia", "finance", "employment"],
            "population": ["population", "bevölkerung", "demographic", "demographie"],
        }

        boost = 0.0
        for theme in query_themes:
            if any(term in searchable_text for term in theme_terms.get(theme, [theme])):
                boost += 0.12

        return min(score * (1.0 + min(boost, 0.30)), 1.0)

    def calculate(self, query_keywords: List[str], dataset: Dict, *, query_themes: Optional[List[str]] = None) -> float:
        if not query_keywords:
            return 0.5

        doc_text = self._get_doc_text(dataset).lower()
        doc_tokens = self._tokenize(doc_text)
        if not doc_tokens:
            return 0.0

        query_tokens = self._tokenize(" ".join(query_keywords))
        if not query_tokens:
            return 0.0

        query_term_freq = defaultdict(int)
        for token in query_tokens:
            query_term_freq[token] += 1

        doc_term_freq = defaultdict(int)
        for token in doc_tokens:
            doc_term_freq[token] += 1

        shared_terms = set(query_term_freq).intersection(doc_term_freq)
        if not shared_terms:
            return self._apply_theme_boost(0.0, dataset, query_themes=query_themes, query_terms=query_tokens)

        document_length = len(doc_tokens)
        average_document_length = self._average_document_length or float(document_length or 1)

        score = 0.0
        for term in query_term_freq:
            tf = doc_term_freq.get(term, 0)
            if tf <= 0:
                continue

            idf = self._idf(term)
            if idf <= 0:
                continue

            numerator = tf * (self._k1 + 1.0)
            denominator = tf + self._k1 * (1.0 - self._b + self._b * (document_length / average_document_length))
            if denominator <= 0:
                continue

            query_boost = (1.0 + math.log(query_term_freq[term])) if query_term_freq[term] > 1 else 1.0

            term_l = term.lower()
            high_priority = {"bicycle", "bike", "biking", "cycling", "cycle", "velo", "vélo", "fahrrad", "bicicletta"}
            medium_priority = {"mobility", "transport", "verkehr", "traffic", "transit", "transportation"}
            low_priority = {"data", "dataset", "datasets", "related", "show", "statistics", "statistic"}

            if term_l in high_priority:
                query_boost *= 2.0
            elif term_l in medium_priority:
                query_boost *= 1.3
            elif term_l in low_priority:
                query_boost *= 0.5

            score += idf * (numerator / denominator) * query_boost

        coverage = len(shared_terms) / float(len(set(query_tokens)))

        high_priority = {"bicycle", "bike", "biking", "cycling", "cycle", "velo", "vélo", "fahrrad", "bicicletta"}
        high_priority_in_query = any(term.lower() in high_priority for term in query_tokens)
        high_priority_matched = any(term.lower() in high_priority for term in shared_terms)
        if high_priority_in_query and not high_priority_matched:
            coverage *= 0.65

        similarity = (score / (score + 1.0) if score > 0 else 0.0) * coverage
        return self._apply_theme_boost(similarity, dataset, query_themes=query_themes, query_terms=query_tokens)


class ExplanationGenerator:
    """Generates human-readable explanations for ranking decisions."""

    RECENCY_EXPLANATIONS = {
        "very_recent": "📅 Very recent data (updated within the last month)",
        "recent": "📅 Recently updated data (within the past year)",
        "moderate": "📅 Moderately recent data (1-3 years old)",
        "old": "📅 Older dataset (more than 3 years since update)",
        "very_old": "📅 Historical data (not recently updated)",
    }

    COMPLETENESS_EXPLANATIONS = {
        "complete": "📋 Excellent metadata documentation",
        "high": "📋 Well-documented with comprehensive metadata",
        "medium": "📋 Adequately documented metadata",
        "partial": "📋 Partial metadata documentation",
        "low": "📋 Minimal metadata available",
    }

    RESOURCE_EXPLANATIONS = {
        "comprehensive": "📁 Extensive resource collection with multiple formats",
        "rich": "📁 Multiple resources and formats available",
        "moderate": "📁 Several resources provided",
        "limited": "📁 Limited resources available",
        "minimal": "📁 Single resource only",
    }

    SIMILARITY_EXPLANATIONS = {
        "exact_match": "🎯 Excellent match to your search terms",
        "highly_relevant": "🎯 Highly relevant to your query",
        "relevant": "🎯 Relevant to your search",
        "somewhat_relevant": "🎯 Partially matches your query",
        "not_relevant": "🎯 Limited relevance to search terms",
    }

    def generate(
        self,
        factors: RankingFactors,
        vague_predicates: Dict[str, bool],
        *,
        display_score: Optional[float] = None,
    ) -> str:
        lines: List[str] = []

        score_value = display_score if display_score is not None else factors.fuzzy_relevance
        score_pct = int(score_value * 100)
        if score_pct >= 80:
            lines.append(f"**Excellent Match** (Relevance: {score_pct}%)")
        elif score_pct >= 60:
            lines.append(f"**Good Match** (Relevance: {score_pct}%)")
        elif score_pct >= 40:
            lines.append(f"**Moderate Match** (Relevance: {score_pct}%)")
        else:
            lines.append(f"**Lower Relevance** (Score: {score_pct}%)")

        if display_score is not None and abs(display_score - factors.fuzzy_relevance) > 0.01:
            lines.append(f"_Fuzzy base score: {int(factors.fuzzy_relevance * 100)}%_")

        lines.append("")

        factor_lines: List[str] = []

        sim_exp = self.SIMILARITY_EXPLANATIONS.get(
            factors.similarity_term,
            f"🎯 Similarity: {int(factors.similarity_score * 100)}%",
        )
        factor_lines.append(sim_exp)

        rec_exp = self.RECENCY_EXPLANATIONS.get(
            factors.recency_term,
            f"📅 Recency score: {int(factors.recency_score * 100)}%",
        )
        if vague_predicates.get("recency"):
            factor_lines.insert(0, f"**{rec_exp}** ← You asked for recent data")
        else:
            factor_lines.append(rec_exp)

        comp_exp = self.COMPLETENESS_EXPLANATIONS.get(
            factors.completeness_term,
            f"📋 Completeness: {int(factors.completeness_score * 100)}%",
        )
        if vague_predicates.get("completeness"):
            factor_lines.insert(0, f"**{comp_exp}** ← You asked for complete data")
        else:
            factor_lines.append(comp_exp)

        res_exp = self.RESOURCE_EXPLANATIONS.get(
            factors.resource_term,
            f"📁 Resources: {int(factors.resource_score * 100)}%",
        )
        factor_lines.append(res_exp)

        lines.extend(factor_lines)
        return "\n".join(lines)


class FuzzyHCIRRanker:
    """Main fuzzy human-centered ranking system."""

    def __init__(self):
        self.fuzzy_engine = CalibratedFuzzyEngine()
        self.metadata_analyzer = MetadataAnalyzer()
        self.query_processor = MultilingualQueryProcessor()
        self.similarity_calculator = SimilarityCalculator()
        self.explanation_generator = ExplanationGenerator()

    def rank(
        self,
        datasets: List[Dict],
        query: str,
        *,
        factor_weights: Optional[Dict[str, float]] = None,
    ) -> List[DatasetResult]:
        datasets = deduplicate_datasets(datasets)

        query_info = self.query_processor.process(query)
        keywords = query_info["keywords"]
        vague_predicates = query_info["vague_predicates"]

        self.similarity_calculator.fit(datasets)

        results: List[DatasetResult] = []

        weights = factor_weights or {}
        w_recency = float(weights.get("recency", 1.0))
        w_completeness = float(weights.get("completeness", 1.0))
        w_resources = float(weights.get("resources", 1.0))
        w_similarity = float(weights.get("similarity", 1.0))

        for ds in datasets:
            try:
                mod_str = ds.get("metadata_modified", "")
                if mod_str:
                    mod_date = datetime.fromisoformat(mod_str.replace("Z", "+00:00"))
                    days_since = (datetime.now(mod_date.tzinfo) - mod_date).days
                else:
                    days_since = 365
            except Exception:
                days_since = 365

            completeness = self.metadata_analyzer.compute_completeness(ds)

            resources = ds.get("resources", [])
            resource_count = len(resources) if resources else 0

            similarity = self.similarity_calculator.calculate(
                keywords,
                ds,
                query_themes=query_info.get("themes", []),
            )

            fuzzy_relevance, fuzzified = self.fuzzy_engine.infer(
                recency_days=float(days_since),
                completeness=float(completeness),
                resource_count=int(resource_count),
                similarity=float(similarity),
            )

            recency_score = 1.0 - min(float(days_since) / 3650.0, 1.0)
            resource_score = min(float(resource_count) / 10.0, 1.0)

            denom = (w_recency + w_completeness + w_resources + w_similarity)
            weighted_factor_score = (
                (w_recency * recency_score)
                + (w_completeness * completeness)
                + (w_resources * resource_score)
                + (w_similarity * similarity)
            ) / denom if denom > 0 else (recency_score + completeness + resource_score + similarity) / 4.0

            relevance = float((0.65 * fuzzy_relevance) + (0.35 * weighted_factor_score))

            factors = RankingFactors(
                recency_score=recency_score,
                completeness_score=completeness,
                resource_score=resource_score,
                similarity_score=similarity,
                fuzzy_relevance=fuzzy_relevance,
                recency_term=fuzzified["recency"].dominant_term[0],
                completeness_term=fuzzified["completeness"].dominant_term[0],
                resource_term=fuzzified["resources"].dominant_term[0],
                similarity_term=fuzzified["similarity"].dominant_term[0],
            )

            explanation = self.explanation_generator.generate(
                factors,
                vague_predicates,
                display_score=relevance,
            )

            title = ds.get("title", {})
            if isinstance(title, str):
                title = {"en": title}

            description = ds.get("description", {})
            if isinstance(description, dict):
                desc_text = (
                    description.get("en")
                    or description.get("de")
                    or description.get("fr")
                    or next(iter(description.values()), "")
                )
            else:
                desc_text = str(description) if description else ""

            org = ds.get("organization", {})
            if isinstance(org, dict):
                org_name = org.get("title") or org.get("name") or "Unknown"
            else:
                org_name = str(org) if org else "Unknown"

            groups = ds.get("groups", [])
            themes = [g.get("name", "") for g in groups if isinstance(g, dict)]

            tags = ds.get("tags", [])
            tag_names: List[str] = []
            for tag in tags:
                if isinstance(tag, dict):
                    tag_names.append(tag.get("name", ""))
                else:
                    tag_names.append(str(tag))

            result = DatasetResult(
                id=str(ds.get("name", ds.get("id", ""))),
                title=title,
                description=str(desc_text)[:500],
                organization=str(org_name),
                resources=resources,
                themes=themes,
                tags=tag_names[:10],
                modified=str(ds.get("metadata_modified", "")),
                created=str(ds.get("metadata_created", "")),
                license=str(ds.get("license_id", "Unknown")),
                url=f"https://opendata.swiss/en/dataset/{ds.get('name', '')}",
                relevance_score=relevance,
                factors=factors,
                explanation=explanation,
            )

            results.append(result)

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        for index, result in enumerate(results):
            result.rank = index + 1
        return results
