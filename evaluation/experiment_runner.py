"""
Experimental Evaluation Runner

This module runs complete experimental comparisons between:
1. Fuzzy OGD Retrieval System (our approach)
2. Keyword Baseline (BM25/TF-IDF)
3. Simple Metadata Ranking

Research Questions Evaluated:
- RQ2: Does fuzzy ranking outperform keyword-based baseline?
- RQ4: How does fuzzy ranking compare to alternative approaches?

Methodology:
- Uses benchmark queries with ground truth relevance judgments
- Computes standard IR metrics (MAP, nDCG, P@K, R@K)
- Reports statistical significance via paired t-tests

Author: Deep Shukla
Thesis: Improving Access to Swiss OGD through Fuzzy HCIR
University of Fribourg, Human-IST Institute
"""

import json
import csv
import time
import logging
import numpy as np
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
import requests
import re
import math
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluation_framework import (
    EvaluationEngine, EvaluationQuery, RankingResult, 
    RelevanceJudgment, IRMetrics,
    compute_quadratic_weighted_kappa,
    compute_percentage_agreement,
    compute_disagreement_count,
    landis_koch_category,
)
from code.prototype.ranking.fuzzy import FuzzyHCIRRanker
from code.prototype.ranking.ai_semantic_baseline import AISemanticBaseline
from code.fuzzy_system.fuzzy_rules import RuleBase
from code.fuzzy_system.inference_engine import MamdaniInferenceEngine
from code.ranking.fuzzy_ranker import SimilarityCalculator, MetadataScorer


@dataclass
class RuleWeightSensitivityConfig:
    """Configuration for a single rule-weight sensitivity analysis run."""
    name: str
    display_name: str
    description: str
    category_multipliers: Dict[str, float]


def _categorize_rule(rule: Any) -> str:
    """Assign a rule to a logical sensitivity-analysis category."""
    variables = {var for var, _ in rule.antecedents}
    if "thematic_similarity" in variables:
        return "thematic_similarity"
    if "recency" in variables:
        return "recency"
    if "completeness" in variables:
        return "metadata_quality"
    if "resource_availability" in variables:
        return "resource_availability"
    return "other"


def _group_rules_by_category(rule_base: RuleBase) -> Dict[str, List[Any]]:
    """Group rules by logical category for sensitivity analysis."""
    categories = {
        "metadata_quality": [],
        "thematic_similarity": [],
        "recency": [],
        "resource_availability": [],
        "other": [],
    }
    for rule in rule_base.get_rules():
        categories[_categorize_rule(rule)].append(rule)
    return categories


# =============================================================================
# RETRIEVAL SYSTEMS
# =============================================================================

class BaseRetriever:
    """Base class for retrieval systems."""
    
    name = "base"
    BASE_URL = "https://opendata.swiss/api/3/action"
    
    def search(self, query: str, num_results: int = 100) -> List[Tuple[str, float]]:
        """
        Search and return ranked results.
        
        Returns:
            List of (dataset_id, score) tuples
        """
        raise NotImplementedError


def _load_query_records(path: Path) -> List[Dict[str, Any]]:
    """Load benchmark query records from the existing benchmark files.

    Args:
        path: Path to the benchmark query file.

    Returns:
        Normalized list of query records with query_id and query_text.
    """
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "queries" in payload:
        raw_queries = payload["queries"]
    elif isinstance(payload, dict) and "benchmark_queries" in payload:
        raw_queries = payload["benchmark_queries"]
    elif isinstance(payload, list):
        raw_queries = payload
    else:
        raw_queries = []

    records: List[Dict[str, Any]] = []
    for item in raw_queries:
        if not isinstance(item, dict):
            continue

        query_id = str(item.get("id") or item.get("query_id") or "").strip()
        query_text = (
            item.get("query")
            or item.get("query_text")
            or item.get("query_de")
            or item.get("query_en")
            or ""
        )

        if not query_id or not query_text:
            continue

        records.append(
            {
                "query_id": query_id,
                "query_text": str(query_text),
                "domain": str(item.get("domain", "")),
                "query_language": str(item.get("language") or item.get("query_language") or "de"),
                "intent": str(item.get("intent", "")),
            }
        )

    return records


def _fetch_dataset_metadata(dataset_id: str) -> Dict[str, Any]:
    """Fetch dataset metadata for pooled candidate export.

    Args:
        dataset_id: CKAN dataset identifier.

    Returns:
        Dataset metadata payload or an empty dictionary on failure.
    """
    if not dataset_id:
        return {}

    try:
        resp = requests.get(
            f"{BaseRetriever.BASE_URL}/package_show",
            params={"id": dataset_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("result", {})
    except Exception:
        return {}


def _dataset_title(dataset: Dict[str, Any]) -> str:
    """Extract a readable dataset title from a CKAN payload."""
    title = dataset.get("title", "")
    if isinstance(title, dict):
        for language in ("de", "en", "fr", "it"):
            if title.get(language):
                return str(title[language])
        return str(next(iter(title.values()), ""))
    return str(title or dataset.get("name", ""))


def _ensure_directory(path: Path) -> Path:
    """Create a directory and all parents if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mean_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute a t-based confidence interval for a list of values."""
    if not values:
        return 0.0, 0.0

    if len(values) == 1:
        return float(values[0]), float(values[0])

    from scipy import stats

    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    sem = float(stats.sem(array))
    if np.isnan(sem):
        return mean, mean

    interval = stats.t.interval(confidence, len(array) - 1, loc=mean, scale=sem)
    lower = float(interval[0]) if interval[0] is not None else mean
    upper = float(interval[1]) if interval[1] is not None else mean
    return lower, upper


def _bootstrap_mean_difference(
    values_a: List[float],
    values_b: List[float],
    iterations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap the mean difference between two paired score lists."""
    if len(values_a) != len(values_b):
        raise ValueError("Bootstrap lists must have identical lengths")
    if not values_a:
        return 0.0, 0.0, 0.0

    diffs = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap_means = []
    sample_size = len(diffs)

    for _ in range(iterations):
        sample = rng.choice(diffs, size=sample_size, replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    lower = float(np.percentile(bootstrap_means, 2.5))
    upper = float(np.percentile(bootstrap_means, 97.5))
    return float(np.mean(diffs)), lower, upper


def _holm_bonferroni(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni correction to a list of p-values."""
    if not p_values:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    m = len(p_values)

    for rank, (original_index, p_value) in enumerate(indexed, start=1):
        corrected = min(1.0, (m - rank + 1) * float(p_value))
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max

    return adjusted


def _metric_display_name(metric_key: str) -> str:
    """Map internal metric keys to publication names."""
    mapping = {
        "average_precision": "MAP",
        "precision_at_5": "P@5",
        "ndcg_at_10": "nDCG@10",
        "reciprocal_rank": "MRR",
    }
    return mapping.get(metric_key, metric_key)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float with a fallback."""
    try:
        return float(value)
    except Exception:
        return default


class RuleWeightSensitivityRetriever(BaseRetriever):
    """Use the existing Mamdani inference engine with configurable rule weights."""

    name = "fuzzy_sensitivity"

    def __init__(self, config: RuleWeightSensitivityConfig):
        self.config = config
        self.similarity_calc = SimilarityCalculator()
        self.metadata_scorer = MetadataScorer()
        self.inference_engine = self._build_inference_engine()

    def _build_inference_engine(self) -> MamdaniInferenceEngine:
        rule_base = RuleBase()
        categories = _group_rules_by_category(rule_base)

        for category, rules in categories.items():
            multiplier = self.config.category_multipliers.get(category, 1.0)
            for rule in rules:
                rule.weight = max(0.0, 1.0 * multiplier)

        return MamdaniInferenceEngine(rule_base=rule_base, defuzzification_method="centroid")

    def _fetch_candidate_datasets(self, query: str, num_results: int = 100) -> List[Dict[str, Any]]:
        params = {"q": query, "rows": num_results}
        try:
            resp = requests.get(f"{self.BASE_URL}/package_search", params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()["result"]["results"]
        except Exception as exc:
            logger.warning("Rule-weight sensitivity retriever failed for query '%s': %s", query, exc)
            return []

    def _to_metadata_features(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        modified = dataset.get("metadata_modified") or dataset.get("metadata_modified_date") or ""
        try:
            from datetime import datetime
            if modified:
                mod_date = datetime.fromisoformat(str(modified).replace("Z", "+00:00")).date()
                now = datetime.now().date()
                recency_days = max(0, (now - mod_date).days)
            else:
                recency_days = 730
        except Exception:
            recency_days = 730

        metadata = {
            "title": dataset.get("title", ""),
            "description": dataset.get("notes") or dataset.get("description", ""),
            "tags": dataset.get("tags", []),
            "groups": dataset.get("groups", []),
            "organization": dataset.get("organization", {}),
            "license_id": dataset.get("license_id"),
            "resources": dataset.get("resources", []),
            "days_since_modified": recency_days,
        }
        return metadata

    def search(self, query: str, num_results: int = 10) -> List[Tuple[str, float]]:
        datasets = self._fetch_candidate_datasets(query, num_results=num_results)
        if not datasets:
            return []

        self.similarity_calc.fit(datasets)
        from code.query_processing import create_parser

        parser = create_parser()
        parsed_query = parser.parse(query)
        query_terms = parsed_query.keywords if parsed_query else []
        query_themes = parsed_query.themes if parsed_query else None

        ranked_results: List[Tuple[float, str]] = []
        for dataset in datasets:
            metadata = self._to_metadata_features(dataset)
            thematic_similarity = self.similarity_calc.calculate(query_terms, dataset, query_themes=query_themes)
            recency = self.metadata_scorer.calculate_recency(metadata.get("days_since_modified"))
            completeness = self.metadata_scorer.calculate_completeness(metadata)
            resource_availability = self.metadata_scorer.calculate_resource_availability(metadata)

            result = self.inference_engine.infer(
                {
                    "recency": recency,
                    "completeness": completeness,
                    "thematic_similarity": thematic_similarity,
                    "resource_availability": resource_availability,
                }
            )
            ranked_results.append((result.crisp_output, str(dataset.get("id") or dataset.get("name") or "")))

        ranked_results.sort(key=lambda item: item[0], reverse=True)
        return [(dataset_id, score) for score, dataset_id in ranked_results[:num_results]]


class PortalBaseline(BaseRetriever):
    """
    Baseline: Use opendata.swiss portal's default search.
    
    This represents what users currently get with the default search.
    """
    
    name = "portal_default"
    
    def search(self, query: str, num_results: int = 100) -> List[Tuple[str, float]]:
        params = {
            'q': query,
            'rows': num_results
        }
        
        try:
            resp = requests.get(f"{self.BASE_URL}/package_search", params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json()['result']['results']
            
            # Portal returns results in ranked order
            # Assign decreasing scores based on position
            ranked = []
            for i, ds in enumerate(results):
                score = 1.0 - (i / (len(results) + 1))  # Decreasing scores
                ranked.append((ds['id'], score))
            
            return ranked
            
        except Exception as e:
            print(f"Portal search error: {e}")
            return []


class KeywordBaseline(BaseRetriever):
    """
    Baseline: BM25/TF-IDF keyword scoring.
    
    Implements BM25 scoring on metadata fields (title, description, keywords).
    """
    
    name = "keyword_bm25"
    
    # BM25 parameters
    K1 = 1.2
    B = 0.75
    
    def __init__(self):
        self.doc_cache = {}  # Cache fetched documents
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        if not text:
            return []
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove stopwords (basic German + English)
        stopwords = {'der', 'die', 'das', 'und', 'in', 'von', 'zu', 'mit', 'für',
                     'the', 'and', 'of', 'to', 'in', 'for', 'is', 'on', 'with'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]
    
    def _get_doc_text(self, dataset: Dict) -> str:
        """Extract searchable text from dataset."""
        parts = []
        
        # Title (with boost - repeat 3 times)
        title = dataset.get('title', {})
        if isinstance(title, dict):
            for lang in ['de', 'en', 'fr']:
                if title.get(lang):
                    parts.extend([title[lang]] * 3)  # Boost title
        elif title:
            parts.extend([str(title)] * 3)
        
        # Description
        desc = dataset.get('notes', '') or dataset.get('description', {})
        if isinstance(desc, dict):
            for lang in ['de', 'en', 'fr']:
                if desc.get(lang):
                    parts.append(desc[lang])
        elif desc:
            parts.append(str(desc))
        
        # Keywords
        for tag in dataset.get('tags', []):
            if isinstance(tag, dict):
                parts.append(tag.get('name', ''))
            else:
                parts.append(str(tag))
        
        # Organization
        org = dataset.get('organization', {})
        if isinstance(org, dict):
            parts.append(org.get('name', ''))
        
        return ' '.join(parts)
    
    def _bm25_score(self, query_terms: List[str], doc_terms: List[str], 
                   avg_doc_len: float, doc_freqs: Dict[str, int]) -> float:
        """
        Compute BM25 score for a document.
        """
        doc_len = len(doc_terms)
        term_freq = defaultdict(int)
        for term in doc_terms:
            term_freq[term] += 1
        
        score = 0.0
        N = 1000  # Assumed collection size
        
        for term in query_terms:
            if term not in term_freq:
                continue
            
            tf = term_freq[term]
            df = doc_freqs.get(term, 1)
            
            # IDF component
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            
            # TF component with length normalization
            numerator = tf * (self.K1 + 1)
            denominator = tf + self.K1 * (1 - self.B + self.B * (doc_len / avg_doc_len))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, num_results: int = 100) -> List[Tuple[str, float]]:
        # First, get candidates from portal
        params = {
            'q': query,
            'rows': num_results
        }
        
        try:
            resp = requests.get(f"{self.BASE_URL}/package_search", params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json()['result']['results']
        except Exception as e:
            print(f"Search error: {e}")
            return []
        
        if not results:
            return []
        
        # Tokenize query
        query_terms = self._tokenize(query)
        
        # Process documents
        doc_terms_all = []
        doc_ids = []
        
        for ds in results:
            doc_text = self._get_doc_text(ds)
            terms = self._tokenize(doc_text)
            doc_terms_all.append(terms)
            doc_ids.append(ds['id'])
        
        # Compute document frequencies
        doc_freqs = defaultdict(int)
        for terms in doc_terms_all:
            for term in set(terms):
                doc_freqs[term] += 1
        
        # Average document length
        avg_doc_len = np.mean([len(t) for t in doc_terms_all]) if doc_terms_all else 1
        
        # Score each document
        scores = []
        for terms in doc_terms_all:
            score = self._bm25_score(query_terms, terms, avg_doc_len, doc_freqs)
            scores.append(score)
        
        # Normalize scores
        max_score = max(scores) if scores and max(scores) > 0 else 1
        normalized = [(doc_ids[i], scores[i] / max_score) for i in range(len(scores))]
        
        # Sort by score
        normalized.sort(key=lambda x: x[1], reverse=True)
        
        return normalized


class MetadataQualityRanker(BaseRetriever):
    """
    Baseline: Pure metadata quality-based ranking.
    
    Ranks by metadata quality factors without fuzzy inference:
    - Recency
    - Completeness
    - Resource availability
    
    Uses simple weighted sum instead of fuzzy rules.
    """
    
    name = "metadata_quality"
    
    # Weights for metadata factors
    WEIGHT_RELEVANCE = 0.5  # Keyword match
    WEIGHT_RECENCY = 0.2
    WEIGHT_COMPLETENESS = 0.15
    WEIGHT_RESOURCES = 0.15
    
    def _compute_recency_score(self, days: int) -> float:
        """Linear recency score (newer = higher)."""
        if days <= 0:
            return 1.0
        elif days <= 30:
            return 0.95
        elif days <= 180:
            return 0.8
        elif days <= 365:
            return 0.6
        elif days <= 730:
            return 0.4
        else:
            return 0.2
    
    def _compute_completeness_score(self, dataset: Dict) -> float:
        """Simple completeness check."""
        checks = [
            bool(dataset.get('title')),
            bool(dataset.get('notes') or dataset.get('description')),
            bool(dataset.get('organization')),
            len(dataset.get('tags', [])) >= 1,
            len(dataset.get('groups', [])) >= 1,
            bool(dataset.get('license_id')),
        ]
        return sum(checks) / len(checks)
    
    def _compute_resource_score(self, dataset: Dict) -> float:
        """Score based on resource count."""
        num_resources = len(dataset.get('resources', []))
        if num_resources >= 5:
            return 1.0
        elif num_resources >= 3:
            return 0.8
        elif num_resources >= 1:
            return 0.5
        else:
            return 0.0
    
    def search(self, query: str, num_results: int = 100) -> List[Tuple[str, float]]:
        # Get candidates
        params = {'q': query, 'rows': num_results}
        
        try:
            resp = requests.get(f"{self.BASE_URL}/package_search", params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json()['result']['results']
        except Exception as e:
            print(f"Search error: {e}")
            return []
        
        if not results:
            return []
        
        # Compute relevance scores (position-based from portal)
        relevance_scores = {ds['id']: 1.0 - (i / (len(results) + 1)) 
                          for i, ds in enumerate(results)}
        
        # Score each dataset
        from datetime import datetime
        now = datetime.now()
        
        scored_results = []
        
        for ds in results:
            # Relevance from portal ranking
            relevance = relevance_scores[ds['id']]
            
            # Recency
            modified = ds.get('metadata_modified', '')
            try:
                if modified:
                    mod_date = datetime.strptime(modified.split('T')[0], '%Y-%m-%d')
                    days = (now - mod_date).days
                else:
                    days = 1000
            except:
                days = 1000
            recency = self._compute_recency_score(days)
            
            # Completeness
            completeness = self._compute_completeness_score(ds)
            
            # Resources
            resources = self._compute_resource_score(ds)
            
            # Weighted sum (no fuzzy inference)
            score = (
                self.WEIGHT_RELEVANCE * relevance +
                self.WEIGHT_RECENCY * recency +
                self.WEIGHT_COMPLETENESS * completeness +
                self.WEIGHT_RESOURCES * resources
            )
            
            scored_results.append((ds['id'], score))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results


class FuzzyRetriever(BaseRetriever):
    """
    Legacy fuzzy ranking used by the original benchmark runner.
    Preserved for backward compatibility and regression comparison.
    """
    
    name = "fuzzy_hcir"
    
    def __init__(self):
        # Load calibrated variables
        from code.fuzzy_system.calibrated_variables import (
            RECENCY_CALIBRATED,
            COMPLETENESS_CALIBRATED,
            RESOURCE_AVAILABILITY_CALIBRATED
        )
        self.recency_var = RECENCY_CALIBRATED
        self.completeness_var = COMPLETENESS_CALIBRATED
        self.resources_var = RESOURCE_AVAILABILITY_CALIBRATED
    
    def _triangular_mf(self, x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function."""
        if x <= a or x >= c:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a + 1e-10)
        else:
            return (c - x) / (c - b + 1e-10)
    
    def _trapezoidal_mf(self, x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal membership function."""
        if x <= a or x >= d:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a + 1e-10)
        elif x <= c:
            return 1.0
        else:
            return (d - x) / (d - c + 1e-10)
    
    def _compute_membership(self, value: float, term_def: Dict) -> float:
        """Compute membership degree for a term."""
        mf_type = term_def['type']
        params = term_def['params']
        
        if mf_type == 'triangular':
            return self._triangular_mf(value, *params)
        elif mf_type == 'trapezoidal':
            return self._trapezoidal_mf(value, *params)
        else:
            return 0.0
    
    def _fuzzy_inference(self, recency_days: float, completeness: float, 
                        resources: int, relevance: float) -> float:
        """
        Apply fuzzy rules to compute final score.
        
        Simplified rule base:
        - IF relevant AND recent AND complete THEN excellent
        - IF relevant AND (recent OR complete) THEN good
        - IF somewhat_relevant AND complete THEN moderate
        - etc.
        """
        # Compute memberships for recency
        recency_memberships = {}
        for term, defn in self.recency_var.terms.items():
            recency_memberships[term] = self._compute_membership(recency_days, defn)
        
        # Compute memberships for completeness
        completeness_memberships = {}
        for term, defn in self.completeness_var.terms.items():
            completeness_memberships[term] = self._compute_membership(completeness, defn)
        
        # Compute memberships for resources
        resource_memberships = {}
        for term, defn in self.resources_var.terms.items():
            resource_memberships[term] = self._compute_membership(float(resources), defn)
        
        # Fuzzy rules (simplified)
        rules = []
        
        # Rule 1: Very recent + high completeness + rich resources -> excellent
        rule1 = min(
            recency_memberships.get('very_recent', 0),
            completeness_memberships.get('high', 0),
            resource_memberships.get('rich', 0)
        )
        rules.append(('excellent', rule1))
        
        # Rule 2: Recent + medium completeness -> good
        rule2 = min(
            recency_memberships.get('recent', 0),
            completeness_memberships.get('medium', 0)
        )
        rules.append(('good', rule2))
        
        # Rule 3: Moderate recency + partial completeness -> moderate
        rule3 = min(
            recency_memberships.get('moderate', 0),
            completeness_memberships.get('partial', 0)
        )
        rules.append(('moderate', rule3))
        
        # Rule 4: Old or low completeness -> low
        rule4 = max(
            recency_memberships.get('old', 0),
            completeness_memberships.get('low', 0)
        )
        rules.append(('low', rule4))
        
        # Rule 5: Very old -> very low
        rule5 = recency_memberships.get('very_old', 0)
        rules.append(('very_low', rule5))
        
        # Combine with relevance (keyword match) as weight
        # More relevant documents get higher scores
        
        # Defuzzification (simplified centroid)
        output_centroids = {
            'excellent': 0.95,
            'good': 0.75,
            'moderate': 0.50,
            'low': 0.25,
            'very_low': 0.10
        }
        
        numerator = 0.0
        denominator = 0.0
        
        for term, strength in rules:
            weighted_strength = strength * relevance  # Weight by keyword relevance
            numerator += weighted_strength * output_centroids[term]
            denominator += weighted_strength
        
        if denominator == 0:
            return relevance * 0.5  # Default to half relevance
        
        return numerator / denominator
    
    def search(self, query: str, num_results: int = 100) -> List[Tuple[str, float]]:
        # Get candidates from portal
        params = {'q': query, 'rows': num_results}
        
        try:
            resp = requests.get(f"{self.BASE_URL}/package_search", params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json()['result']['results']
        except Exception as e:
            print(f"Search error: {e}")
            return []
        
        if not results:
            return []
        
        # Compute relevance scores from portal ranking
        relevance_scores = {ds['id']: 1.0 - (i / (len(results) + 1)) 
                          for i, ds in enumerate(results)}
        
        from datetime import datetime
        now = datetime.now()
        
        scored_results = []
        
        for ds in results:
            # Get metadata values
            modified = ds.get('metadata_modified', '')
            try:
                if modified:
                    mod_date = datetime.strptime(modified.split('T')[0], '%Y-%m-%d')
                    recency_days = (now - mod_date).days
                else:
                    recency_days = 1000
            except:
                recency_days = 1000
            
            # Completeness
            checks = [
                bool(ds.get('title')),
                bool(ds.get('notes') or ds.get('description')),
                bool(ds.get('organization')),
                len(ds.get('tags', [])) >= 1,
                len(ds.get('groups', [])) >= 1,
                bool(ds.get('license_id')),
            ]
            completeness = sum(checks) / len(checks)
            
            # Resources
            num_resources = len(ds.get('resources', []))
            
            # Portal relevance
            relevance = relevance_scores[ds['id']]
            
            # Fuzzy inference
            score = self._fuzzy_inference(recency_days, completeness, num_resources, relevance)
            
            scored_results.append((ds['id'], score))
        
        # Sort by fuzzy score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results


class FuzzyHCIRRankerAdapter(BaseRetriever):
    """
    Adapter for the application fuzzy ranker.

    This class fetches the same portal candidate metadata as the legacy
    FuzzyRetriever, then delegates ranking to the application-level
    FuzzyHCIRRanker. It converts DatasetResult output back into the
    benchmark's expected (dataset_uuid, score) tuple form.
    """

    name = "fuzzy_hcir"

    def __init__(self):
        self.rank_engine = FuzzyHCIRRanker()

    def _fetch_candidate_datasets(self, query: str, num_results: int = 100) -> List[Dict[str, Any]]:
        params = {'q': query, 'rows': num_results}
        try:
            resp = requests.get(f"{self.BASE_URL}/package_search", params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()['result']['results']
        except Exception as e:
            print(f"FuzzyHCIRRankerAdapter search error: {e}")
            return []

    def search(self, query: str, num_results: int = 100) -> List[Tuple[str, float]]:
        datasets = self._fetch_candidate_datasets(query, num_results=num_results)
        if not datasets:
            return []

        ranked_results = self.rank_engine.rank(datasets, query)

        # Map the original candidate metadata to CKAN UUID values.
        uuid_by_key = {}
        for ds in datasets:
            ds_id = str(ds.get('id', ''))
            ds_name = str(ds.get('name', ''))
            if ds_id:
                uuid_by_key[ds_id] = ds_id
            if ds_name:
                uuid_by_key[ds_name] = ds_id

        converted_results: List[Tuple[str, float]] = []
        for result in ranked_results:
            dataset_uuid = uuid_by_key.get(result.id, result.id)
            converted_results.append((dataset_uuid, result.relevance_score))

        return converted_results


class AISemanticBaselineAdapter(BaseRetriever):
    """Adapter for the AI semantic baseline used in the pooled workflow."""

    name = "semantic"

    def __init__(self):
        self.semantic_ranker = AISemanticBaseline()

    def _fetch_candidate_datasets(self, query: str, num_results: int = 100) -> List[Dict[str, Any]]:
        params = {"q": query, "rows": num_results}
        try:
            resp = requests.get(f"{self.BASE_URL}/package_search", params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()["result"]["results"]
        except Exception as e:
            print(f"AISemanticBaselineAdapter search error: {e}")
            return []

    def search(self, query: str, num_results: int = 100) -> List[Tuple[str, float]]:
        datasets = self._fetch_candidate_datasets(query, num_results=num_results)
        if not datasets:
            return []

        self.semantic_ranker.index_datasets(datasets)
        semantic_results = self.semantic_ranker.search(query, top_n=num_results)

        uuid_by_key: Dict[str, str] = {}
        for ds in datasets:
            ds_id = str(ds.get("id", ""))
            ds_name = str(ds.get("name", ""))
            if ds_id:
                uuid_by_key[ds_id] = ds_id
            if ds_name:
                uuid_by_key[ds_name] = ds_id

        converted_results: List[Tuple[str, float]] = []
        for result in semantic_results.results:
            dataset_uuid = uuid_by_key.get(result.dataset_id, result.dataset_id)
            converted_results.append((dataset_uuid, result.similarity_score))

        return converted_results


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Runs the full comparative experiment.
    """
    
    def __init__(
        self,
        ground_truth_file: str = "evaluation/ground_truth_final.json",
        benchmark_queries_file: str = "evaluation/benchmark_queries_v2.json",
    ):
        self.ground_truth = {}
        self.benchmark_queries: List[Dict[str, Any]] = []
        self.systems = {}
        self.engine = EvaluationEngine()
        self.ground_truth_file = Path(ground_truth_file)
        self.benchmark_queries_file = Path(benchmark_queries_file)
        self.output_dir = Path("evaluation/results")
        
        # Load ground truth
        if self.ground_truth_file.exists():
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                self.ground_truth = json.load(f)
            print(f"Loaded ground truth for {len(self.ground_truth)} queries")

        if self.benchmark_queries_file.exists():
            self.benchmark_queries = _load_query_records(self.benchmark_queries_file)
            print(f"Loaded benchmark queries for {len(self.benchmark_queries)} queries")
    
    def add_system(self, system: BaseRetriever):
        """Add a retrieval system to evaluate."""
        self.systems[system.name] = system

    def _resolve_ground_truth(self) -> Dict[str, Any]:
        """Load ground truth from the expected final file or supported fallbacks."""
        candidates = [self.ground_truth_file]
        candidates.extend(
            [
                Path("evaluation/ground_truth_auto.json"),
                Path("evaluation/ground_truth_manual.json"),
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                with open(candidate, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if payload:
                    self.ground_truth_file = candidate
                    self.ground_truth = payload
                    if candidate.name != "ground_truth_final.json":
                        output_path = Path("evaluation/ground_truth_final.json")
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, "w", encoding="utf-8") as handle:
                            json.dump(payload, handle, indent=2, ensure_ascii=False)
                        logger.info("Wrote fallback ground truth to %s", output_path)
                    return payload

        raise ValueError(
            "Ground truth data is missing. Expected evaluation/ground_truth_final.json "
            "or an available fallback file such as evaluation/ground_truth_auto.json."
        )

    def _validate_inputs(self) -> None:
        """Check that there is usable ground truth, at least one query, and at least one system."""
        if not self.ground_truth:
            raise ValueError("No ground truth judgments were loaded.")
        if not self.systems:
            raise ValueError("No retrieval systems were configured for evaluation.")
        if not self.benchmark_queries and not self.ground_truth:
            raise ValueError("No benchmark queries were loaded.")

    def _display_system_name(self, system_name: str) -> str:
        mapping = {
            "portal_default": "Portal",
            "keyword_bm25": "BM25",
            "metadata_quality": "Metadata",
            "fuzzy_hcir": "Fuzzy",
            "semantic": "Semantic",
        }
        return mapping.get(system_name, system_name.replace("_", " ").title())

    def _query_records(self) -> List[Dict[str, Any]]:
        """Return the query records that should drive the workflow."""
        if self.benchmark_queries:
            return self.benchmark_queries

        records: List[Dict[str, Any]] = []
        for query_id, data in self.ground_truth.items():
            query_info = data.get("query", {})
            records.append(
                {
                    "query_id": query_id,
                    "query_text": query_info.get("query_text", ""),
                    "domain": query_info.get("domain", ""),
                    "query_language": query_info.get("query_language", "de"),
                    "intent": query_info.get("intent", ""),
                }
            )
        return records

    def _load_query_ground_truth(self, query_id: str) -> List[RelevanceJudgment]:
        """Load relevance judgments for a single query from the ground-truth file."""
        data = self.ground_truth.get(query_id, {})
        judgments = data.get("judgments", [])
        return [
            RelevanceJudgment(
                query_id=query_id,
                dataset_id=str(j.get("dataset_id", "")),
                relevance=int(j.get("relevance", 0)),
                annotator=str(j.get("annotator", "")),
                notes=str(j.get("notes", "")),
            )
            for j in judgments
        ]

    def _build_pool_row(
        self,
        query: Dict[str, Any],
        dataset_id: str,
        dataset_title: str,
        system_ranks: Dict[str, Optional[int]],
        systems_found_in: List[str],
    ) -> Dict[str, Any]:
        """Build a pooled-candidate row for CSV export."""
        return {
            "query_id": query["query_id"],
            "query_text": query["query_text"],
            "dataset_id": dataset_id,
            "dataset_title": dataset_title,
            "systems_found_in": "|".join(dict.fromkeys(systems_found_in)),
            "portal_rank": system_ranks.get("portal_default") or "",
            "bm25_rank": system_ranks.get("bm25") or "",
            "metadata_rank": system_ranks.get("metadata_quality") or "",
            "fuzzy_rank": system_ranks.get("fuzzy_hcir") or "",
            "semantic_rank": system_ranks.get("semantic") or "",
            "judge1_grade": "",
            "judge2_grade": "",
            "adjudicated_grade": "",
            "notes": "",
        }

    def _ensure_output_dir(self) -> Path:
        """Create the evaluation output directories if needed."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "figures").mkdir(parents=True, exist_ok=True)
        Path("evaluation/data").mkdir(parents=True, exist_ok=True)
        return output_dir

    def _write_csv(self, rows: List[Dict[str, Any]], path: Path) -> None:
        """Write a list of dictionaries to a CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                handle.write("")
            return

        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _get_metric_rows(self) -> List[Dict[str, Any]]:
        """Build a per-query/per-system metrics table from evaluation results."""
        rows: List[Dict[str, Any]] = []
        for query_id in sorted(self.engine.queries):
            for system_name in sorted(self.systems):
                metrics = self.engine.metrics.get(query_id, {}).get(system_name)
                if metrics is None:
                    continue
                rows.append(
                    {
                        "query_id": query_id,
                        "system_name": self._display_system_name(system_name),
                        "MAP": round(metrics.average_precision, 6),
                        "P@5": round(metrics.precision_at_5, 6),
                        "nDCG@10": round(metrics.ndcg_at_10, 6),
                        "MRR": round(metrics.reciprocal_rank, 6),
                    }
                )
        return rows

    def _summarize_system_metrics(self) -> List[Dict[str, Any]]:
        """Compute descriptive statistics for each system and metric."""
        metric_names = ["MAP", "P@5", "nDCG@10", "MRR"]
        summaries: List[Dict[str, Any]] = []

        for system_name in sorted(self.systems):
            display_name = self._display_system_name(system_name)
            values_by_metric = {metric: [] for metric in metric_names}
            for query_id in sorted(self.engine.queries):
                metrics = self.engine.metrics.get(query_id, {}).get(system_name)
                if metrics is None:
                    continue
                values_by_metric["MAP"].append(metrics.average_precision)
                values_by_metric["P@5"].append(metrics.precision_at_5)
                values_by_metric["nDCG@10"].append(metrics.ndcg_at_10)
                values_by_metric["MRR"].append(metrics.reciprocal_rank)

            for metric_name in metric_names:
                values = np.asarray(values_by_metric[metric_name], dtype=float)
                if values.size == 0:
                    continue
                mean = float(np.mean(values))
                median = float(np.median(values))
                std = float(np.std(values, ddof=0))
                minimum = float(np.min(values))
                maximum = float(np.max(values))
                ci_lower, ci_upper = _mean_confidence_interval(values.tolist())
                summaries.append(
                    {
                        "system_name": display_name,
                        "metric": metric_name,
                        "mean": round(mean, 6),
                        "median": round(median, 6),
                        "std": round(std, 6),
                        "min": round(minimum, 6),
                        "max": round(maximum, 6),
                        "ci_lower": round(ci_lower, 6),
                        "ci_upper": round(ci_upper, 6),
                    }
                )

        return summaries

    def _compute_pairwise_statistics(self) -> List[Dict[str, Any]]:
        """Compute Wilcoxon signed-rank tests and Holm-Bonferroni corrections."""
        metric_names = ["MAP", "P@5", "nDCG@10", "MRR"]
        rows: List[Dict[str, Any]] = []
        system_names = sorted(self.systems)

        for metric_name in metric_names:
            metric_key = {
                "MAP": "average_precision",
                "P@5": "precision_at_5",
                "nDCG@10": "ndcg_at_10",
                "MRR": "reciprocal_rank",
            }[metric_name]
            p_values: List[float] = []
            comparisons: List[Dict[str, Any]] = []

            for left_index, system_a in enumerate(system_names):
                for system_b in system_names[left_index + 1 :]:
                    values_a = []
                    values_b = []
                    for query_id in sorted(self.engine.queries):
                        metrics_a = self.engine.metrics.get(query_id, {}).get(system_a)
                        metrics_b = self.engine.metrics.get(query_id, {}).get(system_b)
                        if metrics_a is None or metrics_b is None:
                            continue
                        values_a.append(getattr(metrics_a, metric_key))
                        values_b.append(getattr(metrics_b, metric_key))

                    if len(values_a) < 2 or len(values_b) < 2:
                        continue

                    try:
                        _, p_value = stats.wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided")
                    except Exception:
                        p_value = 1.0

                    p_values.append(float(p_value))
                    comparisons.append(
                        {
                            "system_a": self._display_system_name(system_a),
                            "system_b": self._display_system_name(system_b),
                            "metric": metric_name,
                            "statistic": 0.0,
                            "p_value": float(p_value),
                            "corrected_p_value": float(p_value),
                            "significant": bool(float(p_value) < 0.05),
                            "n_queries": len(values_a),
                        }
                    )

            adjusted = _holm_bonferroni(p_values)
            for comparison, adjusted_p in zip(comparisons, adjusted):
                comparison["corrected_p_value"] = round(float(adjusted_p), 6)
                comparison["significant"] = bool(float(adjusted_p) < 0.05)
                rows.append(comparison)

        return rows

    def _compute_bootstrap_intervals(self) -> List[Dict[str, Any]]:
        """Bootstrap the paired mean difference for each metric and system pair."""
        metric_names = ["MAP", "P@5", "nDCG@10", "MRR"]
        rows: List[Dict[str, Any]] = []
        system_names = sorted(self.systems)

        for metric_name in metric_names:
            metric_key = {
                "MAP": "average_precision",
                "P@5": "precision_at_5",
                "nDCG@10": "ndcg_at_10",
                "MRR": "reciprocal_rank",
            }[metric_name]
            for left_index, system_a in enumerate(system_names):
                for system_b in system_names[left_index + 1 :]:
                    values_a = []
                    values_b = []
                    for query_id in sorted(self.engine.queries):
                        metrics_a = self.engine.metrics.get(query_id, {}).get(system_a)
                        metrics_b = self.engine.metrics.get(query_id, {}).get(system_b)
                        if metrics_a is None or metrics_b is None:
                            continue
                        values_a.append(getattr(metrics_a, metric_key))
                        values_b.append(getattr(metrics_b, metric_key))

                    if not values_a or len(values_a) != len(values_b):
                        continue

                    mean_diff, lower, upper = _bootstrap_mean_difference(values_a, values_b, iterations=10000, seed=42)
                    rows.append(
                        {
                            "system_a": self._display_system_name(system_a),
                            "system_b": self._display_system_name(system_b),
                            "metric": metric_name,
                            "mean_difference": round(float(mean_diff), 6),
                            "ci_lower": round(float(lower), 6),
                            "ci_upper": round(float(upper), 6),
                        }
                    )

        return rows

    def _compute_win_loss(self) -> List[Dict[str, Any]]:
        """Compute win/loss/tie counts for pairwise system comparisons."""
        rows: List[Dict[str, Any]] = []
        system_names = sorted(self.systems)
        for left_index, system_a in enumerate(system_names):
            for system_b in system_names[left_index + 1 :]:
                wins = 0
                losses = 0
                ties = 0
                for query_id in sorted(self.engine.queries):
                    metrics_a = self.engine.metrics.get(query_id, {}).get(system_a)
                    metrics_b = self.engine.metrics.get(query_id, {}).get(system_b)
                    if metrics_a is None or metrics_b is None:
                        continue
                    score_a = metrics_a.ndcg_at_10
                    score_b = metrics_b.ndcg_at_10
                    if score_a > score_b + 1e-12:
                        wins += 1
                    elif score_b > score_a + 1e-12:
                        losses += 1
                    else:
                        ties += 1
                rows.append(
                    {
                        "system_a": self._display_system_name(system_a),
                        "system_b": self._display_system_name(system_b),
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                    }
                )
        return rows

    def _write_publication_tables(
        self,
        query_rows: List[Dict[str, Any]],
        summary_rows: List[Dict[str, Any]],
        pairwise_rows: List[Dict[str, Any]],
        bootstrap_rows: List[Dict[str, Any]],
        win_loss_rows: List[Dict[str, Any]],
    ) -> Path:
        """Write markdown publication tables for inclusion in the thesis."""
        output_path = self.output_dir / "publication_tables.md"
        lines: List[str] = []
        lines.append("# Evaluation Tables")
        lines.append("")
        lines.append("## Table 1. Per-query metrics")
        lines.append("")
        lines.append("| Query | System | MAP | P@5 | nDCG@10 | MRR |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in query_rows:
            lines.append(
                f"| {row['query_id']} | {row['system_name']} | {row['MAP']:.4f} | {row['P@5']:.4f} | {row['nDCG@10']:.4f} | {row['MRR']:.4f} |"
            )
        lines.append("")
        lines.append("## Table 2. Mean metrics")
        lines.append("")
        lines.append("| System | Metric | Mean | Median | Std | Min | Max | 95% CI |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in summary_rows:
            lines.append(
                f"| {row['system_name']} | {row['metric']} | {row['mean']:.4f} | {row['median']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} | [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] |"
            )
        lines.append("")
        lines.append("## Table 3. Pairwise significance")
        lines.append("")
        lines.append("| System A | System B | Metric | p | Corrected p | Significant |")
        lines.append("|---|---|---|---:|---:|---:|")
        for row in pairwise_rows:
            lines.append(
                f"| {row['system_a']} | {row['system_b']} | {row['metric']} | {row['p_value']:.4f} | {row['corrected_p_value']:.4f} | {str(row['significant']).lower()} |"
            )
        lines.append("")
        lines.append("## Table 4. Bootstrap confidence intervals")
        lines.append("")
        lines.append("| System A | System B | Metric | Mean diff | 95% CI |")
        lines.append("|---|---|---|---:|---:|")
        for row in bootstrap_rows:
            lines.append(
                f"| {row['system_a']} | {row['system_b']} | {row['metric']} | {row['mean_difference']:.4f} | [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] |"
            )
        lines.append("")
        lines.append("## Table 5. Win/Loss matrix")
        lines.append("")
        lines.append("| System | Wins/Losses/Ties vs. ... |")
        lines.append("|---|---|")
        for row in win_loss_rows:
            lines.append(
                f"| {row['system_a']} vs. {row['system_b']} | {row['wins']}/{row['losses']}/{row['ties']} |"
            )

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        return output_path

    def _create_visualizations(self, summary_rows: List[Dict[str, Any]], pairwise_rows: List[Dict[str, Any]]) -> None:
        """Generate publication-style figures with matplotlib."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        metric_names = ["MAP", "P@5", "nDCG@10", "MRR"]
        metric_data = {metric: [] for metric in metric_names}
        systems = []
        for row in summary_rows:
            if row["metric"] == "MAP":
                metric_data["MAP"].append((row["system_name"], row["mean"]))
            elif row["metric"] == "P@5":
                metric_data["P@5"].append((row["system_name"], row["mean"]))
            elif row["metric"] == "nDCG@10":
                metric_data["nDCG@10"].append((row["system_name"], row["mean"]))
            elif row["metric"] == "MRR":
                metric_data["MRR"].append((row["system_name"], row["mean"]))

        for metric_name in metric_names:
            data = metric_data[metric_name]
            if not data:
                continue
            systems = [name for name, _ in data]
            values = [value for _, value in data]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.boxplot([self._metric_series(metric_name, system_name) for system_name in systems], tick_labels=systems)
            ax.set_title(f"Distribution of {metric_name} scores")
            ax.set_ylabel(metric_name)
            plt.tight_layout()
            fig.savefig(figures_dir / f"boxplot_{metric_name.lower().replace('@', '').replace('10','')}.png", dpi=300)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(systems))
        width = 0.18
        for idx, metric_name in enumerate(metric_names):
            values = [value for _, value in metric_data[metric_name]]
            ax.bar(x + (idx - 1.5) * width, values, width=width, label=metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(systems)
        ax.set_title("Mean metric comparison")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(figures_dir / "bar_mean_metrics.png", dpi=300)
        plt.close(fig)

        labels = sorted({row["system_a"] for row in pairwise_rows} | {row["system_b"] for row in pairwise_rows})
        significance_matrix = []
        for row_name in labels:
            row_values = []
            for col_name in labels:
                if row_name == col_name:
                    row_values.append(0)
                    continue
                match = next((entry for entry in pairwise_rows if entry["system_a"] == row_name and entry["system_b"] == col_name and entry["metric"] == "nDCG@10"), None)
                if match is None:
                    match = next((entry for entry in pairwise_rows if entry["system_b"] == row_name and entry["system_a"] == col_name and entry["metric"] == "nDCG@10"), None)
                row_values.append(1 if match and match["significant"] else 0)
            significance_matrix.append(row_values)

        fig, ax = plt.subplots(figsize=(6.5, 5))
        if labels:
            sns.heatmap(np.array(significance_matrix), annot=True, fmt="d", cmap="viridis", xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
        else:
            ax.text(0.5, 0.5, "No significance comparisons available", ha="center", va="center")
        ax.set_title("Significance heatmap (nDCG@10)")
        plt.tight_layout()
        fig.savefig(figures_dir / "heatmap_significance.png", dpi=300)
        plt.close(fig)

    def _metric_series(self, metric_name: str, system_name: str) -> List[float]:
        values = []
        for query_id in sorted(self.engine.queries):
            metrics = self.engine.metrics.get(query_id, {}).get(next(key for key in self.systems if self._display_system_name(key) == system_name))
            if metrics is None:
                continue
            if metric_name == "MAP":
                values.append(metrics.average_precision)
            elif metric_name == "P@5":
                values.append(metrics.precision_at_5)
            elif metric_name == "nDCG@10":
                values.append(metrics.ndcg_at_10)
            elif metric_name == "MRR":
                values.append(metrics.reciprocal_rank)
        return values

    def _build_rule_weight_sensitivity_configs(self) -> List[RuleWeightSensitivityConfig]:
        """Create the predefined rule-weight sensitivity configurations."""
        return [
            RuleWeightSensitivityConfig(
                name="baseline",
                display_name="Baseline",
                description="Original rule weights (all weights = 1.0)",
                category_multipliers={},
            ),
            RuleWeightSensitivityConfig(
                name="metadata_focus",
                display_name="Metadata Focus",
                description="Emphasize metadata quality and resource rules",
                category_multipliers={
                    "metadata_quality": 1.30,
                    "resource_availability": 1.20,
                },
            ),
            RuleWeightSensitivityConfig(
                name="similarity_focus",
                display_name="Similarity Focus",
                description="Increase emphasis on thematic-similarity rules",
                category_multipliers={
                    "thematic_similarity": 1.30,
                },
            ),
            RuleWeightSensitivityConfig(
                name="recency_focus",
                display_name="Recency Focus",
                description="Increase emphasis on recency rules",
                category_multipliers={
                    "recency": 1.30,
                },
            ),
            RuleWeightSensitivityConfig(
                name="balanced_conservative",
                display_name="Balanced Conservative",
                description="Reduce all rule weights for a more conservative aggregation",
                category_multipliers={
                    "metadata_quality": 0.80,
                    "thematic_similarity": 0.80,
                    "recency": 0.80,
                    "resource_availability": 0.80,
                    "other": 0.85,
                },
            ),
        ]

    def _aggregate_sensitivity_query_metrics(self, query_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute the mean of the per-query sensitivity metrics for a configuration."""
        summary_metrics: Dict[str, float] = {}
        for metric_name in ("MAP", "P@5", "nDCG@10", "MRR"):
            values = [row.get(metric_name, 0.0) for row in query_rows]
            summary_metrics[metric_name] = round(float(np.mean(values)) if values else 0.0, 6)
        return summary_metrics

    def _run_single_sensitivity_configuration(
        self,
        config: RuleWeightSensitivityConfig,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Evaluate one rule-weight configuration with the existing evaluation engine."""
        evaluation_engine = EvaluationEngine()
        query_records = self._query_records()

        for query in query_records:
            query_id = query["query_id"]
            query_info = self.ground_truth.get(query_id, {}).get("query", {})
            judgments = self.ground_truth.get(query_id, {}).get("judgments", [])
            evaluation_engine.add_query(
                EvaluationQuery(
                    query_id=query_id,
                    query_text=query.get("query_text", ""),
                    query_language=query_info.get("query_language", "de"),
                    domain=query_info.get("domain", ""),
                    intent=query_info.get("intent", ""),
                    ground_truth=[
                        RelevanceJudgment(
                            query_id=query_id,
                            dataset_id=str(j.get("dataset_id", "")),
                            relevance=int(j.get("relevance", 0)),
                        )
                        for j in judgments
                    ],
                )
            )

        retriever = RuleWeightSensitivityRetriever(config)
        for query in query_records:
            query_id = query["query_id"]
            query_text = query.get("query_text", "")
            results = retriever.search(query_text, num_results=top_k)
            evaluation_engine.add_result(
                RankingResult(
                    system_name=config.name,
                    query_id=query_id,
                    ranked_docs=[dataset_id for dataset_id, _ in results],
                    scores=[score for _, score in results],
                    execution_time=0.0,
                )
            )

        evaluation_engine.evaluate_all()
        metrics = evaluation_engine.aggregate_metrics().get(config.name, {})
        query_rows: List[Dict[str, Any]] = []
        for query_id in sorted(evaluation_engine.queries):
            metric_entry = evaluation_engine.metrics.get(query_id, {}).get(config.name)
            if metric_entry is None:
                continue
            query_rows.append(
                {
                    "query_id": query_id,
                    "configuration": config.name,
                    "MAP": round(metric_entry.average_precision, 6),
                    "P@5": round(metric_entry.precision_at_5, 6),
                    "nDCG@10": round(metric_entry.ndcg_at_10, 6),
                    "MRR": round(metric_entry.reciprocal_rank, 6),
                }
            )

        summary_metrics = self._aggregate_sensitivity_query_metrics(query_rows)

        return {
            "configuration": config.name,
            "display_name": config.display_name,
            "description": config.description,
            "category_multipliers": dict(config.category_multipliers),
            "metrics": summary_metrics,
            "query_rows": query_rows,
        }

    def run_rule_weight_sensitivity_analysis(self, top_k: int = 10) -> Dict[str, Path]:
        """Run the optional rule-weight sensitivity analysis and write all outputs."""
        self._resolve_ground_truth()
        self._validate_inputs()
        self.output_dir = self._ensure_output_dir()

        configs = self._build_rule_weight_sensitivity_configs()
        results: List[Dict[str, Any]] = []
        all_query_rows: List[Dict[str, Any]] = []

        for config in configs:
            logger.info("Running sensitivity configuration: %s", config.display_name)
            config_result = self._run_single_sensitivity_configuration(config, top_k=top_k)
            results.append(config_result)
            all_query_rows.extend(config_result["query_rows"])

        summary_rows = []
        for config_result in results:
            metrics = config_result["metrics"]
            summary_rows.append(
                {
                    "configuration": config_result["configuration"],
                    "display_name": config_result["display_name"],
                    "description": config_result["description"],
                    "metadata_quality": config_result["category_multipliers"].get("metadata_quality", 1.0),
                    "thematic_similarity": config_result["category_multipliers"].get("thematic_similarity", 1.0),
                    "recency": config_result["category_multipliers"].get("recency", 1.0),
                    "resource_availability": config_result["category_multipliers"].get("resource_availability", 1.0),
                    "other": config_result["category_multipliers"].get("other", 1.0),
                    "MAP": round(_safe_float(metrics.get("MAP", 0.0)), 6),
                    "P@5": round(_safe_float(metrics.get("P@5", 0.0)), 6),
                    "nDCG@10": round(_safe_float(metrics.get("nDCG@10", 0.0)), 6),
                    "MRR": round(_safe_float(metrics.get("MRR", 0.0)), 6),
                    "num_queries": len(self.ground_truth),
                }
            )

        self._write_csv(summary_rows, self.output_dir / "sensitivity_results.csv")
        with open(self.output_dir / "sensitivity_results.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "configurations": [
                        {
                            "configuration": row["configuration"],
                            "display_name": row["display_name"],
                            "description": row["description"],
                            "category_multipliers": {
                                "metadata_quality": row["metadata_quality"],
                                "thematic_similarity": row["thematic_similarity"],
                                "recency": row["recency"],
                                "resource_availability": row["resource_availability"],
                                "other": row["other"],
                            },
                            "metrics": {
                                "MAP": row["MAP"],
                                "P@5": row["P@5"],
                                "nDCG@10": row["nDCG@10"],
                                "MRR": row["MRR"],
                            },
                            "num_queries": row["num_queries"],
                        }
                        for row in summary_rows
                    ],
                    "query_metrics": all_query_rows,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

        markdown_path = self.output_dir / "sensitivity_tables.md"
        with open(markdown_path, "w", encoding="utf-8") as handle:
            handle.write(self._build_sensitivity_markdown(summary_rows))

        self._create_sensitivity_plot(summary_rows)

        return {
            "sensitivity_results": self.output_dir / "sensitivity_results.csv",
            "sensitivity_results_json": self.output_dir / "sensitivity_results.json",
            "sensitivity_tables": markdown_path,
            "sensitivity_plot": self.output_dir / "figures" / "rule_weight_sensitivity.png",
        }

    def _build_sensitivity_markdown(self, summary_rows: List[Dict[str, Any]]) -> str:
        """Create a markdown summary table for rule-weight sensitivity analysis."""
        lines = [
            "# Rule Weight Sensitivity Analysis",
            "",
            "| Configuration | Description | Metadata | Similarity | Recency | Resources | MAP | P@5 | nDCG@10 | MRR |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary_rows:
            lines.append(
                f"| {row['display_name']} | {row['description']} | {row['metadata_quality']:.2f} | {row['thematic_similarity']:.2f} | {row['recency']:.2f} | {row['resource_availability']:.2f} | {row['MAP']:.4f} | {row['P@5']:.4f} | {row['nDCG@10']:.4f} | {row['MRR']:.4f} |"
            )
        return "\n".join(lines) + "\n"

    def _create_sensitivity_plot(self, summary_rows: List[Dict[str, Any]]) -> None:
        """Create a grouped bar chart for the sensitivity-analysis results."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = [row["display_name"] for row in summary_rows]
        metrics = ["MAP", "P@5", "nDCG@10", "MRR"]
        values_by_metric = {metric: [row[metric] for row in summary_rows] for metric in metrics}

        x = np.arange(len(labels))
        width = 0.18
        fig, ax = plt.subplots(figsize=(10, 5.5))

        for idx, metric in enumerate(metrics):
            ax.bar(x + (idx - 1.5) * width, values_by_metric[metric], width=width, label=metric)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Metric score")
        ax.set_title("Rule-weight sensitivity analysis")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.output_dir / "figures" / "rule_weight_sensitivity.png", dpi=300)
        plt.close(fig)

    def _validate_results(self) -> None:
        """Ensure every query has a result for every configured system."""
        missing: List[str] = []
        for query_id in self.ground_truth:
            for system_name in self.systems:
                if system_name not in self.engine.metrics.get(query_id, {}):
                    missing.append(f"{query_id}/{system_name}")
        if missing:
            raise ValueError(f"Evaluation is incomplete. Missing results for: {', '.join(missing[:10])}")

    def _write_outputs(self) -> Dict[str, Path]:
        """Write all requested evaluation artifacts to disk."""
        self.output_dir = self._ensure_output_dir()
        query_rows = self._get_metric_rows()
        summary_rows = self._summarize_system_metrics()
        pairwise_rows = self._compute_pairwise_statistics()
        bootstrap_rows = self._compute_bootstrap_intervals()
        win_loss_rows = self._compute_win_loss()

        self._write_csv(query_rows, self.output_dir / "query_metrics.csv")
        self._write_csv(summary_rows, self.output_dir / "system_summary.csv")
        self._write_csv(pairwise_rows, self.output_dir / "pairwise_statistics.csv")
        self._write_csv(bootstrap_rows, self.output_dir / "bootstrap_confidence_intervals.csv")
        self._write_csv(win_loss_rows, self.output_dir / "win_loss_matrix.csv")

        with open(self.output_dir / "experiment_results.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "systems": [self._display_system_name(name) for name in sorted(self.systems)],
                    "num_queries": len(self.ground_truth),
                    "num_systems": len(self.systems),
                    "num_pairwise_comparisons": len(pairwise_rows),
                    "query_metrics": query_rows,
                    "system_summary": summary_rows,
                    "pairwise_statistics": pairwise_rows,
                    "bootstrap_confidence_intervals": bootstrap_rows,
                    "win_loss": win_loss_rows,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

        publication_path = self._write_publication_tables(query_rows, summary_rows, pairwise_rows, bootstrap_rows, win_loss_rows)
        self._create_visualizations(summary_rows, pairwise_rows)
        return {
            "query_metrics": self.output_dir / "query_metrics.csv",
            "system_summary": self.output_dir / "system_summary.csv",
            "pairwise_statistics": self.output_dir / "pairwise_statistics.csv",
            "bootstrap_confidence_intervals": self.output_dir / "bootstrap_confidence_intervals.csv",
            "win_loss_matrix": self.output_dir / "win_loss_matrix.csv",
            "experiment_results": self.output_dir / "experiment_results.json",
            "publication_tables": publication_path,
        }

    def _query_records(self) -> List[Dict[str, Any]]:
        """Return the benchmark query records used to drive the workflow."""
        if self.benchmark_queries:
            return self.benchmark_queries

        records: List[Dict[str, Any]] = []
        for query_id, data in self.ground_truth.items():
            query_info = data.get("query", {})
            records.append(
                {
                    "query_id": query_id,
                    "query_text": query_info.get("query_text", ""),
                    "domain": query_info.get("domain", ""),
                    "query_language": query_info.get("query_language", "de"),
                    "intent": query_info.get("intent", ""),
                }
            )
        return records

    def _fetch_ranking_results(self, query_text: str, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Run all configured systems for a single query.

        Args:
            query_text: Benchmark query string.
            top_k: Number of results to retrieve from each system.

        Returns:
            Mapping of system name to ranked result tuples.
        """
        results_by_system: Dict[str, List[Tuple[str, float]]] = {}
        for system_name, system in self.systems.items():
            logger.info("Running %s", system_name.replace("_", " ").title())
            results_by_system[system_name] = system.search(query_text, num_results=top_k)
        return results_by_system

    def _has_existing_annotations(self, output_path: Path) -> bool:
        """Return True if the pooled-candidate CSV already contains manual annotations."""
        if not output_path.exists():
            return False

        try:
            with open(output_path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    for column in ("judge1_grade", "judge2_grade", "adjudicated_grade"):
                        if str(row.get(column, "")).strip():
                            return True
        except Exception as exc:
            logger.warning("Unable to inspect existing pooled candidates for annotations: %s", exc)

        return False

    def generate_pooled_candidates(
        self,
        output_file: str = "evaluation/data/pooled_candidates.csv",
        top_k: int = 10,
        regenerate_pooled: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run the retrieval systems and export pooled candidates to CSV.

        Args:
            output_file: CSV path for the pooled candidates.
            top_k: Number of top results to keep from each system.
            regenerate_pooled: If True, overwrite an existing annotated pooled-candidate file.

        Returns:
            List of pooled candidate rows.
        """
        queries = self._query_records()
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and self._has_existing_annotations(output_path) and not regenerate_pooled:
            print("Existing annotated pooled_candidates.csv detected.")
            print("Skipping regeneration to preserve human annotations.")
            try:
                with open(output_path, "r", newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    return [dict(row) for row in reader]
            except Exception as exc:
                logger.warning("Unable to read existing pooled candidates file: %s", exc)
                return []

        if output_path.exists() and regenerate_pooled:
            print("Warning: Existing pooled_candidates.csv will be overwritten because --regenerate-pooled was requested.")

        print("Generating pooled candidates...")
        logger.info("Pooling candidates")

        rows: List[Dict[str, Any]] = []

        for query in queries:
            logger.info("Running query %s", query["query_id"])
            results_by_system = self._fetch_ranking_results(query["query_text"], top_k=top_k)

            pooled: Dict[str, Dict[str, Any]] = {}
            for system_name, ranked_results in results_by_system.items():
                for rank, (dataset_id, _score) in enumerate(ranked_results[:top_k], start=1):
                    pooled_entry = pooled.setdefault(
                        dataset_id,
                        {
                            "dataset_id": dataset_id,
                            "dataset_title": "",
                            "system_ranks": {},
                            "systems_found_in": [],
                        },
                    )
                    pooled_entry["systems_found_in"].append(system_name)
                    pooled_entry["system_ranks"][system_name] = rank

                    if not pooled_entry["dataset_title"]:
                        dataset_metadata = _fetch_dataset_metadata(dataset_id)
                        pooled_entry["dataset_title"] = _dataset_title(dataset_metadata) or dataset_id

            logger.info("Removing duplicates")
            for pooled_entry in pooled.values():
                rows.append(
                    self._build_pool_row(
                        query=query,
                        dataset_id=pooled_entry["dataset_id"],
                        dataset_title=pooled_entry["dataset_title"],
                        system_ranks=pooled_entry["system_ranks"],
                        systems_found_in=pooled_entry["systems_found_in"],
                    )
                )

        fieldnames = [
            "query_id",
            "query_text",
            "dataset_id",
            "dataset_title",
            "systems_found_in",
            "portal_rank",
            "bm25_rank",
            "metadata_rank",
            "fuzzy_rank",
            "semantic_rank",
            "judge1_grade",
            "judge2_grade",
            "adjudicated_grade",
            "notes",
        ]

        print("Writing pooled_candidates.csv...")
        logger.info("Writing CSV")
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Generated {len(rows)} pooled candidates.")
        print(f"Output path: {output_path}")
        return rows

    def export_final_ground_truth(
        self,
        pooled_candidates_file: str = "evaluation/data/pooled_candidates.csv",
        output_file: str = "evaluation/ground_truth_final.json",
    ) -> Dict[str, Any]:
        """Export the final adjudicated ground truth JSON from the pooled CSV.

        Args:
            pooled_candidates_file: CSV containing adjudicated grades.
            output_file: JSON path for the final ground truth.

        Returns:
            JSON-compatible structure written to disk.
        """
        logger.info("Writing final ground truth")
        pooled_path = Path(pooled_candidates_file)
        if not pooled_path.exists():
            raise FileNotFoundError(f"Missing pooled candidates file: {pooled_candidates_file}")

        grouped: Dict[str, Dict[str, Any]] = {}
        with open(pooled_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                adjudicated = str(row.get("adjudicated_grade", "")).strip()
                if adjudicated == "":
                    continue

                query_id = row.get("query_id", "").strip()
                query_text = row.get("query_text", "").strip()
                query_bucket = grouped.setdefault(
                    query_id,
                    {
                        "query": {
                            "query_id": query_id,
                            "query_text": query_text,
                            "query_language": "de",
                            "domain": "",
                            "intent": "",
                            "expected_themes": [],
                            "ground_truth": [],
                        },
                        "judgments": [],
                    },
                )

                query_bucket["judgments"].append(
                    {
                        "dataset_id": row.get("dataset_id", "").strip(),
                        "dataset_title": row.get("dataset_title", "").strip(),
                        "relevance": int(float(adjudicated)),
                        "annotator": "adjudicated",
                        "notes": row.get("notes", "").strip(),
                    }
                )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(grouped, f, indent=2, ensure_ascii=False)

        return grouped

    def compute_agreement_statistics(
        self,
        pooled_candidates_file: str = "evaluation/data/pooled_candidates.csv",
    ) -> Dict[str, Any]:
        """Compute inter-annotator agreement metrics from the pooled CSV.

        Args:
            pooled_candidates_file: CSV containing judge grades.

        Returns:
            Dictionary with agreement metrics.
        """
        logger.info("Computing Cohen Kappa")
        pooled_path = Path(pooled_candidates_file)
        if not pooled_path.exists():
            raise FileNotFoundError(f"Missing pooled candidates file: {pooled_candidates_file}")

        judge1: List[int] = []
        judge2: List[int] = []
        with open(pooled_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                left = str(row.get("judge1_grade", "")).strip()
                right = str(row.get("judge2_grade", "")).strip()
                if left == "" or right == "":
                    continue
                judge1.append(int(float(left)))
                judge2.append(int(float(right)))

        if not judge1:
            return {
                "quadratic_weighted_kappa": 0.0,
                "percentage_agreement": 0.0,
                "disagreement_count": 0,
                "agreement_category": "poor",
                "num_compared": 0,
            }

        kappa = compute_quadratic_weighted_kappa(judge1, judge2)
        return {
            "quadratic_weighted_kappa": kappa,
            "percentage_agreement": compute_percentage_agreement(judge1, judge2),
            "disagreement_count": compute_disagreement_count(judge1, judge2),
            "agreement_category": landis_koch_category(kappa),
            "num_compared": len(judge1),
        }
    
    def run_experiment(self) -> Dict[str, Dict[str, float]]:
        """Run the complete retrieval and evaluation workflow for all systems."""
        self._resolve_ground_truth()
        self._validate_inputs()
        self.engine = EvaluationEngine()

        print("=" * 70)
        print("RUNNING COMPARATIVE EXPERIMENT")
        print("=" * 70)

        for query_id, data in self.ground_truth.items():
            query_info = data.get("query", {})
            judgments = data.get("judgments", [])
            eval_query = EvaluationQuery(
                query_id=query_id,
                query_text=query_info.get("query_text", ""),
                query_language=query_info.get("query_language", "de"),
                domain=query_info.get("domain", ""),
                intent=query_info.get("intent", ""),
                ground_truth=[
                    RelevanceJudgment(
                        query_id=query_id,
                        dataset_id=j.get("dataset_id", ""),
                        relevance=int(j.get("relevance", 0)),
                    )
                    for j in judgments
                ],
            )
            self.engine.add_query(eval_query)

        print("Computing metrics...")
        for system_name, system in self.systems.items():
            print(f"Running {self._display_system_name(system_name)}...")
            for query_id, data in self.ground_truth.items():
                query_text = data.get("query", {}).get("query_text", "")
                start_time = time.time()
                results = system.search(query_text, num_results=10)
                exec_time = time.time() - start_time
                result = RankingResult(
                    system_name=system_name,
                    query_id=query_id,
                    ranked_docs=[doc_id for doc_id, _score in results],
                    scores=[score for _doc_id, score in results],
                    execution_time=exec_time,
                )
                self.engine.add_result(result)
                time.sleep(0.1)

        self.engine.evaluate_all()
        self._validate_results()
        return self.engine.aggregate_metrics()
    
    def generate_report(self, output_file: Optional[str] = None):
        """Generate and save the experiment report and required outputs."""
        results = self.run_experiment()
        output_path = Path(output_file) if output_file else self.output_dir / "experiment_results.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(
                {
                    'timestamp': datetime.now().isoformat(),
                    'systems': [self._display_system_name(name) for name in sorted(self.systems)],
                    'num_queries': len(self.ground_truth),
                    'num_systems': len(self.systems),
                    'aggregate_metrics': results,
                },
                handle,
                indent=2,
            )
        print(self.engine.generate_report())
        print(f"\nResults saved to {output_path}")
        return results

    def run_complete_pipeline(
        self,
        include_sensitivity_analysis: bool = True,
        regenerate_pooled: bool = False,
    ) -> Dict[str, Any]:
        """Run the full evaluation protocol and optionally the weight-sensitivity analysis."""
        self.output_dir = self._ensure_output_dir()
        start_time = time.time()

        print("Generating tables...")
        print("Generating figures...")
        print("Computing significance...")
        print("Computing bootstrap...")

        self.generate_pooled_candidates(regenerate_pooled=regenerate_pooled)
        results = self.run_experiment()
        generated_files = self._write_outputs()
        sensitivity_files: Dict[str, Path] = {}
        if include_sensitivity_analysis:
            print("Running rule-weight sensitivity analysis...")
            sensitivity_files = self.run_rule_weight_sensitivity_analysis()
        execution_time = time.time() - start_time

        print("Finished.")
        print(f"Number of queries: {len(self.ground_truth)}")
        print(f"Number of systems: {len(self.systems)}")
        print(f"Number of pairwise comparisons: {len(self._compute_pairwise_statistics())}")
        print(f"Output directory: {self.output_dir}")
        print(f"Execution time: {execution_time:.2f}s")
        print("Generated files:")
        for path in sorted(generated_files.values()):
            print(f" - {path}")

        generated_files.update(sensitivity_files)
        return {
            "evaluation_results": results,
            "generated_files": generated_files,
            "execution_time_seconds": execution_time,
        }


def main():
    """Main entry point for experiment."""
    regenerate_pooled = "--regenerate-pooled" in sys.argv

    print("=" * 70)
    print("FUZZY OGD RETRIEVAL - EXPERIMENTAL EVALUATION")
    print("University of Fribourg, Human-IST Institute")
    print("=" * 70)
    
    # Initialize runner
    runner = ExperimentRunner()
    
    # Add systems to compare
    runner.add_system(PortalBaseline())
    runner.add_system(KeywordBaseline())
    runner.add_system(MetadataQualityRanker())
    runner.add_system(FuzzyHCIRRankerAdapter())
    runner.add_system(AISemanticBaselineAdapter())
    
    # Run complete pipeline including the optional sensitivity analysis stage
    runner.run_complete_pipeline(include_sensitivity_analysis=True, regenerate_pooled=regenerate_pooled)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
