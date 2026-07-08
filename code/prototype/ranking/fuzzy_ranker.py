"""
Fuzzy Ranker - Main Ranking Module

Combines query processing, fuzzy inference, and metadata analysis
to produce explainable dataset rankings.

Research Context:
- Part of Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR"
- Addresses RQ1, RQ2, RQ3: Core ranking mechanism
"""

import re
import math
from collections import Counter, defaultdict
from math import log
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

from code.config import load_config_from_env
from code.query_processing import create_parser


@dataclass
class RankedDataset:
    """A dataset with its ranking information."""
    dataset_id: str
    title: str
    relevance_score: float
    explanation: str
    metadata_scores: Dict[str, float]
    rank: int
    
    def to_dict(self) -> Dict:
        return {
            "id": self.dataset_id,
            "title": self.title,
            "score": self.relevance_score,
            "rank": self.rank,
            "explanation": self.explanation,
            "metadata_scores": self.metadata_scores
        }


@dataclass
class RankingResult:
    """Complete result of a ranking operation."""
    query: str
    total_datasets: int
    ranked_datasets: List[RankedDataset]
    processing_time_ms: float
    explanation_summary: str
    
    def top_n(self, n: int = 10) -> List[RankedDataset]:
        """Get top N ranked datasets."""
        return self.ranked_datasets[:n]


class SimilarityCalculator:
    """
    Calculate query-document similarity using BM25.
    """

    def __init__(self, method: str = "bm25"):
        self.method = method
        self._document_count = 0
        self._document_frequencies: Dict[str, int] = {}
        self._average_document_length = 0.0
        self._k1 = 1.5
        self._b = 0.75

    @staticmethod
    def _flatten_text(value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, dict):
            return ' '.join(
                SimilarityCalculator._flatten_text(item)
                for item in value.values()
                if item is not None
            )
        if isinstance(value, (list, tuple, set)):
            return ' '.join(SimilarityCalculator._flatten_text(item) for item in value)
        return str(value)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

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
        title_text = self._flatten_text(dataset.get('title', {}))
        desc_text = self._flatten_text(dataset.get('description', {}))
        tags_text = self._flatten_text(dataset.get('tags', []))
        groups_text = self._flatten_text(dataset.get('groups', []))
        org_text = self._flatten_text(dataset.get('organization', {}))

        parts = [
            title_text,
            title_text,
            title_text,
            tags_text,
            tags_text,
            groups_text,
            groups_text,
        ]
        
        # Only include description if we have structured fields
        if title_text or tags_text or groups_text:
            # Include description with reduced weight (only once instead of full weight)
            if desc_text:
                parts.append(desc_text)
        else:
            # If no structured fields, rely on description and org
            if desc_text:
                parts.append(desc_text)
            if org_text:
                parts.append(org_text)

        return ' '.join(part for part in parts if part)

    def _query_term_weight(self, term: str) -> float:
        """Return priority weight for query term."""
        term_lower = term.lower()
        high_priority = {'bicycle', 'bike', 'biking', 'cycling', 'cycle', 'velo', 'vélo', 'fahrrad', 'bicicletta'}
        medium_priority = {'mobility', 'transport', 'verkehr', 'traffic', 'transit', 'transportation'}
        low_priority = {'data', 'dataset', 'datasets', 'related', 'show', 'statistics', 'statistic'}

        if term_lower in high_priority:
            return 1.6
        if term_lower in medium_priority:
            return 1.15
        if term_lower in low_priority:
            return 0.75
        return 1.0

    def _apply_theme_boost(self, score: float, dataset: Dict, query_themes: Optional[List[str]] = None, query_terms: Optional[List[str]] = None) -> float:
        title_text = self._flatten_text(dataset.get('title', '')).lower()
        tags_text = self._flatten_text(dataset.get('tags', [])).lower()
        groups_text = self._flatten_text(dataset.get('groups', [])).lower()
        searchable_text = ' '.join([title_text, tags_text, groups_text])

        if query_terms:
            high_priority = {'bicycle', 'bike', 'biking', 'cycling', 'cycle', 'velo', 'vélo', 'fahrrad', 'bicicletta'}
            medium_priority = {'mobility', 'transport', 'verkehr', 'traffic', 'transit', 'transportation'}
            low_priority = {'data', 'dataset', 'datasets', 'related', 'show', 'statistics', 'statistic'}

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
            'environment': ['environment', 'umwelt', 'environnement', 'ambiente', 'pollution', 'climate'],
            'mobility': ['mobility', 'transport', 'verkehr', 'traffic', 'road', 'rail', 'bicycle', 'bike', 'cycling', 'velo', 'vélo', 'fahrrad', 'transit', 'transportation'],
            'health': ['health', 'gesundheit', 'santé', 'salute', 'hospital'],
            'education': ['education', 'bildung', 'éducation', 'istruzione', 'school'],
            'economy': ['economy', 'wirtschaft', 'économie', 'economia', 'finance', 'employment'],
            'population': ['population', 'bevölkerung', 'demographic', 'demographie']
        }

        boost = 0.0
        for theme in query_themes:
            if any(term in searchable_text for term in theme_terms.get(theme, [theme])):
                boost += 0.12

        return min(score * (1.0 + min(boost, 0.30)), 1.0)

    def calculate(
        self,
        query_keywords: List[str],
        dataset: Dict,
        query_themes: Optional[List[str]] = None
    ) -> float:
        """
        Calculate similarity between query and dataset.
        
        Args:
            query_keywords: Extracted query keywords
            dataset: Dataset dictionary
            
        Returns:
            Similarity score (0-1)
        """
        if not query_keywords:
            return 0.5  # Neutral if no keywords

        doc_text = self._get_doc_text(dataset).lower()
        doc_tokens = self._tokenize(doc_text)

        if not doc_tokens:
            return 0.0

        query_tokens = self._tokenize(' '.join(query_keywords))
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
            high_priority = {'bicycle', 'bike', 'biking', 'cycling', 'cycle', 'velo', 'vélo', 'fahrrad', 'bicicletta'}
            medium_priority = {'mobility', 'transport', 'verkehr', 'traffic', 'transit', 'transportation'}
            low_priority = {'data', 'dataset', 'datasets', 'related', 'show', 'statistics', 'statistic'}
            
            if term_l in high_priority:
                query_boost *= 2.0  # Increased from 1.6
            elif term_l in medium_priority:
                query_boost *= 1.3  # Increased from 1.15
            elif term_l in low_priority:
                query_boost *= 0.5  # Increased penalty from 0.75
            
            score += idf * (numerator / denominator) * query_boost

        coverage = len(shared_terms) / float(len(set(query_tokens)))
        similarity = (score / (score + 1.0) if score > 0 else 0.0) * coverage

        return self._apply_theme_boost(similarity, dataset, query_themes=query_themes, query_terms=query_tokens)


class MetadataScorer:
    """
    Calculate crisp scores for metadata quality dimensions.
    """
    
    @staticmethod
    def calculate_recency(days_since_modified: Optional[int]) -> float:
        """Calculate recency score."""
        if days_since_modified is None:
            return 730
        return min(days_since_modified, 730)
    
    @staticmethod
    def calculate_completeness(metadata: Dict) -> float:
        """Calculate metadata completeness score."""
        checks = [
            bool(metadata.get("title")),
            bool(metadata.get("description")) and len(str(metadata.get("description", ""))) > 50,
            len(metadata.get("tags", [])) > 0,
            len(metadata.get("resources", [])) > 0,
            bool(metadata.get("organization")),
            bool(metadata.get("license_id")),
            bool(metadata.get("temporal_coverage")) or bool(metadata.get("temporalCoverage")),
            len(metadata.get("groups", [])) > 0
        ]
        
        return sum(checks) / len(checks)
    
    @staticmethod
    def calculate_resource_availability(metadata: Dict) -> float:
        """Calculate resource availability score."""
        resources = metadata.get("resources", [])
        return min(len(resources), 20)


class FuzzyRanker:
    """
    Main ranking system combining all components.
    """
    
    def __init__(self, defuzzification_method: str = "centroid"):
        """Initialize the fuzzy ranker."""
        self.similarity_calc = SimilarityCalculator()
        self.metadata_scorer = MetadataScorer()
        self._inference_engine = None
        self._defuzz_method = defuzzification_method
    
    @property
    def inference_engine(self):
        """Lazy load inference engine."""
        if self._inference_engine is None:
            from code.fuzzy_system import create_inference_engine
            self._inference_engine = create_inference_engine(self._defuzz_method)
        return self._inference_engine
    
    def rank_dataset(
        self,
        query_terms: List[str],
        dataset_metadata: Dict,
        query_themes: Optional[List[str]] = None
    ) -> Tuple[float, Dict[str, float], str]:
        """Rank a single dataset for a query."""
        thematic_sim = self.similarity_calc.calculate(query_terms, dataset_metadata, query_themes=query_themes)
        recency = self.metadata_scorer.calculate_recency(
            dataset_metadata.get("days_since_modified")
        )
        completeness = self.metadata_scorer.calculate_completeness(dataset_metadata)
        resource_avail = self.metadata_scorer.calculate_resource_availability(dataset_metadata)
        
        input_scores = {
            "thematic_similarity": thematic_sim,
            "recency": recency,
            "completeness": completeness,
            "resource_availability": resource_avail
        }
        
        # Run fuzzy inference
        try:
            result = self.inference_engine.infer(
                similarity=thematic_sim,
                recency=recency,
                completeness=completeness,
                resources=resource_avail
            )
            relevance = result.get("relevance", 0.5)
        except Exception as e:
            relevance = thematic_sim * 0.6 + (1 - min(recency, 730) / 730) * 0.2 + completeness * 0.2
        
        explanation = f"Relevance: {int(relevance * 100)}%"
        
        return relevance, input_scores, explanation
    
    def rank(
        self,
        query: str,
        candidate_datasets: List[Dict],
        query_themes: Optional[List[str]] = None
    ) -> RankingResult:
        """Rank candidate datasets."""
        start_time = time.time()
        
        # Fit similarity calculator
        self.similarity_calc.fit(candidate_datasets)
        
        # Parse query to extract keywords
        from code.query_processing import create_parser
        parser = create_parser()
        parsed_query = parser.parse(query)
        query_terms = parsed_query.keywords if parsed_query else []
        
        # Use detected themes if not provided
        if not query_themes and parsed_query:
            query_themes = parsed_query.themes
        
        # Rank datasets
        rankings = []
        for dataset in candidate_datasets:
            relevance, scores, explanation = self.rank_dataset(
                query_terms, dataset, query_themes=query_themes
            )
            rankings.append((dataset, relevance, scores, explanation))
        
        # Sort by relevance
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Create result
        ranked_datasets = []
        for rank, (dataset, relevance, scores, explanation) in enumerate(rankings, 1):
            ranked_datasets.append(
                RankedDataset(
                    dataset_id=dataset.get("id", "unknown"),
                    title=dataset.get("title", "Unknown"),
                    relevance_score=relevance,
                    explanation=explanation,
                    metadata_scores=scores,
                    rank=rank
                )
            )
        
        elapsed_ms = (time.time() - start_time) * 1000

        #start change
        # ✅ ADD THIS BLOCK HERE
        clean_results = []
        for r in ranked_datasets:
            if r and hasattr(r, "title") and r.title:
                clean_results.append(r)

        ranked_datasets = clean_results
        #end change

        return RankingResult(
            query=query,
            total_datasets=len(candidate_datasets),
            ranked_datasets=ranked_datasets,
            processing_time_ms=elapsed_ms,
            explanation_summary=f"Ranked {len(candidate_datasets)} datasets in {elapsed_ms:.1f}ms"
        )

    def rank_datasets(
        self,
        query: str,
        candidate_datasets: List[Dict],
        top_n: int = 10,
        query_themes: Optional[List[str]] = None
    ) -> RankingResult:
        """Rank datasets and return top N results."""
        result = self.rank(query, candidate_datasets, query_themes=query_themes)
        result.ranked_datasets = result.ranked_datasets[:top_n]
        return result


def create_ranker(defuzzification_method: str = "centroid") -> FuzzyRanker:
    """Factory function to create a FuzzyRanker instance."""
    return FuzzyRanker(defuzzification_method=defuzzification_method)
