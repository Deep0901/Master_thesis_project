"""
Fuzzy Ranker - Main Ranking Module

Combines query processing, fuzzy inference, and metadata analysis
to produce explainable dataset rankings.

Research Context:
- Part of Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR"
- Addresses RQ1, RQ2, RQ3: Core ranking mechanism
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

# Note: Adjust imports based on your actual project structure
# from ..fuzzy_system import create_inference_engine, InferenceResult
# from ..query_processing import create_parser, create_normalizer
# from ..data_collection import MetadataExtractor


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
    Calculate thematic similarity between queries and datasets.
    
    Uses TF-IDF or semantic similarity for matching.
    """
    
    def __init__(self, method: str = "tfidf"):
        """
        Initialize similarity calculator.
        
        Args:
            method: Similarity method ("tfidf", "jaccard", "semantic")
        """
        self.method = method
        self._vectorizer = None
    
    def calculate(
        self,
        query_terms: List[str],
        dataset_metadata: Dict
    ) -> float:
        """
        Calculate similarity between query and dataset.
        
        Args:
            query_terms: Extracted query keywords
            dataset_metadata: Dataset metadata dictionary
            
        Returns:
            Similarity score [0, 1]
        """
        if self.method == "jaccard":
            return self._jaccard_similarity(query_terms, dataset_metadata)
        elif self.method == "tfidf":
            return self._tfidf_similarity(query_terms, dataset_metadata)
        else:
            return self._jaccard_similarity(query_terms, dataset_metadata)
    
    def _jaccard_similarity(
        self,
        query_terms: List[str],
        dataset_metadata: Dict
    ) -> float:
        """Simple Jaccard similarity."""
        # Collect dataset terms
        dataset_text = ' '.join([
            str(dataset_metadata.get("title", "")),
            str(dataset_metadata.get("description", "")),
            ' '.join(dataset_metadata.get("tags", [])),
            ' '.join(dataset_metadata.get("groups", []))
        ]).lower()
        
        dataset_terms = set(dataset_text.split())
        query_set = set(t.lower() for t in query_terms)
        
        if not query_set or not dataset_terms:
            return 0.0
        
        intersection = query_set.intersection(dataset_terms)
        union = query_set.union(dataset_terms)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _tfidf_similarity(
        self,
        query_terms: List[str],
        dataset_metadata: Dict
    ) -> float:
        """TF-IDF based similarity (simplified)."""
        # For a production system, this would use sklearn TF-IDF
        # Here we use an enhanced keyword matching approach
        
        dataset_text = ' '.join([
            str(dataset_metadata.get("title", "")) * 3,  # Title weighted higher
            str(dataset_metadata.get("description", "")),
            ' '.join(dataset_metadata.get("tags", [])) * 2,  # Tags weighted
        ]).lower()
        
        if not query_terms:
            return 0.0
        
        matches = sum(1 for term in query_terms if term.lower() in dataset_text)
        partial_matches = sum(
            0.5 for term in query_terms
            if any(term.lower() in word for word in dataset_text.split())
        )
        
        total_score = matches + partial_matches
        max_score = len(query_terms)
        
        return min(total_score / max_score, 1.0) if max_score > 0 else 0.0


class MetadataScorer:
    """
    Calculate crisp scores for metadata quality dimensions.
    
    These scores are fed into the fuzzy inference system.
    """
    
    @staticmethod
    def calculate_recency(days_since_modified: Optional[int]) -> float:
        """
        Calculate recency score.
        
        Args:
            days_since_modified: Days since last metadata update
            
        Returns:
            Days value for fuzzy system (lower = more recent)
        """
        if days_since_modified is None:
            return 730  # Default to 2 years if unknown
        return min(days_since_modified, 730)
    
    @staticmethod
    def calculate_completeness(metadata: Dict) -> float:
        """
        Calculate metadata completeness score.
        
        Args:
            metadata: Dataset metadata dictionary
            
        Returns:
            Completeness ratio [0, 1]
        """
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
        """
        Calculate resource availability score.
        
        Args:
            metadata: Dataset metadata dictionary
            
        Returns:
            Number of resources (capped at 20)
        """
        resources = metadata.get("resources", [])
        return min(len(resources), 20)


class FuzzyRanker:
    """
    Main ranking system combining all components.
    
    Pipeline:
    1. Parse and normalize query
    2. Retrieve candidate datasets
    3. Calculate metadata scores
    4. Apply fuzzy inference
    5. Generate explanations
    6. Rank and return results
    """
    
    def __init__(
        self,
        defuzzification_method: str = "centroid"
    ):
        """
        Initialize the fuzzy ranker.
        
        Args:
            defuzzification_method: Method for defuzzification
        """
        self.similarity_calc = SimilarityCalculator(method="tfidf")
        self.metadata_scorer = MetadataScorer()
        
        # Lazy load fuzzy system to avoid circular imports
        self._inference_engine = None
        self._defuzz_method = defuzzification_method
    
    @property
    def inference_engine(self):
        """Lazy load inference engine."""
        if self._inference_engine is None:
            # Import here to avoid circular dependency
            from code.fuzzy_system import create_inference_engine
            self._inference_engine = create_inference_engine(self._defuzz_method)
        return self._inference_engine
    
    def rank_dataset(
        self,
        query_terms: List[str],
        dataset_metadata: Dict
    ) -> Tuple[float, Dict[str, float], str]:
        """
        Rank a single dataset for a query.
        
        Args:
            query_terms: Extracted query keywords
            dataset_metadata: Dataset metadata
            
        Returns:
            Tuple of (relevance_score, input_scores, explanation)
        """
        # Calculate input scores
        thematic_sim = self.similarity_calc.calculate(query_terms, dataset_metadata)
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
            result = self.inference_engine.infer(input_scores)
            relevance_score = result.crisp_output
            explanation = result.get_explanation(top_n=2)
        except Exception as e:
            # Fallback scoring if fuzzy system fails
            relevance_score = thematic_sim * 100
            explanation = f"Fallback scoring: similarity={thematic_sim:.2f}"
        
        return relevance_score, input_scores, explanation
    
    def rank_datasets(
        self,
        query: str,
        datasets: List[Dict],
        top_n: int = 20
    ) -> RankingResult:
        """
        Rank multiple datasets for a query.
        
        Args:
            query: User search query
            datasets: List of dataset metadata dictionaries
            top_n: Maximum results to return
            
        Returns:
            RankingResult with ranked datasets
        """
        start_time = time.time()
        
        # Parse query
        from code.query_processing import create_parser
        parser = create_parser()
        parsed = parser.parse(query)
        query_terms = parsed.keywords
        
        # Rank all datasets
        scored_datasets = []
        
        for dataset in datasets:
            score, input_scores, explanation = self.rank_dataset(
                query_terms, dataset
            )
            
            title = dataset.get("title", "")
            if isinstance(title, dict):
                title = title.get("en") or title.get("de") or str(list(title.values())[0] if title else "")
            
            scored_datasets.append({
                "id": dataset.get("id") or dataset.get("name", ""),
                "title": title,
                "score": score,
                "input_scores": input_scores,
                "explanation": explanation
            })
        
        # Sort by score (descending)
        scored_datasets.sort(key=lambda x: x["score"], reverse=True)
        
        # Create ranked results
        ranked = []
        for rank, ds in enumerate(scored_datasets[:top_n], 1):
            ranked.append(RankedDataset(
                dataset_id=ds["id"],
                title=ds["title"][:100] if ds["title"] else "Unknown",
                relevance_score=ds["score"],
                explanation=ds["explanation"],
                metadata_scores=ds["input_scores"],
                rank=rank
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        # Generate summary
        if ranked:
            summary = (
                f"Query: '{query}'\n"
                f"Found {len(scored_datasets)} datasets, showing top {len(ranked)}\n"
                f"Top result: '{ranked[0].title}' (score: {ranked[0].relevance_score:.1f})"
            )
        else:
            summary = f"Query: '{query}'\nNo datasets found."
        
        return RankingResult(
            query=query,
            total_datasets=len(datasets),
            ranked_datasets=ranked,
            processing_time_ms=processing_time,
            explanation_summary=summary
        )


def create_ranker(defuzzification: str = "centroid") -> FuzzyRanker:
    """Create a configured FuzzyRanker instance."""
    return FuzzyRanker(defuzzification_method=defuzzification)


if __name__ == "__main__":
    # Demo with mock data
    print("=" * 60)
    print("FUZZY RANKER DEMONSTRATION")
    print("=" * 60)
    
    # Create mock datasets
    mock_datasets = [
        {
            "id": "ds1",
            "name": "air-quality-zurich-2024",
            "title": {"en": "Air Quality Measurements Zurich 2024", "de": "Luftqualitätsmessungen Zürich 2024"},
            "description": "Daily air quality measurements including PM2.5, NO2, and O3 levels",
            "tags": ["air quality", "pollution", "environment", "zurich"],
            "groups": ["environment"],
            "resources": [{"format": "CSV"}, {"format": "JSON"}, {"format": "API"}],
            "organization": "City of Zurich",
            "days_since_modified": 5,
            "license_id": "cc-by"
        },
        {
            "id": "ds2",
            "name": "traffic-volume-2020",
            "title": {"en": "Traffic Volume Statistics 2020", "de": "Verkehrsaufkommen Statistik 2020"},
            "description": "Annual traffic volume data for Swiss highways",
            "tags": ["traffic", "transport", "mobility"],
            "groups": ["mobility"],
            "resources": [{"format": "PDF"}],
            "organization": "ASTRA",
            "days_since_modified": 600,
            "license_id": "cc-by"
        },
        {
            "id": "ds3",
            "name": "population-2023",
            "title": {"en": "Population Statistics 2023", "de": "Bevölkerungsstatistik 2023"},
            "description": "Detailed population data by canton and municipality",
            "tags": ["population", "demographics", "census"],
            "groups": ["population"],
            "resources": [{"format": "CSV"}, {"format": "XLSX"}],
            "organization": "BFS",
            "days_since_modified": 90,
            "license_id": "cc-by"
        }
    ]
    
    ranker = FuzzyRanker()
    ranker._inference_engine = None  # Force simple scoring for demo
    
    # Test query
    query = "air quality pollution Zurich recent"
    
    # Simple scoring without fuzzy system (for demo)
    print(f"\nQuery: '{query}'")
    print("\nRanking datasets...")
    
    from code.query_processing import create_parser
    parser = create_parser()
    parsed = parser.parse(query)
    
    print(f"Extracted keywords: {parsed.keywords}")
    print(f"Temporal modifier: {parsed.temporal_modifier.value}")
    print(f"Detected themes: {parsed.themes}")
    
    print("\n--- Dataset Scores ---")
    for ds in mock_datasets:
        sim = ranker.similarity_calc.calculate(parsed.keywords, ds)
        completeness = ranker.metadata_scorer.calculate_completeness(ds)
        recency = ranker.metadata_scorer.calculate_recency(ds.get("days_since_modified"))
        
        title = ds["title"]["en"] if isinstance(ds["title"], dict) else ds["title"]
        print(f"\n{title}:")
        print(f"  Similarity: {sim:.2f}")
        print(f"  Completeness: {completeness:.2f}")
        print(f"  Recency: {recency} days")
