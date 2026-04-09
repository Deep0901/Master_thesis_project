"""
Ranking Package

Dataset ranking modules for the OGD search system.

Modules:
- fuzzy_ranker: Main fuzzy logic-based ranking
- baseline_keyword: Traditional keyword retrieval baseline
- ai_semantic_baseline: Embedding-based semantic search
- explanation_generator: Human-readable explanations
"""

from .fuzzy_ranker import (
    FuzzyRanker, RankedDataset, RankingResult,
    SimilarityCalculator, MetadataScorer,
    create_ranker
)

from .baseline_keyword import (
    BaselineKeywordRetrieval, BaselineResult, BaselineSearchResult,
    TFIDFCalculator,
    create_baseline
)

from .explanation_generator import (
    ExplanationGenerator, RankingExplanation, ExplanationComponent,
    create_explanation_generator
)

__all__ = [
    # Fuzzy Ranker
    'FuzzyRanker', 'RankedDataset', 'RankingResult',
    'SimilarityCalculator', 'MetadataScorer',
    'create_ranker',
    
    # Baseline
    'BaselineKeywordRetrieval', 'BaselineResult', 'BaselineSearchResult',
    'TFIDFCalculator',
    'create_baseline',
    
    # Explanations
    'ExplanationGenerator', 'RankingExplanation', 'ExplanationComponent',
    'create_explanation_generator'
]
