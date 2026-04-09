"""
Query Processing Package

Handles user query parsing, normalization, and expansion
for the OGD search system.

Modules:
- query_parser: Basic query parsing and structure extraction
- llm_normalizer: LLM-based multilingual normalization
- synonym_expander: Query expansion with synonyms
- query_translator: Cross-lingual query handling
"""

from .query_parser import (
    QueryParser, ParsedQuery,
    QueryLanguage, TemporalModifier, QualityModifier,
    create_parser
)

from .llm_normalizer import (
    QueryNormalizer, NormalizationResult,
    LLMProvider, MockLLMProvider, OpenAIProvider,
    create_normalizer
)

__all__ = [
    # Parser
    'QueryParser', 'ParsedQuery',
    'QueryLanguage', 'TemporalModifier', 'QualityModifier',
    'create_parser',
    
    # Normalizer
    'QueryNormalizer', 'NormalizationResult',
    'LLMProvider', 'MockLLMProvider', 'OpenAIProvider',
    'create_normalizer'
]
