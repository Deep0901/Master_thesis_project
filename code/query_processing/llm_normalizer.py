"""
LLM-based Query Normalization and Synonym Expansion

Uses Large Language Models for:
- Multilingual query normalization
- Synonym expansion
- Query intent classification

Research Context:
- Part of Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR"
- Addresses controlled LLM usage at query level (not for ranking)
- Supports RQ2: Multilingual search performance

Note: LLM is used ONLY for query preprocessing, maintaining
fuzzy logic as the core ranking mechanism for explainability.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class NormalizationResult:
    """Result of LLM-based query normalization."""
    original_query: str
    normalized_query: str
    detected_language: str
    english_translation: str
    synonyms: List[str]
    related_terms: List[str]
    confidence: float
    processing_time_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "original": self.original_query,
            "normalized": self.normalized_query,
            "language": self.detected_language,
            "english": self.english_translation,
            "synonyms": self.synonyms,
            "related_terms": self.related_terms,
            "confidence": self.confidence,
            "time_ms": self.processing_time_ms
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for a prompt."""
        pass


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing without API calls.
    
    Uses rule-based fallbacks for basic functionality.
    """
    
    # Basic translation dictionary
    TRANSLATIONS = {
        "luftqualität": "air quality",
        "verkehr": "transport",
        "umwelt": "environment",
        "gesundheit": "health",
        "bevölkerung": "population",
        "qualité de l'air": "air quality",
        "transports": "transport",
        "environnement": "environment",
        "santé": "health",
        "qualità dell'aria": "air quality",
        "trasporti": "transport",
        "ambiente": "environment",
        "salute": "health"
    }
    
    # Basic synonym dictionary
    SYNONYMS = {
        "air quality": ["pollution", "emissions", "air pollution", "atmosphere"],
        "transport": ["traffic", "mobility", "vehicles", "roads", "transportation"],
        "environment": ["ecology", "nature", "climate", "sustainability"],
        "health": ["medical", "healthcare", "hospitals", "disease"],
        "population": ["demographics", "inhabitants", "residents", "census"]
    }
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate mock completion based on prompt analysis."""
        # This is a simplified mock - real implementation would call LLM API
        return json.dumps({
            "normalized": prompt.lower().strip(),
            "language": "en",
            "english": prompt,
            "synonyms": [],
            "related_terms": [],
            "confidence": 0.8
        })
    
    def translate_to_english(self, text: str) -> str:
        """Basic translation lookup."""
        text_lower = text.lower()
        for foreign, english in self.TRANSLATIONS.items():
            if foreign in text_lower:
                text_lower = text_lower.replace(foreign, english)
        return text_lower
    
    def get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a term."""
        term_lower = term.lower()
        for key, syns in self.SYNONYMS.items():
            if term_lower in key or key in term_lower:
                return syns
        return []


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider for query normalization.
    
    Requires OPENAI_API_KEY environment variable.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                self.client = None
        else:
            self.client = None
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Check API key and installation.")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for normalizing search queries about Swiss government data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=kwargs.get("max_tokens", 500),
            temperature=kwargs.get("temperature", 0.3)
        )
        
        return response.choices[0].message.content


class QueryNormalizer:
    """
    LLM-based query normalizer for OGD search.
    
    Handles:
    - Multilingual query normalization
    - Translation to English
    - Synonym expansion
    - Related term identification
    """
    
    NORMALIZATION_PROMPT = """
    Analyze and normalize the following search query for a Swiss Open Government Data portal.
    
    Query: "{query}"
    
    Respond in JSON format with these fields:
    {{
        "normalized": "cleaned query in original language",
        "language": "detected language code (de/fr/it/en)",
        "english": "English translation of the query",
        "synonyms": ["list of synonyms for main terms"],
        "related_terms": ["list of related search terms"],
        "confidence": 0.0-1.0
    }}
    
    Focus on:
    - Correcting spelling errors
    - Expanding abbreviations (e.g., BFS -> Bundesamt für Statistik)
    - Identifying domain-specific terminology
    - Providing relevant synonyms for better retrieval
    
    Respond ONLY with the JSON object.
    """
    
    def __init__(self, provider: Optional[LLMProvider] = None):
        """
        Initialize the normalizer.
        
        Args:
            provider: LLM provider (uses MockLLMProvider if None)
        """
        self.provider = provider or MockLLMProvider()
        self._cache: Dict[str, NormalizationResult] = {}
    
    def normalize(self, query: str, use_cache: bool = True) -> NormalizationResult:
        """
        Normalize a search query using LLM.
        
        Args:
            query: Raw user query
            use_cache: Whether to use cached results
            
        Returns:
            NormalizationResult with normalized query and expansions
        """
        # Check cache
        cache_key = query.lower().strip()
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        start_time = time.time()
        
        # For mock provider, use simple normalization
        if isinstance(self.provider, MockLLMProvider):
            result = self._normalize_fallback(query)
        else:
            result = self._normalize_with_llm(query)
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        # Cache result
        if use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def _normalize_with_llm(self, query: str) -> NormalizationResult:
        """Normalize using LLM provider."""
        prompt = self.NORMALIZATION_PROMPT.format(query=query)
        
        try:
            response = self.provider.complete(prompt)
            data = json.loads(response)
            
            return NormalizationResult(
                original_query=query,
                normalized_query=data.get("normalized", query),
                detected_language=data.get("language", "unknown"),
                english_translation=data.get("english", query),
                synonyms=data.get("synonyms", []),
                related_terms=data.get("related_terms", []),
                confidence=data.get("confidence", 0.5),
                processing_time_ms=0
            )
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback on error
            return self._normalize_fallback(query)
    
    def _normalize_fallback(self, query: str) -> NormalizationResult:
        """Simple rule-based fallback normalization."""
        mock = MockLLMProvider()
        
        # Detect language (simple heuristic)
        language = "en"
        if any(c in query.lower() for c in ['ä', 'ö', 'ü', 'ß']):
            language = "de"
        elif any(c in query.lower() for c in ['é', 'è', 'ê', 'à', 'ç']):
            language = "fr"
        elif any(c in query.lower() for c in ['à', 'è', 'ì', 'ò', 'ù']):
            language = "it"
        
        # Translate
        english = mock.translate_to_english(query) if language != "en" else query
        
        # Get synonyms
        synonyms = []
        for word in english.split():
            synonyms.extend(mock.get_synonyms(word))
        synonyms = list(set(synonyms))[:10]
        
        return NormalizationResult(
            original_query=query,
            normalized_query=query.lower().strip(),
            detected_language=language,
            english_translation=english,
            synonyms=synonyms,
            related_terms=[],
            confidence=0.6,
            processing_time_ms=0
        )
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query with synonyms and related terms.
        
        Returns multiple query variants for broader search.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded query variants
        """
        result = self.normalize(query)
        
        expanded = [
            result.normalized_query,
            result.english_translation
        ]
        
        # Add synonym combinations
        for syn in result.synonyms[:5]:
            expanded.append(f"{result.normalized_query} {syn}")
        
        return list(set(expanded))


def create_normalizer(use_openai: bool = False) -> QueryNormalizer:
    """
    Factory function to create a query normalizer.
    
    Args:
        use_openai: Whether to use OpenAI API (requires API key)
        
    Returns:
        Configured QueryNormalizer
    """
    if use_openai and os.getenv("OPENAI_API_KEY"):
        provider = OpenAIProvider()
    else:
        provider = MockLLMProvider()
    
    return QueryNormalizer(provider)


if __name__ == "__main__":
    # Demo: Normalize sample queries
    normalizer = create_normalizer(use_openai=False)
    
    test_queries = [
        "aktuelle Luftqualitätsdaten",
        "données environnement Genève",
        "trasporto pubblico statistiche",
        "recent air pollution data Switzerland",
        "BFS Bevölkerungsstatistik"
    ]
    
    print("=" * 60)
    print("LLM QUERY NORMALIZER DEMONSTRATION")
    print("=" * 60)
    
    for query in test_queries:
        result = normalizer.normalize(query)
        print(f"\nOriginal: '{query}'")
        print(f"  Normalized: '{result.normalized_query}'")
        print(f"  Language: {result.detected_language}")
        print(f"  English: '{result.english_translation}'")
        print(f"  Synonyms: {result.synonyms[:5]}")
        print(f"  Confidence: {result.confidence:.2f}")
