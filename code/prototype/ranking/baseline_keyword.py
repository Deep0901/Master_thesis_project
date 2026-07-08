"""
Baseline Keyword Retrieval System

Implements a traditional keyword-based search for comparison
with the fuzzy ranking system.

Research Context:
- Part of Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR"
- Addresses RQ2 & RQ4: Baseline comparison system

This baseline uses:
- TF-IDF weighting
- Boolean keyword matching
- Simple relevance scoring
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
from collections import Counter
import time


@dataclass
class BaselineResult:
    """Result from baseline keyword search."""
    dataset_id: str
    title: str
    score: float
    matched_terms: List[str]
    rank: int


@dataclass
class BaselineSearchResult:
    """Complete baseline search result."""
    query: str
    results: List[BaselineResult]
    total_matches: int
    processing_time_ms: float


class TFIDFCalculator:
    """Calculate TF-IDF scores for document ranking."""
    
    def __init__(self, documents: List[Dict] = None):
        """
        Initialize with optional document collection for IDF calculation.
        
        Args:
            documents: List of dataset metadata for building IDF
        """
        self.idf_scores: Dict[str, float] = {}
        self.total_docs = 0
        
        if documents:
            self.fit(documents)
    
    def fit(self, documents: List[Dict]):
        """
        Build IDF scores from document collection.
        
        Args:
            documents: List of dataset metadata dictionaries
        """
        self.total_docs = len(documents)
        term_doc_count: Dict[str, int] = Counter()
        
        for doc in documents:
            # Get all text from document
            text = self._get_document_text(doc)
            terms = set(self._tokenize(text))
            
            for term in terms:
                term_doc_count[term] += 1
        
        # Calculate IDF
        for term, doc_count in term_doc_count.items():
            self.idf_scores[term] = math.log(self.total_docs / (doc_count + 1)) + 1
    
    def _get_document_text(self, doc: Dict) -> str:
        """Extract searchable text from document."""
        parts = []
        
        # Title
        title = doc.get("title", "")
        if isinstance(title, dict):
            parts.extend(title.values())
        else:
            parts.append(str(title))
        
        # Description
        desc = doc.get("description", "") or doc.get("notes", "")
        if isinstance(desc, dict):
            parts.extend(desc.values())
        else:
            parts.append(str(desc))
        
        # Tags/keywords
        tags = doc.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, dict):
                    parts.append(tag.get("name", ""))
                else:
                    parts.append(str(tag))
        
        # Groups
        groups = doc.get("groups", [])
        if isinstance(groups, list):
            for group in groups:
                if isinstance(group, dict):
                    parts.append(group.get("name", ""))
                else:
                    parts.append(str(group))
        
        return " ".join(filter(None, parts))
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w for w in text.split() if len(w) > 2]
    
    def calculate_tf(self, text: str) -> Dict[str, float]:
        """
        Calculate term frequency.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of term frequencies
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        
        tf = Counter(tokens)
        max_freq = max(tf.values())
        
        # Normalize by max frequency
        return {term: count / max_freq for term, count in tf.items()}
    
    def get_idf(self, term: str) -> float:
        """Get IDF score for a term."""
        return self.idf_scores.get(term.lower(), 1.0)
    
    def calculate_tfidf(self, text: str) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for document.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of TF-IDF scores
        """
        tf = self.calculate_tf(text)
        return {term: freq * self.get_idf(term) for term, freq in tf.items()}


class BaselineKeywordRetrieval:
    """
    Traditional keyword-based retrieval system.
    
    This serves as the baseline for comparison with fuzzy ranking.
    Uses TF-IDF weighting and simple boolean matching.
    """
    
    def __init__(self, datasets: List[Dict] = None):
        """
        Initialize the baseline system.
        
        Args:
            datasets: Optional dataset collection for indexing
        """
        self.datasets: List[Dict] = []
        self.tfidf = TFIDFCalculator()
        self._index: Dict[str, List[int]] = {}  # term -> [doc_indices]
        
        if datasets:
            self.index_datasets(datasets)
    
    def index_datasets(self, datasets: List[Dict]):
        """
        Build search index from datasets.
        
        Args:
            datasets: List of dataset metadata
        """
        self.datasets = datasets
        self.tfidf.fit(datasets)
        
        # Build inverted index
        self._index = {}
        for idx, doc in enumerate(datasets):
            text = self.tfidf._get_document_text(doc)
            terms = set(self.tfidf._tokenize(text))
            
            for term in terms:
                if term not in self._index:
                    self._index[term] = []
                self._index[term].append(idx)
    
    def search(
        self,
        query: str,
        top_n: int = 20,
        method: str = "tfidf"
    ) -> BaselineSearchResult:
        """
        Search for datasets matching query.
        
        Args:
            query: Search query string
            top_n: Maximum results to return
            method: Scoring method ("tfidf", "boolean", "bm25")
            
        Returns:
            BaselineSearchResult with ranked datasets
        """
        start_time = time.time()
        
        # Tokenize query
        query_terms = self.tfidf._tokenize(query)
        
        if not query_terms:
            return BaselineSearchResult(
                query=query,
                results=[],
                total_matches=0,
                processing_time_ms=0
            )
        
        # Find candidate documents
        candidate_indices = set()
        for term in query_terms:
            if term in self._index:
                candidate_indices.update(self._index[term])
        
        # Score candidates
        scored = []
        for idx in candidate_indices:
            doc = self.datasets[idx]
            
            if method == "boolean":
                score, matched = self._boolean_score(query_terms, doc)
            elif method == "bm25":
                score, matched = self._bm25_score(query_terms, doc)
            else:  # tfidf
                score, matched = self._tfidf_score(query_terms, doc)
            
            if score > 0:
                title = doc.get("title", "")
                if isinstance(title, dict):
                    title = title.get("en") or title.get("de") or str(list(title.values())[0] if title else "")
                
                scored.append({
                    "idx": idx,
                    "id": doc.get("id") or doc.get("name", ""),
                    "title": title,
                    "score": score,
                    "matched": matched
                })
        
        # Sort by score
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        # Build results
        results = []
        for rank, item in enumerate(scored[:top_n], 1):
            results.append(BaselineResult(
                dataset_id=item["id"],
                title=item["title"][:100] if item["title"] else "Unknown",
                score=item["score"],
                matched_terms=item["matched"],
                rank=rank
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return BaselineSearchResult(
            query=query,
            results=results,
            total_matches=len(scored),
            processing_time_ms=processing_time
        )
    
    def _boolean_score(
        self,
        query_terms: List[str],
        doc: Dict
    ) -> Tuple[float, List[str]]:
        """
        Simple boolean scoring.
        
        Returns the proportion of query terms matched.
        """
        doc_text = self.tfidf._get_document_text(doc).lower()
        matched = [t for t in query_terms if t in doc_text]
        
        score = len(matched) / len(query_terms) if query_terms else 0
        return score, matched
    
    def _tfidf_score(
        self,
        query_terms: List[str],
        doc: Dict
    ) -> Tuple[float, List[str]]:
        """
        TF-IDF based scoring.
        """
        doc_text = self.tfidf._get_document_text(doc)
        doc_tfidf = self.tfidf.calculate_tfidf(doc_text)
        
        score = 0
        matched = []
        
        for term in query_terms:
            term_lower = term.lower()
            if term_lower in doc_tfidf:
                score += doc_tfidf[term_lower] * self.tfidf.get_idf(term_lower)
                matched.append(term)
        
        # Normalize by query length
        if query_terms:
            score /= len(query_terms)
        
        return score, matched
    
    def _bm25_score(
        self,
        query_terms: List[str],
        doc: Dict,
        k1: float = 1.5,
        b: float = 0.75
    ) -> Tuple[float, List[str]]:
        """
        BM25 scoring (simplified).
        """
        doc_text = self.tfidf._get_document_text(doc)
        doc_tokens = self.tfidf._tokenize(doc_text)
        doc_len = len(doc_tokens)
        
        # Average document length
        avg_len = sum(
            len(self.tfidf._tokenize(self.tfidf._get_document_text(d)))
            for d in self.datasets
        ) / len(self.datasets) if self.datasets else doc_len
        
        term_counts = Counter(doc_tokens)
        
        score = 0
        matched = []
        
        for term in query_terms:
            term_lower = term.lower()
            tf = term_counts.get(term_lower, 0)
            
            if tf > 0:
                idf = self.tfidf.get_idf(term_lower)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avg_len)
                score += idf * numerator / denominator
                matched.append(term)
        
        return score, matched


def create_baseline(datasets: List[Dict] = None) -> BaselineKeywordRetrieval:
    """Factory function to create baseline system."""
    return BaselineKeywordRetrieval(datasets)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("BASELINE KEYWORD RETRIEVAL DEMONSTRATION")
    print("=" * 60)
    
    # Mock datasets
    mock_datasets = [
        {
            "id": "ds1",
            "title": {"en": "Air Quality Measurements Zurich"},
            "description": "Daily measurements of air pollution including PM2.5, NO2",
            "tags": [{"name": "air quality"}, {"name": "pollution"}],
            "groups": [{"name": "environment"}]
        },
        {
            "id": "ds2",
            "title": {"en": "Traffic Statistics 2023"},
            "description": "Annual traffic volume and vehicle counts",
            "tags": [{"name": "traffic"}, {"name": "transport"}],
            "groups": [{"name": "mobility"}]
        },
        {
            "id": "ds3",
            "title": {"en": "Population Census Data"},
            "description": "Population demographics by canton",
            "tags": [{"name": "population"}, {"name": "census"}],
            "groups": [{"name": "statistics"}]
        }
    ]
    
    baseline = create_baseline(mock_datasets)
    
    # Test search
    queries = [
        "air pollution zurich",
        "traffic statistics",
        "population data"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        result = baseline.search(query)
        print(f"Found {result.total_matches} matches in {result.processing_time_ms:.1f}ms")
        
        for r in result.results[:3]:
            print(f"  {r.rank}. {r.title} (score: {r.score:.3f})")
            print(f"     Matched: {r.matched_terms}")
