"""
AI Semantic Baseline - Embedding-Based Search

Implements AI-driven semantic search using embeddings
for comparison with the interpretable fuzzy approach.

Research Context:
- Part of Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR"
- Addresses RQ4: Comparing fuzzy vs AI-driven semantic retrieval

This baseline uses:
- Sentence transformers for embedding generation
- Cosine similarity for semantic matching
- Optional re-ranking with cross-encoders
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class SemanticResult:
    """Result from semantic search."""
    dataset_id: str
    title: str
    similarity_score: float
    rank: int
    embedding_norm: float


@dataclass
class SemanticSearchResult:
    """Complete semantic search result."""
    query: str
    results: List[SemanticResult]
    total_matches: int
    processing_time_ms: float
    model_used: str


class EmbeddingProvider:
    """
    Abstract base class for embedding providers.
    
    Supports multiple embedding backends for flexibility.
    """
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of embeddings [n_texts, embedding_dim]
        """
        raise NotImplementedError
    
    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        raise NotImplementedError


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock embedding provider for testing without GPU.
    
    Generates deterministic pseudo-embeddings based on text features.
    """
    
    def __init__(self, dim: int = 384):
        """
        Initialize mock provider.
        
        Args:
            dim: Embedding dimension to simulate
        """
        self.dim = dim
        self._word_vectors: Dict[str, np.ndarray] = {}
    
    def _get_word_vector(self, word: str) -> np.ndarray:
        """Get or create a consistent vector for a word."""
        if word not in self._word_vectors:
            # Create deterministic pseudo-random vector based on word hash
            np.random.seed(hash(word) % (2**32))
            self._word_vectors[word] = np.random.randn(self.dim)
        return self._word_vectors[word]
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate pseudo-embeddings based on word averaging."""
        embeddings = []
        
        for text in texts:
            words = text.lower().split()
            if words:
                word_vecs = [self._get_word_vector(w) for w in words]
                embedding = np.mean(word_vecs, axis=0)
            else:
                embedding = np.zeros(self.dim)
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def get_model_name(self) -> str:
        return f"MockEmbedding-{self.dim}d"


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Real embedding provider using sentence-transformers.
    
    Recommended models for multilingual OGD retrieval:
    - paraphrase-multilingual-MiniLM-L12-v2 (fast, multilingual)
    - distiluse-base-multilingual-cased-v1 (multilingual)
    - all-MiniLM-L6-v2 (fast English)
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize sentence transformer.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence transformer."""
        return self.model.encode(texts, normalize_embeddings=True)
    
    def get_model_name(self) -> str:
        return self.model_name


class SemanticIndex:
    """
    Index for fast semantic similarity search.
    
    Stores embeddings and supports efficient nearest neighbor search.
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        use_faiss: bool = False
    ):
        """
        Initialize semantic index.
        
        Args:
            embedding_provider: Provider for generating embeddings
            use_faiss: Whether to use FAISS for fast search
        """
        self.provider = embedding_provider
        self.use_faiss = use_faiss
        
        self.embeddings: Optional[np.ndarray] = None
        self.dataset_ids: List[str] = []
        self.dataset_titles: List[str] = []
        self._faiss_index = None
    
    def index_datasets(self, datasets: List[Dict]):
        """
        Build index from datasets.
        
        Args:
            datasets: List of dataset metadata
        """
        # Extract text for embedding
        texts = []
        self.dataset_ids = []
        self.dataset_titles = []
        
        for ds in datasets:
            # Get title
            title = ds.get("title", "")
            if isinstance(title, dict):
                title = " ".join(str(v) for v in title.values() if v)
            
            # Get description
            desc = ds.get("description", "") or ds.get("notes", "")
            if isinstance(desc, dict):
                desc = " ".join(str(v) for v in desc.values() if v)
            
            # Get tags
            tags = ds.get("tags", [])
            if isinstance(tags, list):
                tag_text = " ".join(
                    t.get("name", t) if isinstance(t, dict) else str(t)
                    for t in tags
                )
            else:
                tag_text = ""
            
            # Combine with title weighting
            combined = f"{title} {title} {desc} {tag_text}"
            texts.append(combined)
            
            self.dataset_ids.append(ds.get("id") or ds.get("name", ""))
            self.dataset_titles.append(title[:100] if title else "Unknown")
        
        # Generate embeddings
        self.embeddings = self.provider.encode(texts)
        
        # Build FAISS index if requested
        if self.use_faiss and len(datasets) > 0:
            self._build_faiss_index()
    
    def _build_faiss_index(self):
        """Build FAISS index for fast search."""
        try:
            import faiss
            
            dim = self.embeddings.shape[1]
            
            # Use IVF index for larger collections
            if len(self.embeddings) > 1000:
                nlist = min(100, len(self.embeddings) // 10)
                quantizer = faiss.IndexFlatIP(dim)
                self._faiss_index = faiss.IndexIVFFlat(
                    quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
                )
                self._faiss_index.train(self.embeddings)
                self._faiss_index.add(self.embeddings)
            else:
                self._faiss_index = faiss.IndexFlatIP(dim)
                self._faiss_index.add(self.embeddings)
                
        except ImportError:
            self._faiss_index = None
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Search for similar datasets.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity) tuples
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        # Encode query
        query_embedding = self.provider.encode([query])[0]
        
        if self._faiss_index is not None:
            # Use FAISS search
            scores, indices = self._faiss_index.search(
                query_embedding.reshape(1, -1), top_k
            )
            return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        else:
            # Brute force cosine similarity
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [(idx, similarities[idx]) for idx in top_indices]


class AISemanticBaseline:
    """
    AI-driven semantic search baseline.
    
    Uses embedding similarity for ranking without explicit rules.
    This serves as the "black box" comparison to the fuzzy approach.
    """
    
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        use_faiss: bool = False
    ):
        """
        Initialize semantic baseline.
        
        Args:
            embedding_provider: Optional custom provider (defaults to mock)
            use_faiss: Whether to use FAISS for search
        """
        self.provider = embedding_provider or MockEmbeddingProvider()
        self.index = SemanticIndex(self.provider, use_faiss)
        self.datasets: List[Dict] = []
    
    def index_datasets(self, datasets: List[Dict]):
        """
        Index datasets for semantic search.
        
        Args:
            datasets: List of dataset metadata
        """
        self.datasets = datasets
        self.index.index_datasets(datasets)
    
    def search(self, query: str, top_n: int = 20) -> SemanticSearchResult:
        """
        Perform semantic search.
        
        Args:
            query: Search query
            top_n: Maximum results to return
            
        Returns:
            SemanticSearchResult with ranked datasets
        """
        start_time = time.time()
        
        # Search index
        search_results = self.index.search(query, top_n)
        
        # Build results
        results = []
        for rank, (idx, score) in enumerate(search_results, 1):
            if score > 0.01:  # Filter low scores
                results.append(SemanticResult(
                    dataset_id=self.index.dataset_ids[idx],
                    title=self.index.dataset_titles[idx],
                    similarity_score=float(score),
                    rank=rank,
                    embedding_norm=float(np.linalg.norm(self.index.embeddings[idx]))
                ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return SemanticSearchResult(
            query=query,
            results=results,
            total_matches=len(results),
            processing_time_ms=processing_time,
            model_used=self.provider.get_model_name()
        )
    
    def explain_missing(
        self,
        fuzzy_explanation: Dict = None
    ) -> str:
        """
        Return an explanation of why semantic search cannot explain.
        
        This highlights the explainability advantage of fuzzy ranking (RQ3).
        
        Args:
            fuzzy_explanation: Optional fuzzy explanation for comparison
            
        Returns:
            Statement about lack of explainability
        """
        explanation = (
            "This ranking was produced by an AI embedding model that converts "
            "text into high-dimensional vectors and measures similarity in that space.\n\n"
            "Unlike the fuzzy ranking system, this approach cannot explain:\n"
            "• Which specific features made this dataset relevant\n"
            "• How different quality factors contributed to the score\n"
            "• Why this dataset ranked higher than another\n\n"
            "The ranking is based on learned patterns in the embedding model, "
            "making it a 'black box' difficult to interpret or audit."
        )
        
        if fuzzy_explanation:
            explanation += f"\n\nFor comparison, the fuzzy system explained:\n{fuzzy_explanation}"
        
        return explanation


def create_semantic_baseline(
    use_real_model: bool = False,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    use_faiss: bool = False
) -> AISemanticBaseline:
    """
    Factory function to create semantic baseline.
    
    Args:
        use_real_model: Whether to use real sentence transformers
        model_name: Model name if using real model
        use_faiss: Whether to use FAISS for fast search
        
    Returns:
        Configured AISemanticBaseline
    """
    if use_real_model:
        provider = SentenceTransformerProvider(model_name)
    else:
        provider = MockEmbeddingProvider()
    
    return AISemanticBaseline(provider, use_faiss)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("AI SEMANTIC BASELINE DEMONSTRATION")
    print("=" * 60)
    
    # Mock datasets
    mock_datasets = [
        {
            "id": "ds1",
            "title": {"en": "Air Quality Measurements Zurich"},
            "description": "Daily measurements of air pollution including PM2.5, NO2",
            "tags": [{"name": "air quality"}, {"name": "pollution"}]
        },
        {
            "id": "ds2",
            "title": {"en": "Traffic Statistics 2023"},
            "description": "Annual traffic volume and vehicle counts on highways",
            "tags": [{"name": "traffic"}, {"name": "transport"}]
        },
        {
            "id": "ds3",
            "title": {"en": "Population Census Data"},
            "description": "Population demographics by canton and municipality",
            "tags": [{"name": "population"}, {"name": "census"}]
        }
    ]
    
    # Create baseline with mock provider
    baseline = create_semantic_baseline(use_real_model=False)
    baseline.index_datasets(mock_datasets)
    
    # Test searches
    queries = [
        "air pollution environment",
        "traffic mobility vehicle",
        "population demographics"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        result = baseline.search(query)
        print(f"Model: {result.model_used}")
        print(f"Processing time: {result.processing_time_ms:.1f}ms")
        
        for r in result.results:
            print(f"  {r.rank}. {r.title}")
            print(f"     Similarity: {r.similarity_score:.4f}")
    
    print("\n--- Explainability Comparison ---")
    print(baseline.explain_missing())
