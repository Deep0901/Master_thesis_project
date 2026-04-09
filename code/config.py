"""
Configuration settings for the OGD Fuzzy Retrieval System.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class CKANConfig:
    """CKAN API configuration."""
    base_url: str = "https://opendata.swiss/api/3"
    timeout: int = 30
    max_results: int = 1000
    cache_enabled: bool = True
    cache_ttl_hours: int = 24


@dataclass
class FuzzyConfig:
    """Fuzzy inference system configuration."""
    defuzzification_method: str = "centroid"
    resolution: int = 100
    recency_max_days: int = 730
    completeness_weights: dict = None
    
    def __post_init__(self):
        if self.completeness_weights is None:
            self.completeness_weights = {
                "title": 1.0,
                "description": 1.5,
                "tags": 1.0,
                "resources": 1.2,
                "organization": 0.8,
                "license": 0.8,
                "temporal": 1.0,
                "groups": 0.7
            }


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "mock"  # "mock", "openai"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    max_tokens: int = 150
    temperature: float = 0.3
    
    def __post_init__(self):
        # Try to load API key from environment
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    provider: str = "mock"  # "mock", "sentence-transformers"
    model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    dimension: int = 384
    use_faiss: bool = False


@dataclass
class UIConfig:
    """UI configuration."""
    page_title: str = "Swiss OGD Search - Fuzzy HCIR"
    page_icon: str = "🔍"
    results_per_page: int = 20
    show_explanations: bool = True
    enable_baseline_comparison: bool = True


@dataclass
class Config:
    """Main configuration container."""
    ckan: CKANConfig = None
    fuzzy: FuzzyConfig = None
    llm: LLMConfig = None
    embedding: EmbeddingConfig = None
    ui: UIConfig = None
    
    # Paths
    project_root: Path = None
    data_dir: Path = None
    cache_dir: Path = None
    logs_dir: Path = None
    
    # Feature flags
    debug_mode: bool = False
    enable_llm_normalization: bool = False
    enable_semantic_baseline: bool = True
    
    def __post_init__(self):
        # Initialize nested configs
        self.ckan = self.ckan or CKANConfig()
        self.fuzzy = self.fuzzy or FuzzyConfig()
        self.llm = self.llm or LLMConfig()
        self.embedding = self.embedding or EmbeddingConfig()
        self.ui = self.ui or UIConfig()
        
        # Set up paths
        if self.project_root is None:
            self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.cache_dir = self.project_root / ".cache"
        self.logs_dir = self.project_root / "logs"
        
        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)


# Global default configuration
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Get the default configuration."""
    return DEFAULT_CONFIG


def load_config_from_env() -> Config:
    """Load configuration from environment variables."""
    config = Config()
    
    # Override from environment
    if os.getenv("DEBUG_MODE"):
        config.debug_mode = os.getenv("DEBUG_MODE").lower() == "true"
    
    if os.getenv("ENABLE_LLM"):
        config.enable_llm_normalization = os.getenv("ENABLE_LLM").lower() == "true"
        config.llm.provider = "openai"
    
    if os.getenv("CKAN_BASE_URL"):
        config.ckan.base_url = os.getenv("CKAN_BASE_URL")
    
    if os.getenv("DEFUZZ_METHOD"):
        config.fuzzy.defuzzification_method = os.getenv("DEFUZZ_METHOD")
    
    return config


if __name__ == "__main__":
    # Print configuration
    config = get_config()
    print("=== OGD Fuzzy Retrieval Configuration ===")
    print(f"Project Root: {config.project_root}")
    print(f"CKAN URL: {config.ckan.base_url}")
    print(f"Defuzzification: {config.fuzzy.defuzzification_method}")
    print(f"LLM Provider: {config.llm.provider}")
    print(f"Debug Mode: {config.debug_mode}")
