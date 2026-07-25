from code.config import EmbeddingConfig as MainEmbeddingConfig
from code.prototype.config import EmbeddingConfig as PrototypeEmbeddingConfig


def test_embedding_defaults_use_sentence_transformers_provider():
    assert MainEmbeddingConfig().provider == "sentence-transformers"
    assert PrototypeEmbeddingConfig().provider == "sentence-transformers"
