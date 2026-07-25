import importlib


def test_sentence_transformers_dependency_stack_imports():
    sentence_transformers = importlib.import_module("sentence_transformers")
    transformers = importlib.import_module("transformers")

    assert sentence_transformers is not None
    assert transformers is not None
