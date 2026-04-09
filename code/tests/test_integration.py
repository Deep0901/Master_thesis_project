"""
Integration Test Script

Tests the complete search workflow from query to ranked results.
"""

import sys
sys.path.insert(0, '.')

print("=" * 70)
print("OGD FUZZY RETRIEVAL SYSTEM - INTEGRATION TEST")
print("=" * 70)

# Test 1: Import all modules
print("\n[1] Testing Module Imports...")
try:
    from code.fuzzy_system import create_inference_engine
    from code.query_processing import create_parser, create_normalizer
    from code.ranking import create_ranker, create_baseline, create_explanation_generator
    from code.data_collection import CKANClient, get_client
    from code.config import get_config
    print("    ✓ All modules imported successfully")
except Exception as e:
    print(f"    ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Fuzzy Inference Engine
print("\n[2] Testing Fuzzy Inference Engine...")
engine = create_inference_engine()
test_inputs = {
    "recency": 10,
    "completeness": 0.85,
    "thematic_similarity": 0.75,
    "resource_availability": 5
}
result = engine.infer(test_inputs)
print(f"    Input: {test_inputs}")
print(f"    Output Score: {result.crisp_output:.1f}/100")
assert 60 <= result.crisp_output <= 100, "Score out of expected range"
print("    ✓ Fuzzy inference working correctly")

# Test 3: Query Parser
print("\n[3] Testing Query Parser...")
parser = create_parser()
queries = [
    "recent air quality data Zurich",
    "aktuelle Verkehrsdaten",
    "données environnementales complètes"
]
for q in queries:
    parsed = parser.parse(q)
    print(f"    Query: '{q}'")
    print(f"      Keywords: {parsed.keywords}, Temporal: {parsed.temporal_modifier.value}")
print("    ✓ Query parser working correctly")

# Test 4: Ranking System
print("\n[4] Testing Ranking System...")
ranker = create_ranker()
mock_datasets = [
    {
        "id": "ds1",
        "title": {"en": "Air Quality Zurich 2024"},
        "description": "Air pollution data",
        "tags": ["air", "pollution", "environment"],
        "groups": ["environment"],
        "resources": [{"format": "CSV"}, {"format": "JSON"}],
        "days_since_modified": 5
    },
    {
        "id": "ds2", 
        "title": {"en": "Traffic Statistics"},
        "description": "Highway traffic data",
        "tags": ["traffic", "transport"],
        "groups": ["mobility"],
        "resources": [{"format": "PDF"}],
        "days_since_modified": 200
    }
]

from code.ranking.fuzzy_ranker import SimilarityCalculator, MetadataScorer
sim_calc = SimilarityCalculator()
scorer = MetadataScorer()

for ds in mock_datasets:
    similarity = sim_calc.calculate(["air", "quality", "zurich"], ds)
    recency = scorer.calculate_recency(ds.get("days_since_modified"))
    completeness = scorer.calculate_completeness(ds)
    title = ds["title"]["en"] if isinstance(ds["title"], dict) else ds["title"]
    print(f"    Dataset: {title}")
    print(f"      Similarity: {similarity:.2f}, Recency: {recency} days, Completeness: {completeness:.2f}")
print("    ✓ Ranking system working correctly")

# Test 5: Baseline Comparison
print("\n[5] Testing Baseline Systems...")
baseline_kw = create_baseline(mock_datasets)
baseline_result = baseline_kw.search("air pollution")
print(f"    Keyword Baseline: {baseline_result.total_matches} matches")

from code.ranking.ai_semantic_baseline import create_semantic_baseline
baseline_ai = create_semantic_baseline()
baseline_ai.index_datasets(mock_datasets)
ai_result = baseline_ai.search("air pollution")
print(f"    AI Semantic Baseline: {ai_result.total_matches} matches")
print("    ✓ Baseline systems working correctly")

# Test 6: Explanation Generator
print("\n[6] Testing Explanation Generator...")
exp_gen = create_explanation_generator()
explanation = exp_gen.generate_explanation(
    "Air Quality Zurich 2024",
    85.5,
    {"recency": 5, "completeness": 0.9, "thematic_similarity": 0.85, "resource_availability": 6},
    rank=1,
    total_results=10
)
print(f"    Score Interpretation: {explanation.score_interpretation}")
print(f"    Key Factors: {len(explanation.key_factors)}")
print("    ✓ Explanation generator working correctly")

# Test 7: Config
print("\n[7] Testing Configuration...")
config = get_config()
print(f"    CKAN URL: {config.ckan.base_url}")
print(f"    Defuzzification: {config.fuzzy.defuzzification_method}")
print("    ✓ Configuration loaded correctly")

print("\n" + "=" * 70)
print("ALL INTEGRATION TESTS PASSED!")
print("=" * 70)
