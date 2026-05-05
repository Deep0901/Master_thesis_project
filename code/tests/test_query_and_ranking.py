#!/usr/bin/env python3
"""Focused tests for query parsing and fuzzy ranking behavior."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from code.query_processing import create_parser
from code.ranking.fuzzy_ranker import SimilarityCalculator, FuzzyRanker


def test_query_parser_drops_noise_terms():
    parser = create_parser()
    parsed = parser.parse("mobility data related with bicycle")

    assert parsed.keywords == ["mobility", "bicycle"]
    assert "mobility" in parsed.themes


def test_similarity_does_not_double_count_partial_matches():
    calculator = SimilarityCalculator(method="tfidf")
    calculator.fit([
        {
            "id": "aerosol",
            "title": {"en": "Aerosol data Weissfluhjoch"},
            "description": "Atmospheric monitoring data",
            "tags": ["aerosol", "monitoring"],
            "groups": ["environment"],
        }
    ])

    score = calculator.calculate(
        ["mobility", "bicycle", "related", "show"],
        {
            "title": {"en": "Mobility bicycle inventory"},
            "description": "Mobility bicycle inventory",
            "tags": ["bicycle", "mobility"],
            "groups": ["mobility"],
        }
    )

    assert 0.30 <= score <= 0.65


def test_ranker_prefers_relevant_mobility_dataset():
    ranker = FuzzyRanker()

    datasets = [
        {
            "id": "aerosol",
            "title": {"en": "Aerosol data Weissfluhjoch"},
            "description": "Atmospheric monitoring data",
            "tags": ["aerosol", "monitoring"],
            "groups": ["environment"],
            "resources": [{"format": "CSV"}],
            "days_since_modified": 20,
        },
        {
            "id": "mobility-bike",
            "title": {"en": "Bicycle mobility counts"},
            "description": "Counts and trends for bicycle mobility",
            "tags": ["mobility", "bicycle", "transport"],
            "groups": ["mobility"],
            "resources": [{"format": "CSV"}, {"format": "JSON"}],
            "days_since_modified": 5,
        },
    ]

    result = ranker.rank_datasets("mobility data related with bicycle", datasets, top_n=2)

    assert result.ranked_datasets[0].dataset_id == "mobility-bike"
    assert result.ranked_datasets[0].relevance_score >= result.ranked_datasets[1].relevance_score