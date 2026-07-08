"""
Fuzzy Linguistic Variables for OGD Metadata Ranking

This module defines the fuzzy linguistic variables used in the
metadata-based ranking system. Each variable represents a
dimension of dataset relevance that can be expressed in
natural language terms.

Research Context:
- Part of Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR"
- Addresses RQ1: How can fuzzy logic model vagueness in OGD metadata?
- Based on: Zadeh (1965), "Fuzzy Sets"

Variables Defined:
1. RECENCY - Temporal freshness of dataset
2. COMPLETENESS - Metadata quality/fullness
3. THEMATIC_SIMILARITY - Query-topic relevance
4. RESOURCE_AVAILABILITY - Data resource richness
5. RELEVANCE_SCORE - Output aggregated ranking score
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum
import numpy as np


class LinguisticTerm(Enum):
    """Base enumeration for linguistic terms."""
    pass


class RecencyTerm(Enum):
    """Linguistic terms for RECENCY variable."""
    VERY_RECENT = "very_recent"      # Updated within last week
    RECENT = "recent"                 # Updated within last month
    MODERATE = "moderate"             # Updated within last 6 months
    OLD = "old"                       # Updated within last 2 years
    VERY_OLD = "very_old"            # Older than 2 years


class CompletenessTerm(Enum):
    """Linguistic terms for COMPLETENESS variable."""
    COMPLETE = "complete"             # All metadata fields filled
    MOSTLY_COMPLETE = "mostly_complete"  # Most important fields filled
    PARTIAL = "partial"               # Some fields filled
    SPARSE = "sparse"                 # Few fields filled
    EMPTY = "empty"                   # Minimal metadata


class ThematicSimilarityTerm(Enum):
    """Linguistic terms for THEMATIC_SIMILARITY variable."""
    EXACT_MATCH = "exact_match"       # Perfect query match
    HIGHLY_RELEVANT = "highly_relevant"  # Strong semantic match
    RELEVANT = "relevant"             # Moderate relevance
    SOMEWHAT_RELEVANT = "somewhat_relevant"  # Weak relevance
    NOT_RELEVANT = "not_relevant"     # No relevance


class ResourceAvailabilityTerm(Enum):
    """Linguistic terms for RESOURCE_AVAILABILITY variable."""
    COMPREHENSIVE = "comprehensive"   # Many diverse formats
    GOOD = "good"                     # Multiple useful formats
    LIMITED = "limited"               # Few formats available
    MINIMAL = "minimal"               # Only basic format


class RelevanceScoreTerm(Enum):
    """Linguistic terms for output RELEVANCE_SCORE variable."""
    EXCELLENT = "excellent"           # Highly recommended
    GOOD = "good"                     # Recommended
    MODERATE = "moderate"             # Acceptable
    LOW = "low"                       # Less relevant
    VERY_LOW = "very_low"            # Not recommended


@dataclass
class FuzzyVariable:
    """
    Represents a fuzzy linguistic variable.
    
    Attributes:
        name: Variable name (e.g., "recency")
        universe: Range of crisp values [min, max]
        terms: Dictionary mapping term names to membership function params
        unit: Optional unit of measurement
        description: Human-readable description
    """
    name: str
    universe: Tuple[float, float]
    terms: Dict[str, Dict]  # term_name -> membership function parameters
    unit: str = ""
    description: str = ""
    
    @property
    def universe_range(self) -> np.ndarray:
        """Generate universe of discourse as numpy array."""
        return np.linspace(self.universe[0], self.universe[1], 1000)


# =============================================================================
# VARIABLE DEFINITIONS
# =============================================================================

RECENCY = FuzzyVariable(
    name="recency",
    universe=(0, 730),  # Days since last update (0-2 years)
    unit="days",
    description="Temporal freshness of dataset metadata",
    terms={
        RecencyTerm.VERY_RECENT.value: {
            "type": "triangular",
            "params": [0, 0, 7]  # Peak at 0, ends at 7 days
        },
        RecencyTerm.RECENT.value: {
            "type": "triangular",
            "params": [0, 7, 30]  # 1 week to 1 month
        },
        RecencyTerm.MODERATE.value: {
            "type": "triangular",
            "params": [14, 90, 180]  # 2 weeks to 6 months
        },
        RecencyTerm.OLD.value: {
            "type": "triangular",
            "params": [90, 365, 548]  # 3 months to 1.5 years
        },
        RecencyTerm.VERY_OLD.value: {
            "type": "triangular",
            "params": [365, 730, 730]  # 1+ years
        }
    }
)

COMPLETENESS = FuzzyVariable(
    name="completeness",
    universe=(0, 1),  # Completeness score 0-100%
    unit="ratio",
    description="Proportion of metadata fields that are filled",
    terms={
        CompletenessTerm.COMPLETE.value: {
            "type": "triangular",
            "params": [0.9, 1.0, 1.0]  # 90-100%
        },
        CompletenessTerm.MOSTLY_COMPLETE.value: {
            "type": "triangular",
            "params": [0.7, 0.85, 0.95]  # 70-95%
        },
        CompletenessTerm.PARTIAL.value: {
            "type": "triangular",
            "params": [0.4, 0.55, 0.75]  # 40-75%
        },
        CompletenessTerm.SPARSE.value: {
            "type": "triangular",
            "params": [0.15, 0.3, 0.5]  # 15-50%
        },
        CompletenessTerm.EMPTY.value: {
            "type": "triangular",
            "params": [0, 0, 0.2]  # 0-20%
        }
    }
)

THEMATIC_SIMILARITY = FuzzyVariable(
    name="thematic_similarity",
    universe=(0, 1),  # Similarity score 0-1
    unit="score",
    description="Semantic similarity between query and dataset theme",
    terms={
        ThematicSimilarityTerm.EXACT_MATCH.value: {
            "type": "triangular",
            "params": [0.9, 1.0, 1.0]  # Near perfect match
        },
        ThematicSimilarityTerm.HIGHLY_RELEVANT.value: {
            "type": "triangular",
            "params": [0.7, 0.85, 0.95]  # Strong match
        },
        ThematicSimilarityTerm.RELEVANT.value: {
            "type": "triangular",
            "params": [0.4, 0.55, 0.75]  # Moderate match
        },
        ThematicSimilarityTerm.SOMEWHAT_RELEVANT.value: {
            "type": "triangular",
            "params": [0.15, 0.3, 0.45]  # Weak match
        },
        ThematicSimilarityTerm.NOT_RELEVANT.value: {
            "type": "triangular",
            "params": [0, 0, 0.2]  # No match
        }
    }
)

RESOURCE_AVAILABILITY = FuzzyVariable(
    name="resource_availability",
    universe=(0, 20),  # Number of resources/formats
    unit="count",
    description="Number and diversity of available data resources",
    terms={
        ResourceAvailabilityTerm.COMPREHENSIVE.value: {
            "type": "triangular",
            "params": [8, 15, 20]  # 8+ resources
        },
        ResourceAvailabilityTerm.GOOD.value: {
            "type": "triangular",
            "params": [4, 6, 10]  # 4-10 resources
        },
        ResourceAvailabilityTerm.LIMITED.value: {
            "type": "triangular",
            "params": [1, 2, 5]  # 1-5 resources
        },
        ResourceAvailabilityTerm.MINIMAL.value: {
            "type": "triangular",
            "params": [0, 0, 2]  # 0-2 resources
        }
    }
)

# Output variable
RELEVANCE_SCORE = FuzzyVariable(
    name="relevance_score",
    universe=(0, 100),  # Output ranking score
    unit="points",
    description="Aggregated ranking relevance score",
    terms={
        RelevanceScoreTerm.EXCELLENT.value: {
            "type": "triangular",
            "params": [80, 100, 100]  # Top ranking
        },
        RelevanceScoreTerm.GOOD.value: {
            "type": "triangular",
            "params": [60, 75, 90]  # Good ranking
        },
        RelevanceScoreTerm.MODERATE.value: {
            "type": "triangular",
            "params": [35, 50, 65]  # Medium ranking
        },
        RelevanceScoreTerm.LOW.value: {
            "type": "triangular",
            "params": [15, 25, 40]  # Low ranking
        },
        RelevanceScoreTerm.VERY_LOW.value: {
            "type": "triangular",
            "params": [0, 0, 20]  # Lowest ranking
        }
    }
)


# =============================================================================
# VARIABLE REGISTRY
# =============================================================================

INPUT_VARIABLES = {
    "recency": RECENCY,
    "completeness": COMPLETENESS,
    "thematic_similarity": THEMATIC_SIMILARITY,
    "resource_availability": RESOURCE_AVAILABILITY
}

OUTPUT_VARIABLES = {
    "relevance_score": RELEVANCE_SCORE
}

ALL_VARIABLES = {**INPUT_VARIABLES, **OUTPUT_VARIABLES}


def get_variable(name: str) -> FuzzyVariable:
    """
    Get a fuzzy variable by name.
    
    Args:
        name: Variable name
        
    Returns:
        FuzzyVariable instance
        
    Raises:
        KeyError: If variable not found
    """
    if name not in ALL_VARIABLES:
        raise KeyError(f"Unknown variable: {name}. Available: {list(ALL_VARIABLES.keys())}")
    return ALL_VARIABLES[name]


def get_variable_terms(name: str) -> List[str]:
    """Get list of term names for a variable."""
    return list(get_variable(name).terms.keys())


def describe_variables() -> str:
    """Generate human-readable description of all variables."""
    lines = ["=" * 60, "FUZZY LINGUISTIC VARIABLES", "=" * 60, ""]
    
    for name, var in ALL_VARIABLES.items():
        lines.append(f"Variable: {var.name.upper()}")
        lines.append(f"  Description: {var.description}")
        lines.append(f"  Universe: [{var.universe[0]}, {var.universe[1]}] {var.unit}")
        lines.append(f"  Terms: {', '.join(var.terms.keys())}")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_variables())
