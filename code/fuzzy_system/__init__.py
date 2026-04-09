"""
Fuzzy System Package for OGD Metadata Ranking

This package implements a complete Mamdani fuzzy inference system
for ranking Open Government Data datasets based on metadata quality.

Modules:
- linguistic_variables: Fuzzy variable definitions
- membership_functions: Membership function implementations
- fuzzy_rules: IF-THEN rule base
- inference_engine: Mamdani inference system
- defuzzifier: Output conversion methods

Usage:
    from code.fuzzy_system import create_inference_engine
    
    engine = create_inference_engine()
    result = engine.infer({
        "recency": 10,
        "completeness": 0.8,
        "thematic_similarity": 0.7,
        "resource_availability": 5
    })
    
    print(f"Relevance Score: {result.crisp_output:.1f}")
    print(result.get_explanation())
"""

from .linguistic_variables import (
    FuzzyVariable,
    RECENCY, COMPLETENESS, THEMATIC_SIMILARITY,
    RESOURCE_AVAILABILITY, RELEVANCE_SCORE,
    INPUT_VARIABLES, OUTPUT_VARIABLES, ALL_VARIABLES,
    get_variable, get_variable_terms, describe_variables
)

from .membership_functions import (
    MembershipFunction,
    triangular, trapezoidal, gaussian, sigmoid,
    create_membership_function, create_from_variable_definition,
    fuzzy_and, fuzzy_or, fuzzy_not, aggregate_memberships
)

from .fuzzy_rules import (
    FuzzyRule, RuleBase, 
    get_default_rules, DEFAULT_RULE_BASE
)

from .inference_engine import (
    MamdaniInferenceEngine,
    FuzzificationResult, RuleActivation, InferenceResult,
    create_inference_engine
)

__all__ = [
    # Variables
    'FuzzyVariable',
    'RECENCY', 'COMPLETENESS', 'THEMATIC_SIMILARITY',
    'RESOURCE_AVAILABILITY', 'RELEVANCE_SCORE',
    'INPUT_VARIABLES', 'OUTPUT_VARIABLES', 'ALL_VARIABLES',
    'get_variable', 'get_variable_terms', 'describe_variables',
    
    # Membership Functions
    'MembershipFunction',
    'triangular', 'trapezoidal', 'gaussian', 'sigmoid',
    'create_membership_function', 'create_from_variable_definition',
    'fuzzy_and', 'fuzzy_or', 'fuzzy_not', 'aggregate_memberships',
    
    # Rules
    'FuzzyRule', 'RuleBase', 'get_default_rules', 'DEFAULT_RULE_BASE',
    
    # Inference
    'MamdaniInferenceEngine',
    'FuzzificationResult', 'RuleActivation', 'InferenceResult',
    'create_inference_engine'
]

__version__ = "0.1.0"
__author__ = "Deep Shukla"
