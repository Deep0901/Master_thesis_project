#!/usr/bin/env python3
"""
Comprehensive Fuzzy Inference Engine - Production Grade

A complete implementation of Mamdani fuzzy inference for OGD ranking,
with data-driven calibration and full explainability support.

Reference Papers:
- Zadeh, L.A. (1965). "Fuzzy Sets"
- Mamdani, E.H. & Assilian, S. (1975). "Fuzzy Logic Controller"
- Kraft et al. (1999). "Fuzzy Set Techniques in IR"

This module implements:
1. Calibrated membership functions from real data
2. Linguistic variables for OGD ranking
3. Complete Mamdani inference with centroid defuzzification
4. Explanation generation for transparency
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class MembershipFunctionType(Enum):
    """Types of membership functions."""
    TRIANGULAR = "triangular"
    TRAPEZOIDAL = "trapezoidal"
    GAUSSIAN = "gaussian"
    SIGMOID = "sigmoid"
    BELL = "bell"


class TNorm(Enum):
    """T-norm (AND) operators."""
    MINIMUM = "min"
    PRODUCT = "product"
    LUKASIEWICZ = "lukasiewicz"


class SNorm(Enum):
    """S-norm (OR) operators."""
    MAXIMUM = "max"
    PROBABILISTIC = "probabilistic"
    BOUNDED = "bounded"


class DefuzzificationMethod(Enum):
    """Defuzzification methods."""
    CENTROID = "centroid"
    BISECTOR = "bisector"
    MOM = "mom"  # Mean of Maximum
    SOM = "som"  # Smallest of Maximum
    LOM = "lom"  # Largest of Maximum


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MembershipFunction:
    """A single membership function definition."""
    name: str
    type: MembershipFunctionType
    parameters: List[float]
    
    def evaluate(self, x: float) -> float:
        """Compute membership degree for value x."""
        if self.type == MembershipFunctionType.TRIANGULAR:
            return self._triangular(x)
        elif self.type == MembershipFunctionType.TRAPEZOIDAL:
            return self._trapezoidal(x)
        elif self.type == MembershipFunctionType.GAUSSIAN:
            return self._gaussian(x)
        elif self.type == MembershipFunctionType.SIGMOID:
            return self._sigmoid(x)
        elif self.type == MembershipFunctionType.BELL:
            return self._bell(x)
        return 0.0
    
    def _triangular(self, x: float) -> float:
        """Triangular MF: [a, b, c]"""
        a, b, c = self.parameters[:3]
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:
            return (c - x) / (c - b) if c != b else 1.0
    
    def _trapezoidal(self, x: float) -> float:
        """Trapezoidal MF: [a, b, c, d]"""
        a, b, c, d = self.parameters[:4]
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b < x <= c:
            return 1.0
        else:
            return (d - x) / (d - c) if d != c else 1.0
    
    def _gaussian(self, x: float) -> float:
        """Gaussian MF: [mean, sigma]"""
        mean, sigma = self.parameters[:2]
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid MF: [a, c] where a is slope, c is crossover"""
        a, c = self.parameters[:2]
        return 1 / (1 + np.exp(-a * (x - c)))
    
    def _bell(self, x: float) -> float:
        """Generalized bell MF: [a, b, c]"""
        a, b, c = self.parameters[:3]
        return 1 / (1 + abs((x - c) / a) ** (2 * b))


@dataclass
class LinguisticVariable:
    """A fuzzy linguistic variable with multiple terms."""
    name: str
    universe_min: float
    universe_max: float
    units: str
    terms: Dict[str, MembershipFunction] = field(default_factory=dict)
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """Convert crisp value to fuzzy memberships."""
        # Clamp to universe
        value = max(self.universe_min, min(self.universe_max, value))
        return {term: mf.evaluate(value) for term, mf in self.terms.items()}
    
    def dominant_term(self, value: float) -> Tuple[str, float]:
        """Get the dominant (highest membership) term."""
        memberships = self.fuzzify(value)
        if not memberships:
            return ("unknown", 0.0)
        return max(memberships.items(), key=lambda x: x[1])


@dataclass
class FuzzyRule:
    """A single fuzzy IF-THEN rule."""
    id: int
    antecedents: Dict[str, str]  # variable -> term
    consequent: Tuple[str, str]  # (variable, term)
    weight: float = 1.0
    description: str = ""
    
    def __str__(self):
        ante_str = " AND ".join(f"{v} IS {t}" for v, t in self.antecedents.items())
        cons_str = f"{self.consequent[0]} IS {self.consequent[1]}"
        return f"IF {ante_str} THEN {cons_str}"


@dataclass
class FuzzificationResult:
    """Result of fuzzifying an input value."""
    variable_name: str
    crisp_value: float
    memberships: Dict[str, float]
    dominant_term: str
    dominant_degree: float


@dataclass
class RuleActivation:
    """Result of evaluating a fuzzy rule."""
    rule: FuzzyRule
    firing_strength: float
    antecedent_degrees: Dict[str, float]


@dataclass
class InferenceResult:
    """Complete result of fuzzy inference."""
    output_variable: str
    crisp_output: float
    output_memberships: Dict[str, float]
    fuzzifications: Dict[str, FuzzificationResult]
    rule_activations: List[RuleActivation]
    active_rules: List[RuleActivation]  # Rules with non-zero firing
    dominant_factors: List[Tuple[str, str, float]]  # Top contributing factors
    
    def get_top_rules(self, n: int = 3) -> List[RuleActivation]:
        """Get top n rules by firing strength."""
        return sorted(self.active_rules, key=lambda x: x.firing_strength, reverse=True)[:n]


# ============================================================================
# CALIBRATED VARIABLES (FROM OPENDATA.SWISS STATISTICS)
# ============================================================================

class CalibratedOGDVariables:
    """
    Linguistic variables calibrated from actual opendata.swiss statistics.
    
    Calibration Data (sampled 600 datasets, March 2026):
    - Recency: mean=1149 days, median=776, P10=2, P25=151, P50=776, P75=1583, P90=3559
    - Completeness: mean=74.9%, median=83.3%, range 50-83%
    - Resources: mean=4.8, median=4, P75=6, P90=8, max=138
    """
    
    @staticmethod
    def create_recency_variable() -> LinguisticVariable:
        """
        Create recency variable (days since modification).
        
        Calibration rationale:
        - very_recent: < P10 (newest 10%, within ~30 days)
        - recent: P5-P30 (within a year)
        - moderate: Around median (1-3 years)
        - old: P60-P90 (3-10 years)
        - very_old: > P85 (oldest datasets)
        """
        var = LinguisticVariable(
            name="recency",
            universe_min=0,
            universe_max=4500,
            units="days"
        )
        
        var.terms = {
            "very_recent": MembershipFunction(
                name="very_recent",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[0, 0, 30]  # Peak at 0, zero at 30 days
            ),
            "recent": MembershipFunction(
                name="recent",
                type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=[7, 30, 150, 365]  # 1 week to 1 year
            ),
            "moderate": MembershipFunction(
                name="moderate",
                type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=[180, 365, 776, 1200]  # 6 months to 3 years
            ),
            "old": MembershipFunction(
                name="old",
                type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=[776, 1200, 2500, 3500]  # 2-10 years
            ),
            "very_old": MembershipFunction(
                name="very_old",
                type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=[2500, 3500, 4500, 4500]  # >7 years
            )
        }
        
        return var
    
    @staticmethod
    def create_completeness_variable() -> LinguisticVariable:
        """
        Create completeness variable (metadata quality ratio).
        
        Based on DCAT-AP CH field population analysis.
        Observed range: 50% - 83% (narrow due to standard requirements)
        """
        var = LinguisticVariable(
            name="completeness",
            universe_min=0,
            universe_max=1.0,
            units="ratio"
        )
        
        var.terms = {
            "low": MembershipFunction(
                name="low",
                type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=[0, 0, 0.45, 0.58]
            ),
            "partial": MembershipFunction(
                name="partial",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[0.48, 0.62, 0.73]
            ),
            "medium": MembershipFunction(
                name="medium",
                type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=[0.65, 0.72, 0.78, 0.85]
            ),
            "high": MembershipFunction(
                name="high",
                type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=[0.78, 0.83, 0.92, 0.97]
            ),
            "complete": MembershipFunction(
                name="complete",
                type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=[0.92, 0.97, 1.0, 1.0]
            )
        }
        
        return var
    
    @staticmethod
    def create_resources_variable() -> LinguisticVariable:
        """
        Create resource availability variable.
        
        Calibration: P25=2, P50=4, P75=6, P90=8, max observed=138
        Most datasets have 1-6 resources.
        """
        var = LinguisticVariable(
            name="resources",
            universe_min=0,
            universe_max=50,
            units="count"
        )
        
        var.terms = {
            "minimal": MembershipFunction(
                name="minimal",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[0, 1, 2]
            ),
            "limited": MembershipFunction(
                name="limited",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[1, 2, 4]
            ),
            "moderate": MembershipFunction(
                name="moderate",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[2, 4, 7]
            ),
            "rich": MembershipFunction(
                name="rich",
                type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=[5, 7, 12, 18]
            ),
            "comprehensive": MembershipFunction(
                name="comprehensive",
                type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=[15, 25, 50, 50]
            )
        }
        
        return var
    
    @staticmethod
    def create_similarity_variable() -> LinguisticVariable:
        """
        Create query-document similarity variable.
        
        This is computed from text matching algorithms (TF-IDF, BM25).
        Range: 0 (no match) to 1 (perfect match)
        """
        var = LinguisticVariable(
            name="similarity",
            universe_min=0,
            universe_max=1.0,
            units="score"
        )
        
        var.terms = {
            "not_relevant": MembershipFunction(
                name="not_relevant",
                type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=[0, 0, 0.10, 0.20]
            ),
            "somewhat_relevant": MembershipFunction(
                name="somewhat_relevant",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[0.12, 0.28, 0.42]
            ),
            "relevant": MembershipFunction(
                name="relevant",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[0.35, 0.52, 0.68]
            ),
            "highly_relevant": MembershipFunction(
                name="highly_relevant",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[0.60, 0.78, 0.92]
            ),
            "exact_match": MembershipFunction(
                name="exact_match",
                type=MembershipFunctionType.TRAPEZOIDAL,
                parameters=[0.85, 0.94, 1.0, 1.0]
            )
        }
        
        return var
    
    @staticmethod
    def create_relevance_output_variable() -> LinguisticVariable:
        """
        Create output relevance variable.
        
        This is the defuzzified output representing dataset relevance.
        """
        var = LinguisticVariable(
            name="relevance",
            universe_min=0,
            universe_max=1.0,
            units="score"
        )
        
        var.terms = {
            "very_low": MembershipFunction(
                name="very_low",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[0, 0, 0.25]
            ),
            "low": MembershipFunction(
                name="low",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[0.12, 0.30, 0.45]
            ),
            "moderate": MembershipFunction(
                name="moderate",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[0.38, 0.52, 0.65]
            ),
            "good": MembershipFunction(
                name="good",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[0.58, 0.74, 0.88]
            ),
            "excellent": MembershipFunction(
                name="excellent",
                type=MembershipFunctionType.TRIANGULAR,
                parameters=[0.82, 0.95, 1.0]
            )
        }
        
        return var


# ============================================================================
# RULE BASE
# ============================================================================

class OGDRuleBase:
    """
    Fuzzy rule base for OGD ranking.
    
    Rules derived from:
    1. Domain expert knowledge (what makes a good dataset?)
    2. User study insights (what do users value?)
    3. Literature review (IR ranking factors)
    """
    
    @staticmethod
    def get_rules() -> List[FuzzyRule]:
        """Generate the complete rule base."""
        rules = []
        rule_id = 1
        
        # ===================
        # HIGH QUALITY RULES
        # ===================
        
        # Perfect match scenarios
        rules.append(FuzzyRule(
            id=rule_id, 
            antecedents={"similarity": "exact_match", "recency": "very_recent", "completeness": "high"},
            consequent=("relevance", "excellent"),
            weight=1.0,
            description="Perfect match: exact keywords, fresh data, well documented"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "exact_match", "recency": "very_recent"},
            consequent=("relevance", "excellent"),
            weight=0.95,
            description="Exact match with very recent data"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "exact_match", "completeness": "complete"},
            consequent=("relevance", "excellent"),
            weight=0.95,
            description="Exact match with complete documentation"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "highly_relevant", "recency": "recent", "completeness": "high"},
            consequent=("relevance", "excellent"),
            weight=0.9,
            description="Highly relevant, recent, well-documented"
        ))
        rule_id += 1
        
        # ===================
        # GOOD QUALITY RULES
        # ===================
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "highly_relevant", "recency": "recent"},
            consequent=("relevance", "good"),
            weight=1.0,
            description="Good relevance with recent updates"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "highly_relevant", "completeness": "high"},
            consequent=("relevance", "good"),
            weight=1.0,
            description="Good relevance with high completeness"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "relevant", "recency": "very_recent", "completeness": "high"},
            consequent=("relevance", "good"),
            weight=0.9,
            description="Relevant with outstanding freshness and docs"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "relevant", "resources": "comprehensive"},
            consequent=("relevance", "good"),
            weight=0.85,
            description="Relevant with extensive resources"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "exact_match", "recency": "moderate"},
            consequent=("relevance", "good"),
            weight=0.85,
            description="Exact match but moderately old"
        ))
        rule_id += 1
        
        # ===================
        # MODERATE QUALITY RULES
        # ===================
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "relevant", "completeness": "medium"},
            consequent=("relevance", "moderate"),
            weight=1.0,
            description="Relevant with average documentation"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "relevant", "recency": "moderate"},
            consequent=("relevance", "moderate"),
            weight=1.0,
            description="Relevant with moderate freshness"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "somewhat_relevant", "recency": "very_recent"},
            consequent=("relevance", "moderate"),
            weight=0.9,
            description="Partial relevance offset by freshness"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "highly_relevant", "recency": "old"},
            consequent=("relevance", "moderate"),
            weight=0.85,
            description="Good relevance but old data"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"recency": "old", "resources": "comprehensive"},
            consequent=("relevance", "moderate"),
            weight=0.7,
            description="Old but comprehensive resources"
        ))
        rule_id += 1
        
        # ===================
        # LOW QUALITY RULES
        # ===================
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"recency": "very_old"},
            consequent=("relevance", "low"),
            weight=1.0,
            description="Very outdated data penalized"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"completeness": "low"},
            consequent=("relevance", "low"),
            weight=1.0,
            description="Poor documentation penalized"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "somewhat_relevant", "recency": "old"},
            consequent=("relevance", "low"),
            weight=0.9,
            description="Weak relevance and old"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"completeness": "partial", "resources": "minimal"},
            consequent=("relevance", "low"),
            weight=0.85,
            description="Poor documentation with minimal resources"
        ))
        rule_id += 1
        
        # ===================
        # VERY LOW QUALITY RULES
        # ===================
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"similarity": "not_relevant"},
            consequent=("relevance", "very_low"),
            weight=1.0,
            description="No relevance to query"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"completeness": "low", "similarity": "somewhat_relevant"},
            consequent=("relevance", "very_low"),
            weight=1.0,
            description="Poor docs and weak relevance"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"resources": "minimal", "completeness": "low"},
            consequent=("relevance", "very_low"),
            weight=0.95,
            description="Minimal resources with poor docs"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"recency": "very_old", "completeness": "low"},
            consequent=("relevance", "very_low"),
            weight=0.95,
            description="Outdated with poor documentation"
        ))
        rule_id += 1
        
        # ===================
        # COMBINED QUALITY BOOSTER RULES
        # ===================
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"recency": "very_recent", "completeness": "complete", "resources": "rich"},
            consequent=("relevance", "excellent"),
            weight=0.8,
            description="Outstanding quality across all dimensions"
        ))
        rule_id += 1
        
        rules.append(FuzzyRule(
            id=rule_id,
            antecedents={"recency": "recent", "completeness": "high", "resources": "moderate"},
            consequent=("relevance", "good"),
            weight=0.8,
            description="Good quality across all dimensions"
        ))
        rule_id += 1
        
        return rules


# ============================================================================
# MAMDANI INFERENCE ENGINE
# ============================================================================

class MamdaniFuzzyEngine:
    """
    Complete Mamdani Fuzzy Inference Engine.
    
    Implements:
    1. Fuzzification of crisp inputs
    2. Rule evaluation with configurable operators
    3. Output aggregation
    4. Defuzzification (centroid, bisector, MOM)
    5. Explanation generation
    """
    
    def __init__(
        self,
        t_norm: TNorm = TNorm.MINIMUM,
        s_norm: SNorm = SNorm.MAXIMUM,
        defuzz_method: DefuzzificationMethod = DefuzzificationMethod.CENTROID,
        universe_resolution: int = 500
    ):
        """
        Initialize the inference engine.
        
        Args:
            t_norm: T-norm for AND operations
            s_norm: S-norm for OR operations
            defuzz_method: Defuzzification method
            universe_resolution: Points for fuzzy set representation
        """
        self.t_norm = t_norm
        self.s_norm = s_norm
        self.defuzz_method = defuzz_method
        self.resolution = universe_resolution
        
        # Initialize variables
        self.input_variables: Dict[str, LinguisticVariable] = {}
        self.output_variable: Optional[LinguisticVariable] = None
        self.rules: List[FuzzyRule] = []
        
        # Set up OGD-specific configuration
        self._setup_ogd_system()
    
    def _setup_ogd_system(self):
        """Set up variables and rules for OGD ranking."""
        # Create calibrated input variables
        self.input_variables = {
            "recency": CalibratedOGDVariables.create_recency_variable(),
            "completeness": CalibratedOGDVariables.create_completeness_variable(),
            "resources": CalibratedOGDVariables.create_resources_variable(),
            "similarity": CalibratedOGDVariables.create_similarity_variable()
        }
        
        # Create output variable
        self.output_variable = CalibratedOGDVariables.create_relevance_output_variable()
        
        # Load rule base
        self.rules = OGDRuleBase.get_rules()
        
        logger.info(f"Initialized fuzzy engine with {len(self.input_variables)} input variables "
                   f"and {len(self.rules)} rules")
    
    def _apply_t_norm(self, a: float, b: float) -> float:
        """Apply T-norm (AND) operator."""
        if self.t_norm == TNorm.MINIMUM:
            return min(a, b)
        elif self.t_norm == TNorm.PRODUCT:
            return a * b
        elif self.t_norm == TNorm.LUKASIEWICZ:
            return max(0, a + b - 1)
        return min(a, b)
    
    def _apply_s_norm(self, a: float, b: float) -> float:
        """Apply S-norm (OR) operator."""
        if self.s_norm == SNorm.MAXIMUM:
            return max(a, b)
        elif self.s_norm == SNorm.PROBABILISTIC:
            return a + b - a * b
        elif self.s_norm == SNorm.BOUNDED:
            return min(1, a + b)
        return max(a, b)
    
    def fuzzify_input(
        self,
        variable_name: str,
        crisp_value: float
    ) -> FuzzificationResult:
        """
        Fuzzify a single input value.
        
        Args:
            variable_name: Name of the linguistic variable
            crisp_value: Crisp input value
            
        Returns:
            FuzzificationResult with all membership degrees
        """
        if variable_name not in self.input_variables:
            raise ValueError(f"Unknown variable: {variable_name}")
        
        var = self.input_variables[variable_name]
        memberships = var.fuzzify(crisp_value)
        dom_term, dom_degree = var.dominant_term(crisp_value)
        
        return FuzzificationResult(
            variable_name=variable_name,
            crisp_value=crisp_value,
            memberships=memberships,
            dominant_term=dom_term,
            dominant_degree=dom_degree
        )
    
    def evaluate_rule(
        self,
        rule: FuzzyRule,
        fuzzifications: Dict[str, FuzzificationResult]
    ) -> RuleActivation:
        """
        Evaluate a single fuzzy rule.
        
        Args:
            rule: The fuzzy rule to evaluate
            fuzzifications: Fuzzification results for all inputs
            
        Returns:
            RuleActivation with firing strength
        """
        # Get membership degrees for all antecedents
        antecedent_degrees = {}
        firing_strength = 1.0  # Start with maximum for AND chaining
        
        for var_name, term in rule.antecedents.items():
            if var_name in fuzzifications:
                degree = fuzzifications[var_name].memberships.get(term, 0.0)
                antecedent_degrees[var_name] = degree
                firing_strength = self._apply_t_norm(firing_strength, degree)
        
        # Apply rule weight
        firing_strength *= rule.weight
        
        return RuleActivation(
            rule=rule,
            firing_strength=firing_strength,
            antecedent_degrees=antecedent_degrees
        )
    
    def aggregate_outputs(
        self,
        rule_activations: List[RuleActivation]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Aggregate all rule outputs into a single fuzzy set.
        
        Args:
            rule_activations: List of rule activations
            
        Returns:
            Tuple of (aggregated fuzzy set array, output term memberships)
        """
        if not self.output_variable:
            raise ValueError("Output variable not defined")
        
        # Create universe array
        x = np.linspace(
            self.output_variable.universe_min,
            self.output_variable.universe_max,
            self.resolution
        )
        
        # Initialize aggregated output
        aggregated = np.zeros(self.resolution)
        output_memberships = {}
        
        # Process each rule
        for activation in rule_activations:
            if activation.firing_strength > 0.001:  # Skip negligible
                output_term = activation.rule.consequent[1]
                
                # Track output term activations
                if output_term not in output_memberships:
                    output_memberships[output_term] = 0.0
                output_memberships[output_term] = self._apply_s_norm(
                    output_memberships[output_term],
                    activation.firing_strength
                )
                
                # Get output membership function
                if output_term in self.output_variable.terms:
                    output_mf = self.output_variable.terms[output_term]
                    
                    # Apply implication (clip to firing strength) and aggregate
                    for i, xi in enumerate(x):
                        mu = output_mf.evaluate(xi)
                        # Mamdani implication: min
                        clipped = min(mu, activation.firing_strength)
                        # Aggregation: max
                        aggregated[i] = self._apply_s_norm(aggregated[i], clipped)
        
        return aggregated, output_memberships
    
    def defuzzify(self, aggregated: np.ndarray) -> float:
        """
        Defuzzify aggregated fuzzy output to crisp value.
        
        Args:
            aggregated: Aggregated fuzzy output array
            
        Returns:
            Crisp output value
        """
        if not self.output_variable:
            raise ValueError("Output variable not defined")
        
        x = np.linspace(
            self.output_variable.universe_min,
            self.output_variable.universe_max,
            self.resolution
        )
        
        if np.sum(aggregated) < 0.001:
            # No rules fired - return middle of universe
            return (self.output_variable.universe_max + self.output_variable.universe_min) / 2
        
        if self.defuzz_method == DefuzzificationMethod.CENTROID:
            # Center of gravity
            return np.sum(x * aggregated) / np.sum(aggregated)
        
        elif self.defuzz_method == DefuzzificationMethod.BISECTOR:
            # Point that divides area in half
            total_area = np.sum(aggregated)
            cumsum = np.cumsum(aggregated)
            idx = np.argmax(cumsum >= total_area / 2)
            return x[idx]
        
        elif self.defuzz_method == DefuzzificationMethod.MOM:
            # Mean of maximum
            max_val = np.max(aggregated)
            if max_val > 0:
                max_indices = np.where(aggregated >= max_val - 0.001)[0]
                return np.mean(x[max_indices])
            return x[len(x) // 2]
        
        elif self.defuzz_method == DefuzzificationMethod.SOM:
            # Smallest of maximum
            max_val = np.max(aggregated)
            if max_val > 0:
                return x[np.argmax(aggregated >= max_val - 0.001)]
            return x[0]
        
        elif self.defuzz_method == DefuzzificationMethod.LOM:
            # Largest of maximum
            max_val = np.max(aggregated)
            if max_val > 0:
                max_indices = np.where(aggregated >= max_val - 0.001)[0]
                return x[max_indices[-1]]
            return x[-1]
        
        # Default to centroid
        return np.sum(x * aggregated) / np.sum(aggregated)
    
    def infer(
        self,
        inputs: Dict[str, float]
    ) -> InferenceResult:
        """
        Perform complete fuzzy inference.
        
        Args:
            inputs: Dictionary of variable_name -> crisp_value
            
        Returns:
            InferenceResult with full details
        """
        # Step 1: Fuzzification
        fuzzifications = {}
        for var_name, value in inputs.items():
            if var_name in self.input_variables:
                fuzzifications[var_name] = self.fuzzify_input(var_name, value)
        
        # Step 2: Rule evaluation
        rule_activations = []
        for rule in self.rules:
            activation = self.evaluate_rule(rule, fuzzifications)
            rule_activations.append(activation)
        
        # Get active rules (non-zero firing)
        active_rules = [a for a in rule_activations if a.firing_strength > 0.001]
        
        # Step 3: Aggregation
        aggregated, output_memberships = self.aggregate_outputs(rule_activations)
        
        # Step 4: Defuzzification
        crisp_output = self.defuzzify(aggregated)
        
        # Identify dominant contributing factors
        dominant_factors = []
        for fuzz in fuzzifications.values():
            if fuzz.dominant_degree > 0.3:
                dominant_factors.append(
                    (fuzz.variable_name, fuzz.dominant_term, fuzz.dominant_degree)
                )
        dominant_factors.sort(key=lambda x: x[2], reverse=True)
        
        return InferenceResult(
            output_variable="relevance",
            crisp_output=crisp_output,
            output_memberships=output_memberships,
            fuzzifications=fuzzifications,
            rule_activations=rule_activations,
            active_rules=active_rules,
            dominant_factors=dominant_factors[:4]
        )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_ogd_fuzzy_engine(
    defuzz_method: str = "centroid",
    resolution: int = 500
) -> MamdaniFuzzyEngine:
    """
    Factory function to create a configured OGD fuzzy engine.
    
    Args:
        defuzz_method: Defuzzification method name
        resolution: Universe resolution
        
    Returns:
        Configured MamdaniFuzzyEngine
    """
    method_map = {
        "centroid": DefuzzificationMethod.CENTROID,
        "bisector": DefuzzificationMethod.BISECTOR,
        "mom": DefuzzificationMethod.MOM,
        "som": DefuzzificationMethod.SOM,
        "lom": DefuzzificationMethod.LOM
    }
    
    return MamdaniFuzzyEngine(
        t_norm=TNorm.MINIMUM,
        s_norm=SNorm.MAXIMUM,
        defuzz_method=method_map.get(defuzz_method, DefuzzificationMethod.CENTROID),
        universe_resolution=resolution
    )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the fuzzy engine
    engine = create_ogd_fuzzy_engine()
    
    # Test case 1: High quality recent dataset
    print("=" * 60)
    print("Test Case 1: High quality recent dataset")
    result = engine.infer({
        "recency": 5,
        "completeness": 0.85,
        "resources": 6,
        "similarity": 0.9
    })
    print(f"Relevance Score: {result.crisp_output:.3f}")
    print(f"Active Rules: {len(result.active_rules)}")
    print(f"Dominant Factors: {result.dominant_factors}")
    
    # Test case 2: Old dataset with poor docs
    print("\n" + "=" * 60)
    print("Test Case 2: Old dataset with poor documentation")
    result = engine.infer({
        "recency": 2000,
        "completeness": 0.55,
        "resources": 1,
        "similarity": 0.4
    })
    print(f"Relevance Score: {result.crisp_output:.3f}")
    print(f"Active Rules: {len(result.active_rules)}")
    print(f"Dominant Factors: {result.dominant_factors}")
    
    # Test case 3: Good relevance but old
    print("\n" + "=" * 60)
    print("Test Case 3: Good relevance but old data")
    result = engine.infer({
        "recency": 1500,
        "completeness": 0.75,
        "resources": 4,
        "similarity": 0.8
    })
    print(f"Relevance Score: {result.crisp_output:.3f}")
    print(f"Active Rules: {len(result.active_rules)}")
    print(f"Top Rules:")
    for r in result.get_top_rules(3):
        print(f"  - {r.rule}: strength={r.firing_strength:.3f}")
