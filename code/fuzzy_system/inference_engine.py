"""
Mamdani Fuzzy Inference Engine

This module implements the Mamdani fuzzy inference system for
aggregating fuzzy rules and producing a ranking score.

Research Context:
- Part of Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR"
- Addresses RQ1: How fuzzy logic models vagueness in metadata ranking
- Based on: Mamdani (1974), "Application of fuzzy algorithms"

Inference Steps:
1. Fuzzification: Convert crisp inputs to membership degrees
2. Rule Evaluation: Apply fuzzy rules to get rule strengths
3. Aggregation: Combine rule outputs
4. Defuzzification: Convert fuzzy output to crisp score
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .linguistic_variables import (
    INPUT_VARIABLES, OUTPUT_VARIABLES, 
    RECENCY, COMPLETENESS, THEMATIC_SIMILARITY, 
    RESOURCE_AVAILABILITY, RELEVANCE_SCORE,
    get_variable
)
from .membership_functions import (
    create_from_variable_definition,
    fuzzy_and, fuzzy_or
)
from .fuzzy_rules import FuzzyRule, RuleBase, get_default_rules


@dataclass
class FuzzificationResult:
    """Result of fuzzifying a crisp input value."""
    variable: str
    crisp_value: float
    memberships: Dict[str, float]  # term -> membership degree
    
    def dominant_term(self) -> Tuple[str, float]:
        """Get the term with highest membership."""
        if not self.memberships:
            return ("unknown", 0.0)
        return max(self.memberships.items(), key=lambda x: x[1])


@dataclass 
class RuleActivation:
    """Result of evaluating a single fuzzy rule."""
    rule: FuzzyRule
    firing_strength: float
    antecedent_memberships: Dict[str, float]
    
    def to_explanation(self) -> str:
        """Generate human-readable explanation of rule activation."""
        if self.firing_strength < 0.01:
            return f"Rule {self.rule.id} not activated"
        
        parts = []
        for (var, term), membership in self.antecedent_memberships.items():
            parts.append(f"{var}={term} ({membership:.2f})")
        
        return (
            f"Rule {self.rule.id}: [{', '.join(parts)}] "
            f"→ strength={self.firing_strength:.2f}"
        )


@dataclass
class InferenceResult:
    """Complete result of fuzzy inference."""
    crisp_output: float
    fuzzy_output: np.ndarray
    rule_activations: List[RuleActivation]
    fuzzification_results: Dict[str, FuzzificationResult]
    dominant_rules: List[Tuple[FuzzyRule, float]]  # Top contributing rules
    
    def get_explanation(self, top_n: int = 3) -> str:
        """
        Generate human-readable explanation of the ranking decision.
        
        This is key for addressing RQ3 (explainability).
        """
        lines = [
            "=" * 50,
            "RANKING EXPLANATION",
            "=" * 50,
            f"\nFinal Relevance Score: {self.crisp_output:.1f}/100",
            "\n--- Input Analysis ---"
        ]
        
        for var_name, fuzz_result in self.fuzzification_results.items():
            term, membership = fuzz_result.dominant_term()
            lines.append(
                f"  {var_name}: {fuzz_result.crisp_value:.2f} → "
                f"'{term}' (μ={membership:.2f})"
            )
        
        lines.append("\n--- Top Contributing Rules ---")
        
        for i, (rule, strength) in enumerate(self.dominant_rules[:top_n], 1):
            lines.append(f"  {i}. {rule.to_natural_language()}")
            lines.append(f"     Strength: {strength:.2f}")
            if rule.description:
                lines.append(f"     Reason: {rule.description}")
        
        return "\n".join(lines)


class MamdaniInferenceEngine:
    """
    Mamdani-type fuzzy inference system for OGD metadata ranking.
    
    Implements the complete fuzzy inference pipeline:
    fuzzification → rule evaluation → aggregation → defuzzification
    """
    
    def __init__(
        self,
        rule_base: Optional[RuleBase] = None,
        defuzzification_method: str = "centroid"
    ):
        """
        Initialize the inference engine.
        
        Args:
            rule_base: Fuzzy rule base (uses default if None)
            defuzzification_method: Method for defuzzification
                Options: "centroid", "bisector", "mom", "som", "lom"
        """
        self.rule_base = rule_base or get_default_rules()
        self.defuzz_method = defuzzification_method
        
        # Cache membership functions for efficiency
        self._mf_cache: Dict[str, Dict[str, Any]] = {}
        self._build_mf_cache()
        
        # Output universe for defuzzification
        self.output_universe = np.linspace(0, 100, 1000)
    
    def _build_mf_cache(self):
        """Pre-build membership functions for all variables."""
        all_vars = {**INPUT_VARIABLES, **OUTPUT_VARIABLES}
        
        for var_name, var in all_vars.items():
            self._mf_cache[var_name] = {}
            for term_name, term_def in var.terms.items():
                mf = create_from_variable_definition(term_def)
                self._mf_cache[var_name][term_name] = mf
    
    def fuzzify(self, variable: str, crisp_value: float) -> FuzzificationResult:
        """
        Convert a crisp value to fuzzy memberships.
        
        Args:
            variable: Name of the fuzzy variable
            crisp_value: Crisp input value
            
        Returns:
            FuzzificationResult with memberships for all terms
        """
        if variable not in self._mf_cache:
            raise ValueError(f"Unknown variable: {variable}")
        
        memberships = {}
        for term_name, mf in self._mf_cache[variable].items():
            memberships[term_name] = float(mf(crisp_value))
        
        return FuzzificationResult(
            variable=variable,
            crisp_value=crisp_value,
            memberships=memberships
        )
    
    def evaluate_rule(
        self,
        rule: FuzzyRule,
        fuzz_results: Dict[str, FuzzificationResult]
    ) -> RuleActivation:
        """
        Evaluate a single fuzzy rule.
        
        Uses minimum (AND) to combine antecedent memberships.
        
        Args:
            rule: The fuzzy rule to evaluate
            fuzz_results: Fuzzification results for all input variables
            
        Returns:
            RuleActivation with firing strength
        """
        antecedent_memberships = {}
        memberships_to_combine = []
        
        for var, term in rule.antecedents:
            if var not in fuzz_results:
                # Variable not provided - could use default or skip
                membership = 0.0
            else:
                membership = fuzz_results[var].memberships.get(term, 0.0)
            
            antecedent_memberships[(var, term)] = membership
            memberships_to_combine.append(membership)
        
        # Apply fuzzy AND (minimum) and rule weight
        firing_strength = fuzzy_and(*memberships_to_combine) * rule.weight
        
        return RuleActivation(
            rule=rule,
            firing_strength=firing_strength,
            antecedent_memberships=antecedent_memberships
        )
    
    def aggregate_outputs(
        self,
        activations: List[RuleActivation]
    ) -> Tuple[np.ndarray, List[Tuple[FuzzyRule, float]]]:
        """
        Aggregate rule outputs into a single fuzzy set.
        
        Uses maximum (OR) to combine clipped output membership functions.
        
        Args:
            activations: List of rule activations
            
        Returns:
            Tuple of (aggregated fuzzy set, list of (rule, strength))
        """
        aggregated = np.zeros_like(self.output_universe)
        dominant_rules = []
        
        for activation in activations:
            if activation.firing_strength > 0.001:  # Skip negligible activations
                # Get the output term from the rule
                _, output_term = activation.rule.consequent
                
                # Get the output membership function
                output_mf = self._mf_cache["relevance_score"][output_term]
                
                # Compute membership over output universe
                output_membership = output_mf(self.output_universe)
                
                # Clip by firing strength (implication)
                clipped = np.minimum(output_membership, activation.firing_strength)
                
                # Aggregate using maximum (union)
                aggregated = np.maximum(aggregated, clipped)
                
                dominant_rules.append((activation.rule, activation.firing_strength))
        
        # Sort by firing strength
        dominant_rules.sort(key=lambda x: x[1], reverse=True)
        
        return aggregated, dominant_rules
    
    def defuzzify(self, fuzzy_output: np.ndarray) -> float:
        """
        Convert aggregated fuzzy output to crisp value.
        
        Args:
            fuzzy_output: Aggregated fuzzy set
            
        Returns:
            Crisp output value
        """
        if np.sum(fuzzy_output) < 1e-10:
            return 0.0
        
        if self.defuzz_method == "centroid":
            # Center of gravity
            return np.sum(self.output_universe * fuzzy_output) / np.sum(fuzzy_output)
        
        elif self.defuzz_method == "bisector":
            # Point that divides the area in half
            cumsum = np.cumsum(fuzzy_output)
            target = cumsum[-1] / 2
            idx = np.searchsorted(cumsum, target)
            return self.output_universe[min(idx, len(self.output_universe) - 1)]
        
        elif self.defuzz_method == "mom":
            # Mean of maximum
            max_val = np.max(fuzzy_output)
            max_indices = np.where(np.isclose(fuzzy_output, max_val, atol=1e-6))[0]
            return np.mean(self.output_universe[max_indices])
        
        elif self.defuzz_method == "som":
            # Smallest of maximum
            max_val = np.max(fuzzy_output)
            max_indices = np.where(np.isclose(fuzzy_output, max_val, atol=1e-6))[0]
            return self.output_universe[max_indices[0]]
        
        elif self.defuzz_method == "lom":
            # Largest of maximum
            max_val = np.max(fuzzy_output)
            max_indices = np.where(np.isclose(fuzzy_output, max_val, atol=1e-6))[0]
            return self.output_universe[max_indices[-1]]
        
        else:
            raise ValueError(f"Unknown defuzzification method: {self.defuzz_method}")
    
    def infer(
        self,
        inputs: Dict[str, float]
    ) -> InferenceResult:
        """
        Perform complete fuzzy inference.
        
        Args:
            inputs: Dictionary of {variable_name: crisp_value}
                Required: thematic_similarity
                Optional: recency, completeness, resource_availability
                
        Returns:
            InferenceResult with crisp output and explanations
        """
        # Step 1: Fuzzification
        fuzz_results = {}
        for var_name, value in inputs.items():
            if var_name in INPUT_VARIABLES:
                fuzz_results[var_name] = self.fuzzify(var_name, value)
        
        # Step 2: Rule Evaluation
        rule_activations = []
        for rule in self.rule_base.get_rules():
            activation = self.evaluate_rule(rule, fuzz_results)
            rule_activations.append(activation)
        
        # Step 3: Aggregation
        fuzzy_output, dominant_rules = self.aggregate_outputs(rule_activations)
        
        # Step 4: Defuzzification
        crisp_output = self.defuzzify(fuzzy_output)
        
        return InferenceResult(
            crisp_output=crisp_output,
            fuzzy_output=fuzzy_output,
            rule_activations=rule_activations,
            fuzzification_results=fuzz_results,
            dominant_rules=dominant_rules
        )


def create_inference_engine(
    defuzzification: str = "centroid"
) -> MamdaniInferenceEngine:
    """
    Factory function to create an inference engine.
    
    Args:
        defuzzification: Defuzzification method to use
        
    Returns:
        Configured MamdaniInferenceEngine
    """
    return MamdaniInferenceEngine(defuzzification_method=defuzzification)


if __name__ == "__main__":
    # Demo: Run inference with sample inputs
    print("=" * 60)
    print("MAMDANI FUZZY INFERENCE ENGINE DEMONSTRATION")
    print("=" * 60)
    
    engine = create_inference_engine()
    
    # Test case 1: Excellent dataset
    print("\n--- Test Case 1: Excellent Dataset ---")
    inputs1 = {
        "recency": 3,              # 3 days old (very recent)
        "completeness": 0.95,      # 95% complete
        "thematic_similarity": 0.92,  # Near exact match
        "resource_availability": 8    # Comprehensive resources
    }
    
    result1 = engine.infer(inputs1)
    print(result1.get_explanation())
    
    # Test case 2: Moderate dataset
    print("\n--- Test Case 2: Moderate Dataset ---")
    inputs2 = {
        "recency": 120,            # 4 months old
        "completeness": 0.55,      # Partial completeness
        "thematic_similarity": 0.6,   # Relevant
        "resource_availability": 3    # Limited resources
    }
    
    result2 = engine.infer(inputs2)
    print(result2.get_explanation())
    
    # Test case 3: Poor dataset
    print("\n--- Test Case 3: Poor Dataset ---")
    inputs3 = {
        "recency": 600,            # Nearly 2 years old
        "completeness": 0.2,       # Sparse
        "thematic_similarity": 0.15,  # Not relevant
        "resource_availability": 1    # Minimal
    }
    
    result3 = engine.infer(inputs3)
    print(result3.get_explanation())
