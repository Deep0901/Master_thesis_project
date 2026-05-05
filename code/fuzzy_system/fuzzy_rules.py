"""
Fuzzy Rules for OGD Metadata Ranking

This module defines the fuzzy IF-THEN rules that form the
knowledge base of the ranking system. Rules combine input
variables to produce relevance scores.

Research Context:
- Part of Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR"
- Addresses RQ1: Rule-based inference for explainable ranking
- Based on: Mamdani (1974), Fuzzy Control Systems

Rule Format:
IF <antecedent> THEN <consequent>
IF (recency IS recent) AND (completeness IS complete) THEN (relevance IS excellent)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


@dataclass
class FuzzyRule:
    """
    Represents a single fuzzy IF-THEN rule.
    
    Attributes:
        id: Unique rule identifier
        antecedents: List of (variable, term) conditions
        consequent: (variable, term) output
        weight: Rule weight [0, 1], default 1.0
        description: Human-readable rule explanation
    """
    id: int
    antecedents: List[Tuple[str, str]]  # [(variable, term), ...]
    consequent: Tuple[str, str]          # (variable, term)
    weight: float = 1.0
    description: str = ""
    
    def to_natural_language(self) -> str:
        """Convert rule to human-readable format."""
        if_parts = [f"{var} IS {term}" for var, term in self.antecedents]
        antecedent_str = " AND ".join(if_parts)
        consequent_str = f"{self.consequent[0]} IS {self.consequent[1]}"
        
        return f"IF {antecedent_str} THEN {consequent_str}"
    
    def __str__(self) -> str:
        return f"R{self.id}: {self.to_natural_language()}"


class RuleBase:
    """
    Collection of fuzzy rules forming the knowledge base.
    
    The rule base encodes expert knowledge about how metadata
    attributes combine to determine dataset relevance for queries.
    """
    
    def __init__(self):
        self.rules: List[FuzzyRule] = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize the default rule base for OGD ranking."""
        
        # =================================================================
        # HIGH RELEVANCE RULES (Relevance: excellent or good)
        # =================================================================
        
        # Rule 1: Perfect match - recent, complete, highly relevant
        self.add_rule(
            antecedents=[
                ("recency", "very_recent"),
                ("completeness", "complete"),
                ("thematic_similarity", "exact_match")
            ],
            consequent=("relevance_score", "excellent"),
            description="Fresh, complete datasets with exact thematic match are excellent"
        )
        
        # Rule 2: Strong match - recent and highly relevant
        self.add_rule(
            antecedents=[
                ("recency", "recent"),
                ("completeness", "mostly_complete"),
                ("thematic_similarity", "highly_relevant")
            ],
            consequent=("relevance_score", "excellent"),
            description="Recent datasets with strong relevance are excellent"
        )
        
        # Rule 3: Complete and exact match (even if older)
        self.add_rule(
            antecedents=[
                ("completeness", "complete"),
                ("thematic_similarity", "exact_match")
            ],
            consequent=("relevance_score", "good"),
            description="Complete datasets with exact match are good regardless of age"
        )
        
        # Rule 4: Very recent with high relevance
        self.add_rule(
            antecedents=[
                ("recency", "very_recent"),
                ("thematic_similarity", "highly_relevant")
            ],
            consequent=("relevance_score", "good"),
            description="Very recent and highly relevant datasets are good"
        )
        
        # Rule 5: Good resources and relevant
        self.add_rule(
            antecedents=[
                ("resource_availability", "comprehensive"),
                ("thematic_similarity", "highly_relevant"),
                ("completeness", "mostly_complete")
            ],
            consequent=("relevance_score", "good"),
            description="Well-documented datasets with many resources are good"
        )
        
        # =================================================================
        # MODERATE RELEVANCE RULES
        # =================================================================
        
        # Rule 6: Moderate age but complete
        self.add_rule(
            antecedents=[
                ("recency", "moderate"),
                ("completeness", "complete"),
                ("thematic_similarity", "relevant")
            ],
            consequent=("relevance_score", "moderate"),
            description="Moderately old but complete and relevant datasets are moderate"
        )
        
        # Rule 7: Relevant but partial completeness
        self.add_rule(
            antecedents=[
                ("completeness", "partial"),
                ("thematic_similarity", "highly_relevant")
            ],
            consequent=("relevance_score", "moderate"),
            description="Highly relevant but incomplete datasets are moderate"
        )
        
        # Rule 8: Recent but only somewhat relevant
        self.add_rule(
            antecedents=[
                ("recency", "recent"),
                ("thematic_similarity", "somewhat_relevant")
            ],
            consequent=("relevance_score", "moderate"),
            description="Recent but weakly relevant datasets are moderate"
        )
        
        # Rule 9: Good completeness, limited resources
        self.add_rule(
            antecedents=[
                ("completeness", "mostly_complete"),
                ("resource_availability", "limited"),
                ("thematic_similarity", "relevant")
            ],
            consequent=("relevance_score", "moderate"),
            description="Well-described but resource-limited datasets are moderate"
        )
        
        # =================================================================
        # LOW RELEVANCE RULES
        # =================================================================
        
        # Rule 10: Old datasets
        self.add_rule(
            antecedents=[
                ("recency", "old"),
                ("thematic_similarity", "relevant")
            ],
            consequent=("relevance_score", "low"),
            description="Old but relevant datasets have low ranking"
        )
        
        # Rule 11: Sparse metadata
        self.add_rule(
            antecedents=[
                ("completeness", "sparse"),
                ("thematic_similarity", "relevant")
            ],
            consequent=("relevance_score", "low"),
            description="Sparse metadata reduces ranking even if relevant"
        )
        
        # Rule 12: Minimal resources
        self.add_rule(
            antecedents=[
                ("resource_availability", "minimal"),
                ("thematic_similarity", "somewhat_relevant")
            ],
            consequent=("relevance_score", "low"),
            description="Minimal resources with weak relevance ranks low"
        )
        
        # =================================================================
        # VERY LOW RELEVANCE RULES
        # =================================================================
        
        # Rule 13: Very old and incomplete
        self.add_rule(
            antecedents=[
                ("recency", "very_old"),
                ("completeness", "sparse")
            ],
            consequent=("relevance_score", "very_low"),
            description="Very old and poorly documented datasets rank very low"
        )
        
        # Rule 14: Not relevant
        self.add_rule(
            antecedents=[
                ("thematic_similarity", "not_relevant")
            ],
            consequent=("relevance_score", "very_low"),
            description="Non-relevant datasets always rank very low"
        )
        
        # Rule 15: Empty metadata
        self.add_rule(
            antecedents=[
                ("completeness", "empty")
            ],
            consequent=("relevance_score", "very_low"),
            description="Datasets with empty metadata rank very low"
        )

        # Rule 15b: Generic weak thematic match catch-all
        self.add_rule(
            antecedents=[
                ("thematic_similarity", "somewhat_relevant")
            ],
            consequent=("relevance_score", "low"),
            description="Weak thematic matches should still be handled by fuzzy rules"
        )

        # Rule 15c: Strong exact matches should rank well even when other inputs are modest
        self.add_rule(
            antecedents=[
                ("thematic_similarity", "exact_match")
            ],
            consequent=("relevance_score", "good"),
            description="Exact thematic matches should always produce a strong relevance signal"
        )
        
        # =================================================================
        # ADDITIONAL NUANCED RULES
        # =================================================================
        
        # Rule 16: Recent + complete + relevant = good
        self.add_rule(
            antecedents=[
                ("recency", "recent"),
                ("completeness", "complete"),
                ("thematic_similarity", "relevant")
            ],
            consequent=("relevance_score", "good"),
            description="Recent, complete, and relevant datasets are good"
        )
        
        # Rule 17: Comprehensive resources boost
        self.add_rule(
            antecedents=[
                ("resource_availability", "comprehensive"),
                ("thematic_similarity", "exact_match")
            ],
            consequent=("relevance_score", "excellent"),
            description="Comprehensive resources with exact match are excellent"
        )
        
        # Rule 18: Moderate all around
        self.add_rule(
            antecedents=[
                ("recency", "moderate"),
                ("completeness", "partial"),
                ("thematic_similarity", "relevant")
            ],
            consequent=("relevance_score", "moderate"),
            description="Average quality across all dimensions is moderate"
        )
        
        # Rule 19: Good recent data
        self.add_rule(
            antecedents=[
                ("recency", "very_recent"),
                ("resource_availability", "good"),
                ("thematic_similarity", "relevant")
            ],
            consequent=("relevance_score", "good"),
            description="Very recent with good resources and relevance is good"
        )
        
        # Rule 20: Highly relevant compensates for age
        self.add_rule(
            antecedents=[
                ("recency", "old"),
                ("completeness", "complete"),
                ("thematic_similarity", "exact_match")
            ],
            consequent=("relevance_score", "moderate"),
            description="Exact match with complete metadata can compensate for age"
        )
    
    def add_rule(
        self,
        antecedents: List[Tuple[str, str]],
        consequent: Tuple[str, str],
        weight: float = 1.0,
        description: str = ""
    ) -> FuzzyRule:
        """
        Add a new rule to the rule base.
        
        Args:
            antecedents: List of (variable, term) conditions
            consequent: Output (variable, term)
            weight: Rule weight
            description: Human-readable explanation
            
        Returns:
            Created FuzzyRule
        """
        rule = FuzzyRule(
            id=len(self.rules) + 1,
            antecedents=antecedents,
            consequent=consequent,
            weight=weight,
            description=description
        )
        self.rules.append(rule)
        return rule
    
    def get_rules(self) -> List[FuzzyRule]:
        """Get all rules."""
        return self.rules
    
    def get_rules_for_output(self, output_term: str) -> List[FuzzyRule]:
        """Get all rules that produce a specific output term."""
        return [r for r in self.rules if r.consequent[1] == output_term]
    
    def get_rules_using_variable(self, variable: str) -> List[FuzzyRule]:
        """Get all rules that use a specific input variable."""
        matching = []
        for rule in self.rules:
            if any(var == variable for var, term in rule.antecedents):
                matching.append(rule)
        return matching
    
    def export_rules_table(self) -> str:
        """Export rules as a formatted table."""
        lines = [
            "=" * 80,
            "FUZZY RULE BASE FOR OGD METADATA RANKING",
            "=" * 80,
            ""
        ]
        
        for rule in self.rules:
            lines.append(f"Rule {rule.id}:")
            lines.append(f"  {rule.to_natural_language()}")
            if rule.description:
                lines.append(f"  → {rule.description}")
            lines.append(f"  Weight: {rule.weight}")
            lines.append("")
        
        lines.append(f"Total rules: {len(self.rules)}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Export rule base as dictionary for serialization."""
        return {
            "rules": [
                {
                    "id": r.id,
                    "antecedents": r.antecedents,
                    "consequent": r.consequent,
                    "weight": r.weight,
                    "description": r.description
                }
                for r in self.rules
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RuleBase':
        """Create rule base from dictionary."""
        rb = cls()
        rb.rules = []  # Clear default rules
        
        for rule_data in data.get("rules", []):
            rb.add_rule(
                antecedents=rule_data["antecedents"],
                consequent=tuple(rule_data["consequent"]),
                weight=rule_data.get("weight", 1.0),
                description=rule_data.get("description", "")
            )
        
        return rb


# Singleton default rule base
DEFAULT_RULE_BASE = RuleBase()


def get_default_rules() -> RuleBase:
    """Get the default rule base."""
    return DEFAULT_RULE_BASE


if __name__ == "__main__":
    # Demo: Display all rules
    rb = RuleBase()
    print(rb.export_rules_table())
    
    # Show rules by output category
    print("\n" + "=" * 40)
    print("RULES BY OUTPUT CATEGORY")
    print("=" * 40)
    
    for term in ["excellent", "good", "moderate", "low", "very_low"]:
        rules = rb.get_rules_for_output(term)
        print(f"\n{term.upper()}: {len(rules)} rules")
