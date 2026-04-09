"""
Human-Readable Explanation Generator

Generates transparent, human-readable explanations for
ranking decisions made by the fuzzy inference system.

Research Context:
- Part of Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR"
- Addresses RQ3: Operationalizing explainability for user trust

Explanation Features:
- Natural language rule descriptions
- Input factor contributions
- Comparative ranking rationale
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class ExplanationComponent:
    """A single component of the explanation."""
    factor: str
    value: float
    linguistic_term: str
    contribution: str  # "positive", "negative", "neutral"
    natural_language: str


@dataclass
class RankingExplanation:
    """Complete explanation for a ranking decision."""
    dataset_title: str
    relevance_score: float
    score_interpretation: str  # "excellent", "good", "moderate", "low", "very_low"
    key_factors: List[ExplanationComponent]
    active_rules: List[str]
    comparative_note: str
    full_explanation: str
    
    def to_dict(self) -> Dict:
        return {
            "title": self.dataset_title,
            "score": self.relevance_score,
            "interpretation": self.score_interpretation,
            "factors": [
                {
                    "factor": f.factor,
                    "value": f.value,
                    "term": f.linguistic_term,
                    "contribution": f.contribution,
                    "explanation": f.natural_language
                }
                for f in self.key_factors
            ],
            "rules": self.active_rules,
            "comparative": self.comparative_note,
            "full_text": self.full_explanation
        }
    
    def to_html(self) -> str:
        """Generate HTML formatted explanation."""
        factors_html = "\n".join([
            f'<li><strong>{f.factor}:</strong> {f.natural_language}</li>'
            for f in self.key_factors
        ])
        
        return f"""
        <div class="ranking-explanation">
            <h3>{self.dataset_title}</h3>
            <div class="score">
                <span class="score-value">{self.relevance_score:.1f}</span>
                <span class="score-label">{self.score_interpretation}</span>
            </div>
            <h4>Key Factors:</h4>
            <ul class="factors-list">
                {factors_html}
            </ul>
            <div class="comparative-note">
                <em>{self.comparative_note}</em>
            </div>
        </div>
        """


class ExplanationGenerator:
    """
    Generate human-readable explanations for ranking decisions.
    
    Translates fuzzy inference results into natural language
    that users can understand and trust.
    """
    
    # Score interpretation thresholds
    SCORE_INTERPRETATIONS = [
        (80, "excellent", "highly recommended"),
        (60, "good", "recommended"),
        (40, "moderate", "potentially relevant"),
        (20, "low", "marginally relevant"),
        (0, "very_low", "not recommended")
    ]
    
    # Factor templates
    FACTOR_TEMPLATES = {
        "recency": {
            "very_recent": "This dataset was updated very recently ({value} days ago), indicating current and fresh data.",
            "recent": "This dataset was updated recently ({value} days ago), suggesting relatively current information.",
            "moderate": "This dataset was last updated {value} days ago, which is moderately recent.",
            "old": "This dataset hasn't been updated in {value} days, which may affect data currency.",
            "very_old": "This dataset is quite outdated ({value} days since last update)."
        },
        "completeness": {
            "complete": "The metadata is comprehensive ({value:.0%} complete), providing full documentation.",
            "mostly_complete": "The metadata is mostly complete ({value:.0%}), with good documentation coverage.",
            "partial": "The metadata is partially complete ({value:.0%}), missing some information.",
            "sparse": "The metadata is sparse ({value:.0%}), with limited documentation.",
            "empty": "The metadata is minimal ({value:.0%}), lacking important information."
        },
        "thematic_similarity": {
            "exact_match": "This dataset is an exact match to your query ({value:.0%} similarity).",
            "highly_relevant": "This dataset is highly relevant to your query ({value:.0%} similarity).",
            "relevant": "This dataset is relevant to your query ({value:.0%} similarity).",
            "somewhat_relevant": "This dataset is somewhat relevant to your query ({value:.0%} similarity).",
            "not_relevant": "This dataset has low relevance to your query ({value:.0%} similarity)."
        },
        "resource_availability": {
            "comprehensive": "The dataset offers comprehensive data resources ({value} formats available).",
            "good": "The dataset provides good resource availability ({value} formats).",
            "limited": "The dataset has limited resource availability ({value} formats).",
            "minimal": "The dataset has minimal resources available ({value} format only)."
        }
    }
    
    # Contribution thresholds
    CONTRIBUTION_THRESHOLDS = {
        "recency": {"positive": 30, "negative": 365},
        "completeness": {"positive": 0.7, "negative": 0.4},
        "thematic_similarity": {"positive": 0.6, "negative": 0.3},
        "resource_availability": {"positive": 4, "negative": 2}
    }
    
    def __init__(self):
        """Initialize the explanation generator."""
        pass
    
    def interpret_score(self, score: float) -> Tuple[str, str]:
        """
        Interpret a relevance score.
        
        Args:
            score: Relevance score 0-100
            
        Returns:
            Tuple of (interpretation, description)
        """
        for threshold, interp, desc in self.SCORE_INTERPRETATIONS:
            if score >= threshold:
                return interp, desc
        return "very_low", "not recommended"
    
    def get_linguistic_term(self, factor: str, value: float) -> str:
        """
        Determine the linguistic term for a factor value.
        
        Args:
            factor: Factor name
            value: Crisp value
            
        Returns:
            Linguistic term
        """
        if factor == "recency":
            if value <= 7:
                return "very_recent"
            elif value <= 30:
                return "recent"
            elif value <= 180:
                return "moderate"
            elif value <= 365:
                return "old"
            else:
                return "very_old"
        
        elif factor == "completeness":
            if value >= 0.9:
                return "complete"
            elif value >= 0.7:
                return "mostly_complete"
            elif value >= 0.4:
                return "partial"
            elif value >= 0.15:
                return "sparse"
            else:
                return "empty"
        
        elif factor == "thematic_similarity":
            if value >= 0.9:
                return "exact_match"
            elif value >= 0.7:
                return "highly_relevant"
            elif value >= 0.4:
                return "relevant"
            elif value >= 0.15:
                return "somewhat_relevant"
            else:
                return "not_relevant"
        
        elif factor == "resource_availability":
            if value >= 8:
                return "comprehensive"
            elif value >= 4:
                return "good"
            elif value >= 2:
                return "limited"
            else:
                return "minimal"
        
        return "unknown"
    
    def get_contribution(self, factor: str, value: float) -> str:
        """
        Determine if a factor contributes positively, negatively, or neutrally.
        
        Args:
            factor: Factor name
            value: Factor value
            
        Returns:
            "positive", "negative", or "neutral"
        """
        thresholds = self.CONTRIBUTION_THRESHOLDS.get(factor, {})
        
        if factor == "recency":
            # For recency, lower is better
            if value <= thresholds.get("positive", 30):
                return "positive"
            elif value >= thresholds.get("negative", 365):
                return "negative"
        else:
            # For other factors, higher is better
            if value >= thresholds.get("positive", 0.6):
                return "positive"
            elif value <= thresholds.get("negative", 0.3):
                return "negative"
        
        return "neutral"
    
    def generate_factor_explanation(
        self,
        factor: str,
        value: float
    ) -> ExplanationComponent:
        """
        Generate explanation for a single factor.
        
        Args:
            factor: Factor name
            value: Factor value
            
        Returns:
            ExplanationComponent with natural language explanation
        """
        term = self.get_linguistic_term(factor, value)
        contribution = self.get_contribution(factor, value)
        
        # Get template
        templates = self.FACTOR_TEMPLATES.get(factor, {})
        template = templates.get(term, f"{factor}: {value}")
        
        # Format template
        try:
            natural_language = template.format(value=value)
        except:
            natural_language = f"{factor}: {value}"
        
        return ExplanationComponent(
            factor=factor,
            value=value,
            linguistic_term=term,
            contribution=contribution,
            natural_language=natural_language
        )
    
    def generate_explanation(
        self,
        dataset_title: str,
        relevance_score: float,
        input_scores: Dict[str, float],
        active_rules: List[str] = None,
        rank: int = None,
        total_results: int = None
    ) -> RankingExplanation:
        """
        Generate complete explanation for a ranking decision.
        
        Args:
            dataset_title: Dataset title
            relevance_score: Final relevance score
            input_scores: Dictionary of input factor scores
            active_rules: List of active rule descriptions
            rank: Position in results (optional)
            total_results: Total number of results (optional)
            
        Returns:
            RankingExplanation with full explanation
        """
        # Interpret score
        interpretation, interpretation_desc = self.interpret_score(relevance_score)
        
        # Generate factor explanations
        key_factors = []
        for factor, value in input_scores.items():
            if factor in self.FACTOR_TEMPLATES:
                key_factors.append(self.generate_factor_explanation(factor, value))
        
        # Sort by contribution (positive first)
        key_factors.sort(
            key=lambda f: (
                0 if f.contribution == "positive" else 
                1 if f.contribution == "neutral" else 2
            )
        )
        
        # Generate comparative note
        if rank and total_results:
            comparative_note = f"Ranked #{rank} out of {total_results} results."
        else:
            comparative_note = f"This dataset is {interpretation_desc} based on your search criteria."
        
        # Build full explanation
        positive_factors = [f.natural_language for f in key_factors if f.contribution == "positive"]
        negative_factors = [f.natural_language for f in key_factors if f.contribution == "negative"]
        
        full_parts = [
            f"This dataset received a score of {relevance_score:.1f}/100 ({interpretation}).",
            ""
        ]
        
        if positive_factors:
            full_parts.append("Strengths:")
            full_parts.extend([f"• {s}" for s in positive_factors])
            full_parts.append("")
        
        if negative_factors:
            full_parts.append("Weaknesses:")
            full_parts.extend([f"• {w}" for w in negative_factors])
        
        full_explanation = "\n".join(full_parts)
        
        return RankingExplanation(
            dataset_title=dataset_title,
            relevance_score=relevance_score,
            score_interpretation=interpretation,
            key_factors=key_factors,
            active_rules=active_rules or [],
            comparative_note=comparative_note,
            full_explanation=full_explanation
        )


def create_explanation_generator() -> ExplanationGenerator:
    """Factory function to create explanation generator."""
    return ExplanationGenerator()


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("EXPLANATION GENERATOR DEMONSTRATION")
    print("=" * 60)
    
    generator = create_explanation_generator()
    
    # Test with sample data
    test_cases = [
        {
            "title": "Air Quality Zurich 2024",
            "score": 85.3,
            "inputs": {
                "recency": 5,
                "completeness": 0.92,
                "thematic_similarity": 0.88,
                "resource_availability": 6
            }
        },
        {
            "title": "Traffic Statistics 2020",
            "score": 42.1,
            "inputs": {
                "recency": 400,
                "completeness": 0.55,
                "thematic_similarity": 0.35,
                "resource_availability": 2
            }
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        
        explanation = generator.generate_explanation(
            dataset_title=case["title"],
            relevance_score=case["score"],
            input_scores=case["inputs"],
            rank=i,
            total_results=len(test_cases)
        )
        
        print(f"Title: {explanation.dataset_title}")
        print(f"Score: {explanation.relevance_score:.1f} ({explanation.score_interpretation})")
        print(f"\n{explanation.full_explanation}")
        print(f"\n{explanation.comparative_note}")
