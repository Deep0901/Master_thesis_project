#!/usr/bin/env python3
"""
Comprehensive Test Suite for Swiss OGD Fuzzy Ranking System

Run with: python -m pytest code/tests/ -v
"""

import pytest
import sys
import os
from typing import Dict, List

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from code.fuzzy_system.production_engine import (
    MembershipFunction, MembershipFunctionType,
    LinguisticVariable, FuzzyRule,
    CalibratedOGDVariables, OGDRuleBase,
    MamdaniFuzzyEngine, create_ogd_fuzzy_engine,
    TNorm, SNorm, DefuzzificationMethod
)


# ============================================================================
# MEMBERSHIP FUNCTION TESTS
# ============================================================================

class TestMembershipFunction:
    """Tests for MembershipFunction class."""
    
    def test_triangular_membership(self):
        """Test triangular membership function."""
        mf = MembershipFunction(
            name="test_tri",
            mf_type=MembershipFunctionType.TRIANGULAR,
            parameters=[0, 5, 10]
        )
        
        # Test peak
        assert mf.evaluate(5) == 1.0
        
        # Test edges
        assert mf.evaluate(0) == 0.0
        assert mf.evaluate(10) == 0.0
        
        # Test midpoints
        assert abs(mf.evaluate(2.5) - 0.5) < 0.001
        assert abs(mf.evaluate(7.5) - 0.5) < 0.001
        
        # Test outside range
        assert mf.evaluate(-5) == 0.0
        assert mf.evaluate(15) == 0.0
    
    def test_trapezoidal_membership(self):
        """Test trapezoidal membership function."""
        mf = MembershipFunction(
            name="test_trap",
            mf_type=MembershipFunctionType.TRAPEZOIDAL,
            parameters=[0, 2, 8, 10]
        )
        
        # Test plateau
        assert mf.evaluate(5) == 1.0
        assert mf.evaluate(2) == 1.0
        assert mf.evaluate(8) == 1.0
        
        # Test edges
        assert mf.evaluate(0) == 0.0
        assert mf.evaluate(10) == 0.0
        
        # Test slopes
        assert abs(mf.evaluate(1) - 0.5) < 0.001
        assert abs(mf.evaluate(9) - 0.5) < 0.001
    
    def test_left_shoulder_membership(self):
        """Test left shoulder (R-function) membership."""
        mf = MembershipFunction(
            name="test_left",
            mf_type=MembershipFunctionType.LEFT_SHOULDER,
            parameters=[0, 5]
        )
        
        # 1.0 at and below lower bound
        assert mf.evaluate(0) == 1.0
        assert mf.evaluate(-10) == 1.0
        
        # 0.0 at and above upper bound
        assert mf.evaluate(5) == 0.0
        assert mf.evaluate(10) == 0.0
        
        # Linear in between
        assert abs(mf.evaluate(2.5) - 0.5) < 0.001
    
    def test_right_shoulder_membership(self):
        """Test right shoulder (L-function) membership."""
        mf = MembershipFunction(
            name="test_right",
            mf_type=MembershipFunctionType.RIGHT_SHOULDER,
            parameters=[5, 10]
        )
        
        # 0.0 at and below lower bound
        assert mf.evaluate(5) == 0.0
        assert mf.evaluate(0) == 0.0
        
        # 1.0 at and above upper bound
        assert mf.evaluate(10) == 1.0
        assert mf.evaluate(15) == 1.0
        
        # Linear in between
        assert abs(mf.evaluate(7.5) - 0.5) < 0.001


# ============================================================================
# LINGUISTIC VARIABLE TESTS
# ============================================================================

class TestLinguisticVariable:
    """Tests for LinguisticVariable class."""
    
    def test_fuzzification(self):
        """Test fuzzification of crisp values."""
        var = CalibratedOGDVariables.create_completeness_variable()
        
        # High completeness
        memberships = var.fuzzify(0.95)
        assert memberships['very_complete'] > memberships['complete']
        assert memberships['complete'] > memberships['moderate']
        
        # Low completeness
        memberships = var.fuzzify(0.35)
        assert memberships['sparse'] > memberships['moderate']
    
    def test_universe_bounds(self):
        """Test that membership respects universe bounds."""
        var = CalibratedOGDVariables.create_recency_variable()
        
        # Should handle values beyond calibrated range
        memberships_low = var.fuzzify(0)
        memberships_high = var.fuzzify(5000)
        
        # Very recent should be dominant for day 0
        assert memberships_low['recent'] > 0.5
        
        # Old should be dominant for very old datasets
        assert memberships_high['old'] > 0.5


# ============================================================================
# CALIBRATED VARIABLES TESTS  
# ============================================================================

class TestCalibratedVariables:
    """Tests for calibrated OGD variables."""
    
    def test_recency_calibration(self):
        """Test recency variable matches real portal statistics."""
        var = CalibratedOGDVariables.create_recency_variable()
        
        # P25 ≈ 223 days should be "recent"
        m_223 = var.fuzzify(223)
        assert m_223['recent'] > 0.3 or m_223['fairly_recent'] > 0.3
        
        # Median ≈ 776 days should be "moderate"
        m_776 = var.fuzzify(776)
        assert m_776['moderate'] > 0.0
        
        # P90 ≈ 2731 days should be "old"
        m_2731 = var.fuzzify(2731)
        assert m_2731['old'] > 0.5
    
    def test_completeness_calibration(self):
        """Test completeness variable calibration."""
        var = CalibratedOGDVariables.create_completeness_variable()
        
        # 50% completeness is minimum observed - should be sparse
        m_50 = var.fuzzify(0.50)
        assert m_50['sparse'] > 0.0 or m_50['moderate'] > 0.0
        
        # 83% is maximum - should be very complete
        m_83 = var.fuzzify(0.83)
        assert m_83['complete'] > 0.3 or m_83['very_complete'] > 0.0
    
    def test_resources_calibration(self):
        """Test resources variable calibration."""
        var = CalibratedOGDVariables.create_resources_variable()
        
        # Median is 4 resources
        m_4 = var.fuzzify(4)
        # Should have moderate membership
        assert m_4['moderate'] > 0.0 or m_4['few'] > 0.0
        
        # Mean is 6 resources
        m_6 = var.fuzzify(6)
        assert m_6['moderate'] > 0.3 or m_6['many'] > 0.0
    
    def test_output_variable(self):
        """Test relevance output variable."""
        var = CalibratedOGDVariables.create_relevance_variable()
        
        # Should have all expected terms
        assert 'not_relevant' in var.terms
        assert 'low' in var.terms
        assert 'moderate' in var.terms
        assert 'high' in var.terms
        assert 'very_relevant' in var.terms


# ============================================================================
# FUZZY RULE TESTS
# ============================================================================

class TestFuzzyRules:
    """Tests for fuzzy rule base."""
    
    def test_rule_creation(self):
        """Test rule creation and properties."""
        rule = FuzzyRule(
            id=1,
            antecedent={'similarity': 'high', 'recency': 'recent'},
            consequent={'relevance': 'very_relevant'},
            weight=1.0
        )
        
        assert rule.id == 1
        assert rule.antecedent['similarity'] == 'high'
        assert rule.consequent['relevance'] == 'very_relevant'
    
    def test_rule_base_completeness(self):
        """Test that rule base covers important scenarios."""
        rules = OGDRuleBase.create_rules()
        
        # Should have reasonable number of rules
        assert len(rules) >= 20
        
        # Check for high relevance rules
        high_rules = [r for r in rules 
                     if r.consequent.get('relevance') in ['very_relevant', 'high']]
        assert len(high_rules) > 0
        
        # Check for low relevance rules
        low_rules = [r for r in rules 
                    if r.consequent.get('relevance') in ['not_relevant', 'low']]
        assert len(low_rules) > 0
    
    def test_rule_base_similarity_importance(self):
        """Test that similarity drives relevance when high."""
        rules = OGDRuleBase.create_rules()
        
        # Rules with high similarity should tend toward high relevance
        high_sim_rules = [r for r in rules if 'similarity' in r.antecedent 
                         and r.antecedent['similarity'] in ['high', 'very_high']]
        
        for rule in high_sim_rules:
            assert rule.consequent['relevance'] in ['moderate', 'high', 'very_relevant'], \
                f"Rule {rule.id}: High similarity should not lead to low relevance"


# ============================================================================
# FUZZY INFERENCE ENGINE TESTS
# ============================================================================

class TestMamdaniFuzzyEngine:
    """Tests for Mamdani fuzzy inference engine."""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance for testing."""
        return create_ogd_fuzzy_engine()
    
    def test_engine_creation(self):
        """Test engine can be created successfully."""
        engine = create_ogd_fuzzy_engine()
        assert engine is not None
        assert len(engine.rules) > 0
    
    def test_basic_inference(self, engine):
        """Test basic inference produces valid output."""
        result = engine.infer({
            'recency': 100,
            'completeness': 0.8,
            'resources': 5,
            'similarity': 0.7
        })
        
        # Output should be in [0, 1]
        assert 0.0 <= result.crisp_output <= 1.0
        
        # Should have fired some rules
        assert len(result.active_rules) > 0
    
    def test_high_quality_input(self, engine):
        """Test that high quality input produces high relevance."""
        result = engine.infer({
            'recency': 30,      # Very recent
            'completeness': 0.9,  # Very complete
            'resources': 8,      # Many resources
            'similarity': 0.9    # High similarity
        })
        
        # Should produce high relevance
        assert result.crisp_output > 0.5, \
            f"High quality input should produce high relevance, got {result.crisp_output}"
    
    def test_low_quality_input(self, engine):
        """Test that low quality input produces low relevance."""
        result = engine.infer({
            'recency': 3000,     # Very old
            'completeness': 0.4,  # Sparse
            'resources': 1,       # Few resources
            'similarity': 0.2     # Low similarity
        })
        
        # Should produce lower relevance
        assert result.crisp_output < 0.7, \
            f"Low quality input should produce lower relevance, got {result.crisp_output}"
    
    def test_similarity_dominance(self, engine):
        """Test that similarity is most important factor."""
        # High similarity with mediocre quality
        result_high_sim = engine.infer({
            'recency': 500,
            'completeness': 0.6,
            'resources': 3,
            'similarity': 0.95
        })
        
        # Low similarity with high quality
        result_low_sim = engine.infer({
            'recency': 30,
            'completeness': 0.9,
            'resources': 10,
            'similarity': 0.2
        })
        
        # High similarity should yield higher relevance
        assert result_high_sim.crisp_output > result_low_sim.crisp_output, \
            "Similarity should be the dominant factor"
    
    def test_monotonicity_recency(self, engine):
        """Test that more recent is generally better (with other factors fixed)."""
        base_inputs = {'completeness': 0.7, 'resources': 5, 'similarity': 0.6}
        
        result_recent = engine.infer({**base_inputs, 'recency': 50})
        result_moderate = engine.infer({**base_inputs, 'recency': 500})
        result_old = engine.infer({**base_inputs, 'recency': 2000})
        
        # Recent should be >= moderate >= old (generally)
        assert result_recent.crisp_output >= result_old.crisp_output - 0.1, \
            "More recent should generally produce higher or equal relevance"
    
    def test_monotonicity_completeness(self, engine):
        """Test that more complete is generally better."""
        base_inputs = {'recency': 200, 'resources': 5, 'similarity': 0.6}
        
        result_high = engine.infer({**base_inputs, 'completeness': 0.9})
        result_low = engine.infer({**base_inputs, 'completeness': 0.45})
        
        assert result_high.crisp_output >= result_low.crisp_output - 0.1, \
            "Higher completeness should generally produce higher relevance"
    
    def test_inference_result_structure(self, engine):
        """Test inference result contains all expected data."""
        result = engine.infer({
            'recency': 100,
            'completeness': 0.7,
            'resources': 4,
            'similarity': 0.6
        })
        
        # Check result structure
        assert hasattr(result, 'crisp_output')
        assert hasattr(result, 'output_memberships')
        assert hasattr(result, 'active_rules')
        assert hasattr(result, 'dominant_factors')
        
        # Check active rules have firing strength
        for rule_act in result.active_rules:
            assert 0.0 <= rule_act.firing_strength <= 1.0
    
    def test_different_tnorms(self):
        """Test engine works with different T-norms."""
        for tnorm in [TNorm.MIN, TNorm.PRODUCT, TNorm.LUKASIEWICZ]:
            engine = create_ogd_fuzzy_engine()
            engine.t_norm = tnorm
            
            result = engine.infer({
                'recency': 100,
                'completeness': 0.75,
                'resources': 5,
                'similarity': 0.7
            })
            
            assert 0.0 <= result.crisp_output <= 1.0, \
                f"T-norm {tnorm} should produce valid output"
    
    def test_different_defuzzification(self):
        """Test engine works with different defuzzification methods."""
        for method in [DefuzzificationMethod.CENTROID, 
                      DefuzzificationMethod.BISECTOR,
                      DefuzzificationMethod.MOM]:
            engine = create_ogd_fuzzy_engine()
            engine.defuzz_method = method
            
            result = engine.infer({
                'recency': 100,
                'completeness': 0.75,
                'resources': 5,
                'similarity': 0.7
            })
            
            assert 0.0 <= result.crisp_output <= 1.0, \
                f"Defuzzification {method} should produce valid output"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete fuzzy ranking workflow."""
    
    def test_full_ranking_workflow(self):
        """Test complete workflow from inputs to ranked score."""
        engine = create_ogd_fuzzy_engine()
        
        # Simulate ranking multiple datasets
        datasets = [
            {'recency': 30, 'completeness': 0.85, 'resources': 6, 'similarity': 0.8},
            {'recency': 500, 'completeness': 0.6, 'resources': 3, 'similarity': 0.7},
            {'recency': 1500, 'completeness': 0.5, 'resources': 2, 'similarity': 0.5},
            {'recency': 100, 'completeness': 0.9, 'resources': 10, 'similarity': 0.95},
        ]
        
        scores = []
        for ds in datasets:
            result = engine.infer(ds)
            scores.append(result.crisp_output)
        
        # Best dataset (last one) should have highest score
        assert max(scores) == scores[3], "Dataset 4 should have highest relevance"
        
        # Worst dataset (third) should have lower score
        assert scores[2] < scores[0], "Dataset 3 should score lower than dataset 1"
    
    def test_edge_cases(self):
        """Test engine handles edge cases gracefully."""
        engine = create_ogd_fuzzy_engine()
        
        # Extreme values
        edge_cases = [
            {'recency': 0, 'completeness': 1.0, 'resources': 0, 'similarity': 1.0},
            {'recency': 10000, 'completeness': 0.0, 'resources': 100, 'similarity': 0.0},
            {'recency': 365, 'completeness': 0.5, 'resources': 5, 'similarity': 0.5},
        ]
        
        for inputs in edge_cases:
            result = engine.infer(inputs)
            assert 0.0 <= result.crisp_output <= 1.0, \
                f"Edge case {inputs} should produce valid output"
    
    def test_explanation_generation(self):
        """Test that explanations can be generated from results."""
        engine = create_ogd_fuzzy_engine()
        
        result = engine.infer({
            'recency': 50,
            'completeness': 0.85,
            'resources': 7,
            'similarity': 0.8
        })
        
        # Get top contributing factors
        factors = result.dominant_factors
        assert len(factors) > 0
        
        # Each factor should have (variable, term, degree)
        for var_name, term, degree in factors:
            assert isinstance(var_name, str)
            assert isinstance(term, str)
            assert 0.0 <= degree <= 1.0


# ============================================================================
# BENCHMARK TESTS
# ============================================================================

class TestBenchmark:
    """Benchmark tests for performance validation."""
    
    def test_inference_speed(self):
        """Test inference completes in reasonable time."""
        import time
        
        engine = create_ogd_fuzzy_engine()
        
        start = time.time()
        for _ in range(100):
            engine.infer({
                'recency': 100,
                'completeness': 0.75,
                'resources': 5,
                'similarity': 0.7
            })
        elapsed = time.time() - start
        
        # Should complete 100 inferences in < 5 seconds
        assert elapsed < 5.0, f"100 inferences took {elapsed:.2f}s (should be < 5s)"
    
    def test_result_consistency(self):
        """Test same inputs produce same outputs."""
        engine = create_ogd_fuzzy_engine()
        
        inputs = {
            'recency': 200,
            'completeness': 0.7,
            'resources': 6,
            'similarity': 0.65
        }
        
        results = [engine.infer(inputs).crisp_output for _ in range(10)]
        
        # All results should be identical
        assert len(set(results)) == 1, "Inference should be deterministic"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
