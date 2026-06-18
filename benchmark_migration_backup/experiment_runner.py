"""
Experimental Evaluation Runner

This module runs complete experimental comparisons between:
1. Fuzzy OGD Retrieval System (our approach)
2. Keyword Baseline (BM25/TF-IDF)
3. Simple Metadata Ranking

Research Questions Evaluated:
- RQ2: Does fuzzy ranking outperform keyword-based baseline?
- RQ4: How does fuzzy ranking compare to alternative approaches?

Methodology:
- Uses benchmark queries with ground truth relevance judgments
- Computes standard IR metrics (MAP, nDCG, P@K, R@K)
- Reports statistical significance via paired t-tests

Author: Deep Shukla
Thesis: Improving Access to Swiss OGD through Fuzzy HCIR
University of Fribourg, Human-IST Institute
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import defaultdict
import requests
import re
import math

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluation_framework import (
    EvaluationEngine, EvaluationQuery, RankingResult, 
    RelevanceJudgment, IRMetrics
)


# =============================================================================
# RETRIEVAL SYSTEMS
# =============================================================================

class BaseRetriever:
    """Base class for retrieval systems."""
    
    name = "base"
    BASE_URL = "https://opendata.swiss/api/3/action"
    
    def search(self, query: str, num_results: int = 100) -> List[Tuple[str, float]]:
        """
        Search and return ranked results.
        
        Returns:
            List of (dataset_id, score) tuples
        """
        raise NotImplementedError


class PortalBaseline(BaseRetriever):
    """
    Baseline: Use opendata.swiss portal's default search.
    
    This represents what users currently get with the default search.
    """
    
    name = "portal_default"
    
    def search(self, query: str, num_results: int = 100) -> List[Tuple[str, float]]:
        params = {
            'q': query,
            'rows': num_results
        }
        
        try:
            resp = requests.get(f"{self.BASE_URL}/package_search", params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json()['result']['results']
            
            # Portal returns results in ranked order
            # Assign decreasing scores based on position
            ranked = []
            for i, ds in enumerate(results):
                score = 1.0 - (i / (len(results) + 1))  # Decreasing scores
                ranked.append((ds['id'], score))
            
            return ranked
            
        except Exception as e:
            print(f"Portal search error: {e}")
            return []


class KeywordBaseline(BaseRetriever):
    """
    Baseline: BM25/TF-IDF keyword scoring.
    
    Implements BM25 scoring on metadata fields (title, description, keywords).
    """
    
    name = "keyword_bm25"
    
    # BM25 parameters
    K1 = 1.2
    B = 0.75
    
    def __init__(self):
        self.doc_cache = {}  # Cache fetched documents
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        if not text:
            return []
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove stopwords (basic German + English)
        stopwords = {'der', 'die', 'das', 'und', 'in', 'von', 'zu', 'mit', 'für',
                     'the', 'and', 'of', 'to', 'in', 'for', 'is', 'on', 'with'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]
    
    def _get_doc_text(self, dataset: Dict) -> str:
        """Extract searchable text from dataset."""
        parts = []
        
        # Title (with boost - repeat 3 times)
        title = dataset.get('title', {})
        if isinstance(title, dict):
            for lang in ['de', 'en', 'fr']:
                if title.get(lang):
                    parts.extend([title[lang]] * 3)  # Boost title
        elif title:
            parts.extend([str(title)] * 3)
        
        # Description
        desc = dataset.get('notes', '') or dataset.get('description', {})
        if isinstance(desc, dict):
            for lang in ['de', 'en', 'fr']:
                if desc.get(lang):
                    parts.append(desc[lang])
        elif desc:
            parts.append(str(desc))
        
        # Keywords
        for tag in dataset.get('tags', []):
            if isinstance(tag, dict):
                parts.append(tag.get('name', ''))
            else:
                parts.append(str(tag))
        
        # Organization
        org = dataset.get('organization', {})
        if isinstance(org, dict):
            parts.append(org.get('name', ''))
        
        return ' '.join(parts)
    
    def _bm25_score(self, query_terms: List[str], doc_terms: List[str], 
                   avg_doc_len: float, doc_freqs: Dict[str, int]) -> float:
        """
        Compute BM25 score for a document.
        """
        doc_len = len(doc_terms)
        term_freq = defaultdict(int)
        for term in doc_terms:
            term_freq[term] += 1
        
        score = 0.0
        N = 1000  # Assumed collection size
        
        for term in query_terms:
            if term not in term_freq:
                continue
            
            tf = term_freq[term]
            df = doc_freqs.get(term, 1)
            
            # IDF component
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            
            # TF component with length normalization
            numerator = tf * (self.K1 + 1)
            denominator = tf + self.K1 * (1 - self.B + self.B * (doc_len / avg_doc_len))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, num_results: int = 100) -> List[Tuple[str, float]]:
        # First, get candidates from portal
        params = {
            'q': query,
            'rows': num_results
        }
        
        try:
            resp = requests.get(f"{self.BASE_URL}/package_search", params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json()['result']['results']
        except Exception as e:
            print(f"Search error: {e}")
            return []
        
        if not results:
            return []
        
        # Tokenize query
        query_terms = self._tokenize(query)
        
        # Process documents
        doc_terms_all = []
        doc_ids = []
        
        for ds in results:
            doc_text = self._get_doc_text(ds)
            terms = self._tokenize(doc_text)
            doc_terms_all.append(terms)
            doc_ids.append(ds['id'])
        
        # Compute document frequencies
        doc_freqs = defaultdict(int)
        for terms in doc_terms_all:
            for term in set(terms):
                doc_freqs[term] += 1
        
        # Average document length
        avg_doc_len = np.mean([len(t) for t in doc_terms_all]) if doc_terms_all else 1
        
        # Score each document
        scores = []
        for terms in doc_terms_all:
            score = self._bm25_score(query_terms, terms, avg_doc_len, doc_freqs)
            scores.append(score)
        
        # Normalize scores
        max_score = max(scores) if scores and max(scores) > 0 else 1
        normalized = [(doc_ids[i], scores[i] / max_score) for i in range(len(scores))]
        
        # Sort by score
        normalized.sort(key=lambda x: x[1], reverse=True)
        
        return normalized


class MetadataQualityRanker(BaseRetriever):
    """
    Baseline: Pure metadata quality-based ranking.
    
    Ranks by metadata quality factors without fuzzy inference:
    - Recency
    - Completeness
    - Resource availability
    
    Uses simple weighted sum instead of fuzzy rules.
    """
    
    name = "metadata_quality"
    
    # Weights for metadata factors
    WEIGHT_RELEVANCE = 0.5  # Keyword match
    WEIGHT_RECENCY = 0.2
    WEIGHT_COMPLETENESS = 0.15
    WEIGHT_RESOURCES = 0.15
    
    def _compute_recency_score(self, days: int) -> float:
        """Linear recency score (newer = higher)."""
        if days <= 0:
            return 1.0
        elif days <= 30:
            return 0.95
        elif days <= 180:
            return 0.8
        elif days <= 365:
            return 0.6
        elif days <= 730:
            return 0.4
        else:
            return 0.2
    
    def _compute_completeness_score(self, dataset: Dict) -> float:
        """Simple completeness check."""
        checks = [
            bool(dataset.get('title')),
            bool(dataset.get('notes') or dataset.get('description')),
            bool(dataset.get('organization')),
            len(dataset.get('tags', [])) >= 1,
            len(dataset.get('groups', [])) >= 1,
            bool(dataset.get('license_id')),
        ]
        return sum(checks) / len(checks)
    
    def _compute_resource_score(self, dataset: Dict) -> float:
        """Score based on resource count."""
        num_resources = len(dataset.get('resources', []))
        if num_resources >= 5:
            return 1.0
        elif num_resources >= 3:
            return 0.8
        elif num_resources >= 1:
            return 0.5
        else:
            return 0.0
    
    def search(self, query: str, num_results: int = 100) -> List[Tuple[str, float]]:
        # Get candidates
        params = {'q': query, 'rows': num_results}
        
        try:
            resp = requests.get(f"{self.BASE_URL}/package_search", params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json()['result']['results']
        except Exception as e:
            print(f"Search error: {e}")
            return []
        
        if not results:
            return []
        
        # Compute relevance scores (position-based from portal)
        relevance_scores = {ds['id']: 1.0 - (i / (len(results) + 1)) 
                          for i, ds in enumerate(results)}
        
        # Score each dataset
        from datetime import datetime
        now = datetime.now()
        
        scored_results = []
        
        for ds in results:
            # Relevance from portal ranking
            relevance = relevance_scores[ds['id']]
            
            # Recency
            modified = ds.get('metadata_modified', '')
            try:
                if modified:
                    mod_date = datetime.strptime(modified.split('T')[0], '%Y-%m-%d')
                    days = (now - mod_date).days
                else:
                    days = 1000
            except:
                days = 1000
            recency = self._compute_recency_score(days)
            
            # Completeness
            completeness = self._compute_completeness_score(ds)
            
            # Resources
            resources = self._compute_resource_score(ds)
            
            # Weighted sum (no fuzzy inference)
            score = (
                self.WEIGHT_RELEVANCE * relevance +
                self.WEIGHT_RECENCY * recency +
                self.WEIGHT_COMPLETENESS * completeness +
                self.WEIGHT_RESOURCES * resources
            )
            
            scored_results.append((ds['id'], score))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results


class FuzzyRetriever(BaseRetriever):
    """
    Our approach: Fuzzy inference-based ranking.
    
    Uses calibrated fuzzy membership functions and rule base
    to compute relevance scores.
    """
    
    name = "fuzzy_hcir"
    
    def __init__(self):
        # Load calibrated variables
        from code.fuzzy_system.calibrated_variables import (
            RECENCY_CALIBRATED,
            COMPLETENESS_CALIBRATED,
            RESOURCE_AVAILABILITY_CALIBRATED
        )
        self.recency_var = RECENCY_CALIBRATED
        self.completeness_var = COMPLETENESS_CALIBRATED
        self.resources_var = RESOURCE_AVAILABILITY_CALIBRATED
    
    def _triangular_mf(self, x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function."""
        if x <= a or x >= c:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a + 1e-10)
        else:
            return (c - x) / (c - b + 1e-10)
    
    def _trapezoidal_mf(self, x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal membership function."""
        if x <= a or x >= d:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a + 1e-10)
        elif x <= c:
            return 1.0
        else:
            return (d - x) / (d - c + 1e-10)
    
    def _compute_membership(self, value: float, term_def: Dict) -> float:
        """Compute membership degree for a term."""
        mf_type = term_def['type']
        params = term_def['params']
        
        if mf_type == 'triangular':
            return self._triangular_mf(value, *params)
        elif mf_type == 'trapezoidal':
            return self._trapezoidal_mf(value, *params)
        else:
            return 0.0
    
    def _fuzzy_inference(self, recency_days: float, completeness: float, 
                        resources: int, relevance: float) -> float:
        """
        Apply fuzzy rules to compute final score.
        
        Simplified rule base:
        - IF relevant AND recent AND complete THEN excellent
        - IF relevant AND (recent OR complete) THEN good
        - IF somewhat_relevant AND complete THEN moderate
        - etc.
        """
        # Compute memberships for recency
        recency_memberships = {}
        for term, defn in self.recency_var.terms.items():
            recency_memberships[term] = self._compute_membership(recency_days, defn)
        
        # Compute memberships for completeness
        completeness_memberships = {}
        for term, defn in self.completeness_var.terms.items():
            completeness_memberships[term] = self._compute_membership(completeness, defn)
        
        # Compute memberships for resources
        resource_memberships = {}
        for term, defn in self.resources_var.terms.items():
            resource_memberships[term] = self._compute_membership(float(resources), defn)
        
        # Fuzzy rules (simplified)
        rules = []
        
        # Rule 1: Very recent + high completeness + rich resources -> excellent
        rule1 = min(
            recency_memberships.get('very_recent', 0),
            completeness_memberships.get('high', 0),
            resource_memberships.get('rich', 0)
        )
        rules.append(('excellent', rule1))
        
        # Rule 2: Recent + medium completeness -> good
        rule2 = min(
            recency_memberships.get('recent', 0),
            completeness_memberships.get('medium', 0)
        )
        rules.append(('good', rule2))
        
        # Rule 3: Moderate recency + partial completeness -> moderate
        rule3 = min(
            recency_memberships.get('moderate', 0),
            completeness_memberships.get('partial', 0)
        )
        rules.append(('moderate', rule3))
        
        # Rule 4: Old or low completeness -> low
        rule4 = max(
            recency_memberships.get('old', 0),
            completeness_memberships.get('low', 0)
        )
        rules.append(('low', rule4))
        
        # Rule 5: Very old -> very low
        rule5 = recency_memberships.get('very_old', 0)
        rules.append(('very_low', rule5))
        
        # Combine with relevance (keyword match) as weight
        # More relevant documents get higher scores
        
        # Defuzzification (simplified centroid)
        output_centroids = {
            'excellent': 0.95,
            'good': 0.75,
            'moderate': 0.50,
            'low': 0.25,
            'very_low': 0.10
        }
        
        numerator = 0.0
        denominator = 0.0
        
        for term, strength in rules:
            weighted_strength = strength * relevance  # Weight by keyword relevance
            numerator += weighted_strength * output_centroids[term]
            denominator += weighted_strength
        
        if denominator == 0:
            return relevance * 0.5  # Default to half relevance
        
        return numerator / denominator
    
    def search(self, query: str, num_results: int = 100) -> List[Tuple[str, float]]:
        # Get candidates from portal
        params = {'q': query, 'rows': num_results}
        
        try:
            resp = requests.get(f"{self.BASE_URL}/package_search", params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json()['result']['results']
        except Exception as e:
            print(f"Search error: {e}")
            return []
        
        if not results:
            return []
        
        # Compute relevance scores from portal ranking
        relevance_scores = {ds['id']: 1.0 - (i / (len(results) + 1)) 
                          for i, ds in enumerate(results)}
        
        from datetime import datetime
        now = datetime.now()
        
        scored_results = []
        
        for ds in results:
            # Get metadata values
            modified = ds.get('metadata_modified', '')
            try:
                if modified:
                    mod_date = datetime.strptime(modified.split('T')[0], '%Y-%m-%d')
                    recency_days = (now - mod_date).days
                else:
                    recency_days = 1000
            except:
                recency_days = 1000
            
            # Completeness
            checks = [
                bool(ds.get('title')),
                bool(ds.get('notes') or ds.get('description')),
                bool(ds.get('organization')),
                len(ds.get('tags', [])) >= 1,
                len(ds.get('groups', [])) >= 1,
                bool(ds.get('license_id')),
            ]
            completeness = sum(checks) / len(checks)
            
            # Resources
            num_resources = len(ds.get('resources', []))
            
            # Portal relevance
            relevance = relevance_scores[ds['id']]
            
            # Fuzzy inference
            score = self._fuzzy_inference(recency_days, completeness, num_resources, relevance)
            
            scored_results.append((ds['id'], score))
        
        # Sort by fuzzy score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Runs the full comparative experiment.
    """
    
    def __init__(self, ground_truth_file: str = "evaluation/ground_truth_auto.json"):
        self.ground_truth = {}
        self.systems = {}
        self.engine = EvaluationEngine()
        
        # Load ground truth
        gt_path = Path(ground_truth_file)
        if gt_path.exists():
            with open(gt_path, 'r', encoding='utf-8') as f:
                self.ground_truth = json.load(f)
            print(f"Loaded ground truth for {len(self.ground_truth)} queries")
    
    def add_system(self, system: BaseRetriever):
        """Add a retrieval system to evaluate."""
        self.systems[system.name] = system
    
    def run_experiment(self) -> Dict[str, Dict[str, float]]:
        """
        Run full experiment comparing all systems.
        
        Returns:
            Dictionary: system_name -> metric_name -> mean_value
        """
        print("=" * 70)
        print("RUNNING COMPARATIVE EXPERIMENT")
        print("=" * 70)
        
        # Prepare evaluation queries
        for query_id, data in self.ground_truth.items():
            query_info = data['query']
            judgments = data['judgments']
            
            eval_query = EvaluationQuery(
                query_id=query_id,
                query_text=query_info['query_text'],
                query_language=query_info.get('query_language', 'de'),
                domain=query_info.get('domain', ''),
                intent=query_info.get('intent', ''),
                ground_truth=[
                    RelevanceJudgment(
                        query_id=query_id,
                        dataset_id=j['dataset_id'],
                        relevance=j['relevance']
                    )
                    for j in judgments
                ]
            )
            
            self.engine.add_query(eval_query)
        
        # Run each system
        for system_name, system in self.systems.items():
            print(f"\nEvaluating: {system_name}")
            
            for query_id, data in self.ground_truth.items():
                query_text = data['query']['query_text']
                
                # Run search
                start_time = time.time()
                results = system.search(query_text, num_results=50)
                exec_time = time.time() - start_time
                
                # Create result object
                result = RankingResult(
                    system_name=system_name,
                    query_id=query_id,
                    ranked_docs=[doc_id for doc_id, score in results],
                    scores=[score for doc_id, score in results],
                    execution_time=exec_time
                )
                
                self.engine.add_result(result)
                
                # Rate limiting
                time.sleep(0.3)
            
            print(f"  Completed {len(self.ground_truth)} queries")
        
        # Evaluate and aggregate
        results = self.engine.evaluate_all()
        
        return results
    
    def generate_report(self, output_file: str = "evaluation/experiment_results.json"):
        """Generate and save experiment report."""
        results = self.run_experiment()
        
        print(self.engine.generate_report())
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'systems': list(self.systems.keys()),
                'num_queries': len(self.ground_truth),
                'aggregate_metrics': results
            }, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        
        return results


def main():
    """Main entry point for experiment."""
    print("=" * 70)
    print("FUZZY OGD RETRIEVAL - EXPERIMENTAL EVALUATION")
    print("University of Fribourg, Human-IST Institute")
    print("=" * 70)
    
    # Initialize runner
    runner = ExperimentRunner()
    
    # Add systems to compare
    runner.add_system(PortalBaseline())
    runner.add_system(KeywordBaseline())
    runner.add_system(MetadataQualityRanker())
    runner.add_system(FuzzyRetriever())
    
    # Run experiment
    results = runner.generate_report()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
