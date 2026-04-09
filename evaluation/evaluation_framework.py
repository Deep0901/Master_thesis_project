"""
Evaluation Framework for Fuzzy OGD Retrieval System

This module implements research-grade evaluation metrics and methodology
for comparing the fuzzy ranking system against baselines.

Research Questions Addressed:
- RQ2: Does fuzzy ranking outperform keyword-based baseline?
- RQ4: How does fuzzy ranking compare to AI semantic search?

Evaluation Metrics Implemented:
1. Precision@K - Precision at rank K
2. Recall@K - Recall at rank K  
3. Mean Average Precision (MAP)
4. Normalized Discounted Cumulative Gain (nDCG)
5. Mean Reciprocal Rank (MRR)
6. F1@K - Harmonic mean of P@K and R@K

Ground Truth:
- Relevance judgments on graded scale: 0 (not relevant) to 3 (highly relevant)
- Based on NIST TREC evaluation methodology

Author: Deep Shukla
Thesis: Improving Access to Swiss OGD through Fuzzy HCIR
University of Fribourg, Human-IST Institute
"""

import json
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RelevanceJudgment:
    """
    Ground truth relevance judgment for a query-document pair.
    
    Graded relevance scale (NIST TREC style):
    0 - Not relevant (does not address information need)
    1 - Marginally relevant (tangentially related)
    2 - Relevant (addresses need but may not be ideal)
    3 - Highly relevant (directly addresses information need)
    """
    query_id: str
    dataset_id: str
    relevance: int  # 0, 1, 2, or 3
    annotator: str = ""
    notes: str = ""


@dataclass
class EvaluationQuery:
    """
    Evaluation query with associated metadata and ground truth.
    """
    query_id: str
    query_text: str
    query_language: str = "de"
    domain: str = ""  # e.g., "environment", "mobility"
    intent: str = ""  # What the user is trying to find
    expected_themes: List[str] = field(default_factory=list)
    ground_truth: List[RelevanceJudgment] = field(default_factory=list)
    
    @property
    def relevant_docs(self) -> Set[str]:
        """Get IDs of all documents with relevance > 0."""
        return {j.dataset_id for j in self.ground_truth if j.relevance > 0}
    
    @property
    def highly_relevant_docs(self) -> Set[str]:
        """Get IDs of highly relevant documents (relevance >= 2)."""
        return {j.dataset_id for j in self.ground_truth if j.relevance >= 2}


@dataclass
class RankingResult:
    """
    Ranked list from a retrieval system.
    """
    system_name: str
    query_id: str
    ranked_docs: List[str]  # Ordered list of dataset IDs
    scores: List[float]  # Corresponding scores
    execution_time: float = 0.0


@dataclass
class EvaluationMetrics:
    """
    Complete evaluation metrics for a single query.
    """
    query_id: str
    system_name: str
    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    recall_at_10: float
    recall_at_20: float
    average_precision: float
    ndcg_at_10: float
    ndcg_at_20: float
    reciprocal_rank: float
    f1_at_10: float


# =============================================================================
# EVALUATION METRICS
# =============================================================================

class IRMetrics:
    """
    Information Retrieval evaluation metrics.
    
    Based on standard IR evaluation methodology:
    - Manning, Raghavan & Schutze (2008): "Information Retrieval"
    - NIST TREC evaluation methodology
    """
    
    @staticmethod
    def precision_at_k(ranked_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Precision at rank K.
        
        P@K = |relevant docs in top K| / K
        
        Args:
            ranked_docs: Ordered list of document IDs
            relevant_docs: Set of relevant document IDs
            k: Rank cutoff
            
        Returns:
            Precision value in [0, 1]
        """
        if k == 0:
            return 0.0
        
        top_k = set(ranked_docs[:k])
        relevant_in_top_k = len(top_k & relevant_docs)
        
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(ranked_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Recall at rank K.
        
        R@K = |relevant docs in top K| / |relevant docs|
        
        Args:
            ranked_docs: Ordered list of document IDs
            relevant_docs: Set of relevant document IDs
            k: Rank cutoff
            
        Returns:
            Recall value in [0, 1]
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        top_k = set(ranked_docs[:k])
        relevant_in_top_k = len(top_k & relevant_docs)
        
        return relevant_in_top_k / len(relevant_docs)
    
    @staticmethod
    def f1_at_k(ranked_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        F1 score at rank K (harmonic mean of P@K and R@K).
        """
        p = IRMetrics.precision_at_k(ranked_docs, relevant_docs, k)
        r = IRMetrics.recall_at_k(ranked_docs, relevant_docs, k)
        
        if p + r == 0:
            return 0.0
        
        return 2 * p * r / (p + r)
    
    @staticmethod
    def average_precision(ranked_docs: List[str], relevant_docs: Set[str]) -> float:
        """
        Average Precision (AP).
        
        AP = (1/|relevant|) * sum_{k: doc_k is relevant} P@k
        
        This is the area under the precision-recall curve.
        
        Args:
            ranked_docs: Ordered list of document IDs
            relevant_docs: Set of relevant document IDs
            
        Returns:
            AP value in [0, 1]
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        sum_precisions = 0.0
        relevant_found = 0
        
        for k, doc_id in enumerate(ranked_docs, 1):
            if doc_id in relevant_docs:
                relevant_found += 1
                precision_at_k = relevant_found / k
                sum_precisions += precision_at_k
        
        return sum_precisions / len(relevant_docs)
    
    @staticmethod
    def reciprocal_rank(ranked_docs: List[str], relevant_docs: Set[str]) -> float:
        """
        Reciprocal Rank (RR).
        
        RR = 1 / rank of first relevant document
        
        Args:
            ranked_docs: Ordered list of document IDs
            relevant_docs: Set of relevant document IDs
            
        Returns:
            RR value in (0, 1] or 0 if no relevant doc found
        """
        for rank, doc_id in enumerate(ranked_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def dcg_at_k(ranked_docs: List[str], relevance_map: Dict[str, int], k: int) -> float:
        """
        Discounted Cumulative Gain at rank K.
        
        DCG@K = sum_{i=1}^{K} (2^{rel_i} - 1) / log2(i + 1)
        
        Args:
            ranked_docs: Ordered list of document IDs
            relevance_map: Mapping from doc_id to relevance grade
            k: Rank cutoff
            
        Returns:
            DCG value
        """
        dcg = 0.0
        
        for i, doc_id in enumerate(ranked_docs[:k], 1):
            rel = relevance_map.get(doc_id, 0)
            gain = (2 ** rel - 1) / math.log2(i + 1)
            dcg += gain
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(ranked_docs: List[str], relevance_map: Dict[str, int], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain at rank K.
        
        nDCG@K = DCG@K / IDCG@K
        
        where IDCG is the DCG of an ideal ranking.
        
        Args:
            ranked_docs: Ordered list of document IDs
            relevance_map: Mapping from doc_id to relevance grade
            k: Rank cutoff
            
        Returns:
            nDCG value in [0, 1]
        """
        dcg = IRMetrics.dcg_at_k(ranked_docs, relevance_map, k)
        
        # Compute ideal ranking (sorted by relevance)
        ideal_ranking = sorted(relevance_map.keys(), 
                              key=lambda x: relevance_map[x], 
                              reverse=True)
        idcg = IRMetrics.dcg_at_k(ideal_ranking, relevance_map, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg


# =============================================================================
# EVALUATION ENGINE
# =============================================================================

class EvaluationEngine:
    """
    Main evaluation engine for comparing retrieval systems.
    
    Supports:
    - Multiple systems comparison
    - Per-query and aggregate metrics
    - Statistical significance testing
    """
    
    def __init__(self):
        self.queries: Dict[str, EvaluationQuery] = {}
        self.results: Dict[str, Dict[str, RankingResult]] = defaultdict(dict)
        self.metrics: Dict[str, Dict[str, EvaluationMetrics]] = defaultdict(dict)
    
    def add_query(self, query: EvaluationQuery):
        """Add an evaluation query with ground truth."""
        self.queries[query.query_id] = query
    
    def add_result(self, result: RankingResult):
        """Add ranking results from a system for a query."""
        self.results[result.query_id][result.system_name] = result
    
    def evaluate_query(self, query_id: str, system_name: str) -> EvaluationMetrics:
        """
        Compute all metrics for one query-system pair.
        """
        query = self.queries[query_id]
        result = self.results[query_id][system_name]
        
        relevant = query.relevant_docs
        ranked = result.ranked_docs
        
        # Build relevance map for nDCG
        relevance_map = {j.dataset_id: j.relevance for j in query.ground_truth}
        
        metrics = EvaluationMetrics(
            query_id=query_id,
            system_name=system_name,
            precision_at_5=IRMetrics.precision_at_k(ranked, relevant, 5),
            precision_at_10=IRMetrics.precision_at_k(ranked, relevant, 10),
            precision_at_20=IRMetrics.precision_at_k(ranked, relevant, 20),
            recall_at_10=IRMetrics.recall_at_k(ranked, relevant, 10),
            recall_at_20=IRMetrics.recall_at_k(ranked, relevant, 20),
            average_precision=IRMetrics.average_precision(ranked, relevant),
            ndcg_at_10=IRMetrics.ndcg_at_k(ranked, relevance_map, 10),
            ndcg_at_20=IRMetrics.ndcg_at_k(ranked, relevance_map, 20),
            reciprocal_rank=IRMetrics.reciprocal_rank(ranked, relevant),
            f1_at_10=IRMetrics.f1_at_k(ranked, relevant, 10)
        )
        
        self.metrics[query_id][system_name] = metrics
        return metrics
    
    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all systems on all queries.
        
        Returns:
            Dictionary: system_name -> metric_name -> mean_value
        """
        for query_id in self.queries:
            for system_name in self.results.get(query_id, {}):
                self.evaluate_query(query_id, system_name)
        
        return self.aggregate_metrics()
    
    def aggregate_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute aggregate metrics across all queries.
        """
        aggregates = defaultdict(lambda: defaultdict(list))
        
        for query_id, systems in self.metrics.items():
            for system_name, m in systems.items():
                aggregates[system_name]['MAP'].append(m.average_precision)
                aggregates[system_name]['P@5'].append(m.precision_at_5)
                aggregates[system_name]['P@10'].append(m.precision_at_10)
                aggregates[system_name]['P@20'].append(m.precision_at_20)
                aggregates[system_name]['R@10'].append(m.recall_at_10)
                aggregates[system_name]['R@20'].append(m.recall_at_20)
                aggregates[system_name]['nDCG@10'].append(m.ndcg_at_10)
                aggregates[system_name]['nDCG@20'].append(m.ndcg_at_20)
                aggregates[system_name]['MRR'].append(m.reciprocal_rank)
                aggregates[system_name]['F1@10'].append(m.f1_at_10)
        
        # Compute means
        results = {}
        for system_name, metrics in aggregates.items():
            results[system_name] = {
                metric: np.mean(values) for metric, values in metrics.items()
            }
        
        return results
    
    def statistical_significance(self, system_a: str, system_b: str, 
                                 metric: str = 'average_precision') -> Tuple[float, float]:
        """
        Paired t-test for statistical significance between two systems.
        
        Returns:
            (t_statistic, p_value)
        """
        from scipy import stats
        
        scores_a = []
        scores_b = []
        
        for query_id in self.queries:
            if system_a in self.metrics.get(query_id, {}) and \
               system_b in self.metrics.get(query_id, {}):
                m_a = self.metrics[query_id][system_a]
                m_b = self.metrics[query_id][system_b]
                
                scores_a.append(getattr(m_a, metric))
                scores_b.append(getattr(m_b, metric))
        
        if len(scores_a) < 2:
            return 0.0, 1.0
        
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        return t_stat, p_value
    
    def generate_report(self) -> str:
        """Generate human-readable evaluation report."""
        results = self.aggregate_metrics()
        
        lines = [
            "=" * 80,
            "RETRIEVAL EVALUATION REPORT",
            "=" * 80,
            f"\nNumber of queries: {len(self.queries)}",
            f"Systems evaluated: {list(results.keys())}",
            ""
        ]
        
        # Per-system metrics
        for system_name, metrics in results.items():
            lines.append(f"\n{system_name.upper()}")
            lines.append("-" * 40)
            for metric, value in sorted(metrics.items()):
                lines.append(f"  {metric}: {value:.4f}")
        
        # Comparison table
        if len(results) > 1:
            lines.append("\n\nCOMPARATIVE TABLE")
            lines.append("-" * 80)
            
            headers = ['Metric'] + list(results.keys())
            lines.append("  " + "\t".join(f"{h:12}" for h in headers))
            
            all_metrics = list(next(iter(results.values())).keys())
            for metric in sorted(all_metrics):
                row = [metric] + [f"{results[s].get(metric, 0):.4f}" for s in results]
                lines.append("  " + "\t".join(f"{v:12}" for v in row))
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)


# =============================================================================
# BENCHMARK QUERY SET
# =============================================================================

# Research-grade evaluation queries for Environment and Mobility domains
# Based on realistic Swiss OGD use cases

BENCHMARK_QUERIES = [
    # Environment Domain (5 queries)
    EvaluationQuery(
        query_id="ENV-01",
        query_text="Aktuelle Luftqualitätsdaten für Schweizer Städte",
        query_language="de",
        domain="environment",
        intent="Find current air quality measurements for Swiss cities",
        expected_themes=["envi", "heal"]
    ),
    EvaluationQuery(
        query_id="ENV-02", 
        query_text="Biodiversität und Artenschutz in der Schweiz",
        query_language="de",
        domain="environment",
        intent="Find datasets about biodiversity and species protection",
        expected_themes=["envi", "agri"]
    ),
    EvaluationQuery(
        query_id="ENV-03",
        query_text="Wasserqualität in Schweizer Seen und Flüssen",
        query_language="de",
        domain="environment", 
        intent="Find water quality monitoring data",
        expected_themes=["envi"]
    ),
    EvaluationQuery(
        query_id="ENV-04",
        query_text="Lärmbelastung und Lärmkarten Schweiz",
        query_language="de",
        domain="environment",
        intent="Find noise pollution maps and measurements",
        expected_themes=["envi", "regi"]
    ),
    EvaluationQuery(
        query_id="ENV-05",
        query_text="Klimawandel und CO2-Emissionen Schweiz",
        query_language="de",
        domain="environment",
        intent="Find climate change and emissions data",
        expected_themes=["envi", "ener"]
    ),
    
    # Mobility Domain (5 queries)
    EvaluationQuery(
        query_id="MOB-01",
        query_text="Öffentlicher Verkehr Fahrplandaten Schweiz",
        query_language="de",
        domain="mobility",
        intent="Find public transport timetable/schedule data",
        expected_themes=["mobi", "regi"]
    ),
    EvaluationQuery(
        query_id="MOB-02",
        query_text="Verkehrsunfälle Statistik Schweiz",
        query_language="de",
        domain="mobility",
        intent="Find traffic accident statistics",
        expected_themes=["mobi", "just"]
    ),
    EvaluationQuery(
        query_id="MOB-03",
        query_text="Elektromobilität Ladestationen Schweiz",
        query_language="de",
        domain="mobility",
        intent="Find electric vehicle charging station locations",
        expected_themes=["mobi", "ener"]
    ),
    EvaluationQuery(
        query_id="MOB-04",
        query_text="Veloverkehr und Radwege in der Schweiz",
        query_language="de",
        domain="mobility",
        intent="Find bicycle traffic and cycling infrastructure data",
        expected_themes=["mobi", "regi"]
    ),
    EvaluationQuery(
        query_id="MOB-05",
        query_text="Parkplätze und Parkhäuser Echtzeitdaten",
        query_language="de",
        domain="mobility",
        intent="Find real-time parking availability data",
        expected_themes=["mobi"]
    ),
    
    # Cross-domain queries (5 queries)  
    EvaluationQuery(
        query_id="XD-01",
        query_text="Bevölkerungsentwicklung und Demografie Schweiz",
        query_language="de",
        domain="cross",
        intent="Find population and demographic statistics",
        expected_themes=["soci", "regi"]
    ),
    EvaluationQuery(
        query_id="XD-02",
        query_text="Energieverbrauch nach Kantonen",
        query_language="de",
        domain="cross",
        intent="Find energy consumption by canton",
        expected_themes=["ener", "regi"]
    ),
    EvaluationQuery(
        query_id="XD-03",
        query_text="Bildungsstatistik Schulen und Universitäten",
        query_language="de",
        domain="cross",
        intent="Find education statistics",
        expected_themes=["educ"]
    ),
    EvaluationQuery(
        query_id="XD-04",
        query_text="Gesundheitsversorgung und Spitäler",
        query_language="de",
        domain="cross",
        intent="Find healthcare facilities and statistics",
        expected_themes=["heal"]
    ),
    EvaluationQuery(
        query_id="XD-05",
        query_text="Wirtschaftsstatistik und BIP nach Region",
        query_language="de",
        domain="cross",
        intent="Find economic statistics by region",
        expected_themes=["econ", "regi"]
    )
]


def save_benchmark_queries(filepath: str = "evaluation/benchmark_queries_v2.json"):
    """Save benchmark queries to JSON file for annotation."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    queries_data = []
    for q in BENCHMARK_QUERIES:
        queries_data.append({
            'query_id': q.query_id,
            'query_text': q.query_text,
            'query_language': q.query_language,
            'domain': q.domain,
            'intent': q.intent,
            'expected_themes': q.expected_themes,
            'ground_truth': []  # To be filled during annotation
        })
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(queries_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(queries_data)} benchmark queries to {filepath}")


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("EVALUATION FRAMEWORK FOR FUZZY OGD RETRIEVAL")
    print("=" * 70)
    
    # Save benchmark queries
    save_benchmark_queries()
    
    # Example evaluation
    engine = EvaluationEngine()
    
    # Add sample query
    sample_query = EvaluationQuery(
        query_id="TEST-01",
        query_text="Luftqualität Zürich",
        domain="environment",
        intent="Find air quality data for Zurich",
        ground_truth=[
            RelevanceJudgment("TEST-01", "ds-001", 3),  # Highly relevant
            RelevanceJudgment("TEST-01", "ds-002", 2),  # Relevant
            RelevanceJudgment("TEST-01", "ds-003", 1),  # Marginally
            RelevanceJudgment("TEST-01", "ds-004", 0),  # Not relevant
        ]
    )
    engine.add_query(sample_query)
    
    # Sample results from two systems
    fuzzy_result = RankingResult(
        system_name="fuzzy",
        query_id="TEST-01",
        ranked_docs=["ds-001", "ds-002", "ds-003", "ds-004"],  # Perfect ranking
        scores=[0.95, 0.80, 0.60, 0.30]
    )
    
    baseline_result = RankingResult(
        system_name="keyword_baseline",
        query_id="TEST-01",
        ranked_docs=["ds-003", "ds-001", "ds-004", "ds-002"],  # Imperfect
        scores=[0.90, 0.85, 0.70, 0.50]
    )
    
    engine.add_result(fuzzy_result)
    engine.add_result(baseline_result)
    
    # Evaluate
    agg = engine.evaluate_all()
    
    print(engine.generate_report())
    
    print("\nBenchmark queries saved to evaluation/benchmark_queries_v2.json")
    print("Use these queries to collect ground truth relevance judgments.")
