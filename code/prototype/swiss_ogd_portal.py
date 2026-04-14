#!/usr/bin/env python3
"""
Swiss Open Government Data Portal - Fuzzy HCIR Search System

A professional research prototype for the Master Thesis:
"Improving Access to Swiss Open Government Data through Fuzzy Human-Centered Information Retrieval"

Author: Deep Shukla
Institution: Human-IST Institute, University of Fribourg
Supervisor: Janick Spycher | Examiner: Prof. Dr. Edy Portmann

Features:
- Live integration with opendata.swiss CKAN API
- Data-driven fuzzy ranking with Mamdani inference
- Explainable results with transparency badges
- Multilingual support (DE, FR, IT, EN)
- Comparison mode (Portal vs. BM25 vs. Fuzzy)
- Professional Swiss government styling

Run: streamlit run code/prototype/swiss_ogd_portal.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import json
import re
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

# ============================================================================
# PAGE CONFIGURATION - Swiss Government Design
# ============================================================================

st.set_page_config(
    page_title="opendata.swiss | Fuzzy Search Research",
    page_icon="🇨🇭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://handbook.opendata.swiss/',
        'Report a bug': None,
        'About': """
        ## Swiss OGD Fuzzy HCIR Research Prototype
        
        Master Thesis Project - University of Fribourg
        
        **Research Questions:**
        - RQ1: Fuzzy modeling of vague query intent
        - RQ2: Comparison with keyword baselines
        - RQ3: Explainability and user trust
        - RQ4: Advantages over AI/semantic approaches
        """
    }
)

# ============================================================================
# SWISS GOVERNMENT DESIGN SYSTEM CSS
# ============================================================================

SWISS_GOV_CSS = """
<style>
/* Swiss Confederation Design System Colors */
:root {
    --swiss-red: #D8232A;
    --swiss-dark: #333333;
    --swiss-gray: #757575;
    --swiss-light-gray: #F5F5F5;
    --swiss-blue: #006699;
    --swiss-green: #4A9F35;
    --swiss-orange: #F29400;
    --swiss-white: #FFFFFF;
}

/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Swiss Header Banner */
.swiss-header {
    background: linear-gradient(90deg, var(--swiss-red) 0%, #B31B21 100%);
    color: white;
    padding: 20px 30px;
    margin: -1rem -1rem 2rem -1rem;
    border-radius: 0;
}

.swiss-header h1 {
    color: white !important;
    font-size: 1.8rem !important;
    margin: 0 !important;
    font-weight: 700 !important;
}

.swiss-header p {
    color: rgba(255,255,255,0.9);
    margin: 5px 0 0 0;
    font-size: 0.95rem;
}

/* Swiss Cross Logo */
.swiss-cross {
    display: inline-block;
    width: 32px;
    height: 32px;
    background: white;
    position: relative;
    margin-right: 15px;
    vertical-align: middle;
}

.swiss-cross:before {
    content: '';
    position: absolute;
    background: var(--swiss-red);
    top: 25%;
    left: 8%;
    width: 84%;
    height: 50%;
}

.swiss-cross:after {
    content: '';
    position: absolute;
    background: var(--swiss-red);
    top: 8%;
    left: 25%;
    width: 50%;
    height: 84%;
}

/* Search Box */
.search-container {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 25px;
    margin-bottom: 25px;
}

/* Result Cards */
.result-card {
    background: white;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 15px;
    transition: box-shadow 0.2s ease;
    border-left: 4px solid var(--swiss-blue);
}

.result-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.result-card.high-relevance {
    border-left-color: var(--swiss-green);
}

.result-card.medium-relevance {
    border-left-color: var(--swiss-orange);
}

.result-card.low-relevance {
    border-left-color: var(--swiss-red);
}

/* Score Badges */
.score-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    margin-right: 8px;
}

.score-excellent {
    background: linear-gradient(135deg, #4A9F35, #3D8A2D);
    color: white;
}

.score-good {
    background: linear-gradient(135deg, #006699, #005580);
    color: white;
}

.score-moderate {
    background: linear-gradient(135deg, #F29400, #D98500);
    color: white;
}

.score-low {
    background: linear-gradient(135deg, #D8232A, #B81D23);
    color: white;
}

/* Factor Indicators */
.factor-bar {
    height: 8px;
    border-radius: 4px;
    background: #E0E0E0;
    margin: 5px 0;
    overflow: hidden;
}

.factor-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}

.factor-fill.recency { background: linear-gradient(90deg, #4A9F35, #6BC850); }
.factor-fill.completeness { background: linear-gradient(90deg, #006699, #0088CC); }
.factor-fill.resources { background: linear-gradient(90deg, #9B59B6, #B370CF); }
.factor-fill.similarity { background: linear-gradient(90deg, #F29400, #FFB340); }

/* Explanation Box */
.explanation-box {
    background: linear-gradient(135deg, #F8F9FA, #FFFFFF);
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
    font-size: 0.9rem;
}

.explanation-title {
    font-weight: 600;
    color: var(--swiss-dark);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
}

.explanation-title svg {
    margin-right: 8px;
}

/* Tag Pills */
.tag-pill {
    display: inline-block;
    background: #E8F4F8;
    color: var(--swiss-blue);
    padding: 4px 10px;
    border-radius: 15px;
    font-size: 0.8rem;
    margin: 2px;
}

/* Organization Badge */
.org-badge {
    display: inline-flex;
    align-items: center;
    color: var(--swiss-gray);
    font-size: 0.85rem;
}

.org-badge svg {
    margin-right: 5px;
}

/* Format Badges */
.format-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 4px;
}

.format-csv { background: #E3F2FD; color: #1565C0; }
.format-json { background: #FFF3E0; color: #E65100; }
.format-xml { background: #F3E5F5; color: #7B1FA2; }
.format-pdf { background: #FFEBEE; color: #C62828; }
.format-api { background: #E8F5E9; color: #2E7D32; }
.format-geojson { background: #E0F2F1; color: #00695C; }
.format-xlsx { background: #E8EAF6; color: #303F9F; }
.format-default { background: #ECEFF1; color: #546E7A; }

/* Comparison Table */
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
}

.comparison-table th {
    background: var(--swiss-light-gray);
    padding: 12px;
    text-align: left;
    font-weight: 600;
}

.comparison-table td {
    padding: 12px;
    border-bottom: 1px solid #E0E0E0;
}

/* Metrics Display */
.metric-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    border: 1px solid #E0E0E0;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--swiss-blue);
}

.metric-label {
    color: var(--swiss-gray);
    font-size: 0.85rem;
    margin-top: 5px;
}

/* Sidebar Styling */
.sidebar .sidebar-content {
    background: var(--swiss-light-gray);
}

/* Footer */
.swiss-footer {
    background: var(--swiss-dark);
    color: white;
    padding: 20px;
    margin-top: 40px;
    text-align: center;
    font-size: 0.85rem;
}

/* Loading Animation */
.loading-dots {
    display: inline-flex;
    gap: 4px;
}

.loading-dots span {
    width: 8px;
    height: 8px;
    background: var(--swiss-red);
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
}

.loading-dots span:nth-child(1) { animation-delay: -0.32s; }
.loading-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}

/* Responsive Design */
@media (max-width: 768px) {
    .swiss-header h1 { font-size: 1.4rem !important; }
    .result-card { padding: 15px; }
}
</style>
"""

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FuzzyMembership:
    """Fuzzy membership values for a linguistic variable."""
    variable: str
    crisp_value: float
    memberships: Dict[str, float] = field(default_factory=dict)
    
    @property
    def dominant_term(self) -> Tuple[str, float]:
        if not self.memberships:
            return ("unknown", 0.0)
        return max(self.memberships.items(), key=lambda x: x[1])


@dataclass
class RankingFactors:
    """Breakdown of ranking factors for a dataset."""
    recency_score: float
    completeness_score: float
    resource_score: float
    similarity_score: float
    fuzzy_relevance: float
    
    recency_term: str = ""
    completeness_term: str = ""
    resource_term: str = ""
    similarity_term: str = ""


@dataclass
class DatasetResult:
    """A ranked dataset with full metadata and explanation."""
    id: str
    title: Dict[str, str]
    description: str
    organization: str
    resources: List[Dict]
    themes: List[str]
    tags: List[str]
    modified: str
    created: str
    license: str
    url: str
    
    rank: int = 0
    relevance_score: float = 0.0
    factors: Optional[RankingFactors] = None
    explanation: str = ""
    
    @property
    def days_since_modified(self) -> int:
        try:
            mod_date = datetime.fromisoformat(self.modified.replace('Z', '+00:00'))
            return (datetime.now(mod_date.tzinfo) - mod_date).days
        except:
            return 365
    
    @property
    def format_list(self) -> List[str]:
        formats = []
        for r in self.resources:
            fmt = r.get('format', 'Unknown').upper()
            if fmt and fmt not in formats:
                formats.append(fmt)
        return formats[:5]


# ============================================================================
# OPENDATA.SWISS CKAN API CLIENT
# ============================================================================

class OpenDataSwissClient:
    """
    Client for the opendata.swiss CKAN API.
    
    Handles all communication with the Swiss national OGD portal,
    including rate limiting and error handling.
    """
    
    BASE_URL = "https://opendata.swiss/api/3/action"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Swiss-OGD-Fuzzy-HCIR-Research/1.0'
        })
        self._last_request_time = 0
        self._min_request_interval = 0.2  # 200ms between requests
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def search(
        self,
        query: str,
        rows: int = 30,
        start: int = 0,
        fq: Optional[str] = None,
        sort: str = "score desc"
    ) -> Tuple[List[Dict], int]:
        """
        Search datasets on opendata.swiss.
        
        Args:
            query: Search query string
            rows: Number of results to return
            start: Offset for pagination
            fq: Filter query (e.g., "groups:environment")
            sort: Sort order
            
        Returns:
            Tuple of (results list, total count)
        """
        self._rate_limit()
        
        params = {
            'q': query,
            'rows': rows,
            'start': start,
            'sort': sort
        }
        if fq:
            params['fq'] = fq
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/package_search",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                results = data['result']['results']
                count = data['result']['count']
                return results, count
            else:
                return [], 0
                
        except requests.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return [], 0
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Get a single dataset by ID."""
        self._rate_limit()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/package_show",
                params={'id': dataset_id},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                return data['result']
            return None
            
        except requests.RequestException:
            return None
    
    def get_organizations(self) -> List[Dict]:
        """Get all organizations."""
        self._rate_limit()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/organization_list",
                params={'all_fields': True},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                return data['result']
            return []
            
        except requests.RequestException:
            return []
    
    def get_themes(self) -> List[Dict]:
        """Get all thematic categories (groups)."""
        self._rate_limit()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/group_list",
                params={'all_fields': True},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                return data['result']
            return []
            
        except requests.RequestException:
            return []


# ============================================================================
# FUZZY INFERENCE ENGINE (CALIBRATED FROM REAL DATA)
# ============================================================================

class CalibratedFuzzyEngine:
    """
    Mamdani Fuzzy Inference Engine calibrated from opendata.swiss statistics.
    
    Membership function parameters derived from actual portal data percentiles:
    - Recency: Based on days_since_modified distribution (mean=1149, median=776)
    - Completeness: Based on DCAT-AP CH field population (range 50-83%)
    - Resources: Based on resource count distribution (median=4, max=138)
    
    Reference: Zadeh (1965), Mamdani & Assilian (1975)
    """
    
    # Recency membership functions (days since modified)
    # Calibrated from percentile analysis: P10=2, P25=151, P50=776, P75=1583, P90=3559
    RECENCY_MF = {
        'very_recent': ('trimf', [0, 0, 30]),        # Fresh data (< 1 month)
        'recent': ('trapmf', [7, 30, 150, 365]),      # Recent (1-12 months)
        'moderate': ('trapmf', [180, 365, 776, 1200]), # Moderate (1-3 years)
        'old': ('trapmf', [776, 1200, 2500, 3500]),    # Old (2-10 years)
        'very_old': ('trimf', [2500, 3500, 4500])     # Very old (10+ years)
    }
    
    # Completeness membership functions (ratio 0-1)
    # Based on DCAT-AP CH field analysis: min=50%, max=83%
    COMPLETENESS_MF = {
        'low': ('trimf', [0, 0, 0.55]),
        'partial': ('trimf', [0.45, 0.60, 0.72]),
        'medium': ('trapmf', [0.65, 0.72, 0.78, 0.85]),
        'high': ('trapmf', [0.75, 0.83, 0.92, 0.97]),
        'complete': ('trimf', [0.90, 0.97, 1.0])
    }
    
    # Resource count membership functions
    # Calibrated from distribution: P25=2, P50=4, P75=6, P90=8, max=138
    RESOURCES_MF = {
        'minimal': ('trimf', [0, 1, 2]),
        'limited': ('trimf', [1, 2, 4]),
        'moderate': ('trimf', [2, 4, 6]),
        'rich': ('trapmf', [4, 6, 10, 15]),
        'comprehensive': ('trimf', [10, 20, 50])
    }
    
    # Thematic similarity membership functions
    SIMILARITY_MF = {
        'not_relevant': ('trimf', [0, 0, 0.15]),
        'somewhat_relevant': ('trimf', [0.10, 0.25, 0.40]),
        'relevant': ('trimf', [0.30, 0.50, 0.70]),
        'highly_relevant': ('trimf', [0.60, 0.78, 0.92]),
        'exact_match': ('trimf', [0.85, 0.95, 1.0])
    }
    
    # Output relevance membership functions
    RELEVANCE_MF = {
        'very_low': ('trimf', [0, 0, 0.25]),
        'low': ('trimf', [0.10, 0.28, 0.45]),
        'moderate': ('trimf', [0.35, 0.50, 0.65]),
        'good': ('trimf', [0.55, 0.72, 0.88]),
        'excellent': ('trimf', [0.80, 0.92, 1.0])
    }
    
    # Fuzzy rules (derived from domain knowledge and data analysis)
    RULES = [
        # High similarity + quality = excellent
        ({'similarity': 'exact_match', 'recency': 'very_recent', 'completeness': 'high'}, 'excellent'),
        ({'similarity': 'exact_match', 'recency': 'very_recent'}, 'excellent'),
        ({'similarity': 'exact_match', 'completeness': 'complete'}, 'excellent'),
        ({'similarity': 'highly_relevant', 'recency': 'recent', 'completeness': 'high'}, 'excellent'),
        
        # Good combinations
        ({'similarity': 'highly_relevant', 'recency': 'recent'}, 'good'),
        ({'similarity': 'highly_relevant', 'completeness': 'high'}, 'good'),
        ({'similarity': 'relevant', 'recency': 'very_recent', 'completeness': 'high'}, 'good'),
        ({'similarity': 'relevant', 'resources': 'comprehensive'}, 'good'),
        
        # Moderate combinations
        ({'similarity': 'relevant', 'completeness': 'medium'}, 'moderate'),
        ({'similarity': 'relevant', 'recency': 'moderate'}, 'moderate'),
        ({'similarity': 'somewhat_relevant', 'recency': 'very_recent'}, 'moderate'),
        ({'similarity': 'highly_relevant', 'recency': 'old'}, 'moderate'),
        
        # Quality degradation rules
        ({'recency': 'very_old'}, 'low'),
        ({'completeness': 'low'}, 'low'),
        ({'completeness': 'low', 'similarity': 'somewhat_relevant'}, 'very_low'),
        ({'similarity': 'not_relevant'}, 'very_low'),
        ({'resources': 'minimal', 'completeness': 'low'}, 'very_low'),
        
        # Combined quality boosters
        ({'recency': 'very_recent', 'completeness': 'complete', 'resources': 'rich'}, 'excellent'),
        ({'recency': 'recent', 'completeness': 'high', 'resources': 'moderate'}, 'good'),
        ({'recency': 'old', 'resources': 'comprehensive'}, 'moderate'),
    ]
    
    def __init__(self):
        """Initialize the fuzzy engine."""
        self.universe_points = 1000  # Resolution for fuzzy operations
    
    def _triangular_mf(self, x: float, params: List[float]) -> float:
        """Triangular membership function."""
        a, b, c = params
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:  # b < x < c
            return (c - x) / (c - b) if c != b else 1.0
    
    def _trapezoidal_mf(self, x: float, params: List[float]) -> float:
        """Trapezoidal membership function."""
        a, b, c, d = params
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b < x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c) if d != c else 1.0
    
    def _compute_membership(
        self,
        value: float,
        mf_type: str,
        params: List[float]
    ) -> float:
        """Compute membership degree for a value."""
        if mf_type == 'trimf':
            return self._triangular_mf(value, params)
        elif mf_type == 'trapmf':
            return self._trapezoidal_mf(value, params)
        return 0.0
    
    def fuzzify(
        self,
        variable: str,
        value: float
    ) -> FuzzyMembership:
        """
        Fuzzify a crisp input value.
        
        Args:
            variable: Variable name ('recency', 'completeness', 'resources', 'similarity')
            value: Crisp input value
            
        Returns:
            FuzzyMembership with degrees for each linguistic term
        """
        mf_dict = {
            'recency': self.RECENCY_MF,
            'completeness': self.COMPLETENESS_MF,
            'resources': self.RESOURCES_MF,
            'similarity': self.SIMILARITY_MF
        }
        
        if variable not in mf_dict:
            return FuzzyMembership(variable, value, {})
        
        memberships = {}
        for term, (mf_type, params) in mf_dict[variable].items():
            memberships[term] = self._compute_membership(value, mf_type, params)
        
        return FuzzyMembership(variable, value, memberships)
    
    def evaluate_rules(
        self,
        fuzzified_inputs: Dict[str, FuzzyMembership]
    ) -> Dict[str, float]:
        """
        Evaluate all fuzzy rules and aggregate output memberships.
        
        Uses Mamdani inference:
        - AND operator: minimum
        - Implication: minimum
        - Aggregation: maximum
        """
        output_memberships = defaultdict(float)
        
        for antecedents, consequent in self.RULES:
            # Compute rule firing strength (AND of all antecedents)
            firing_strength = 1.0
            
            for var, term in antecedents.items():
                if var in fuzzified_inputs:
                    membership = fuzzified_inputs[var].memberships.get(term, 0.0)
                    firing_strength = min(firing_strength, membership)
            
            # Aggregate to output (MAX)
            output_memberships[consequent] = max(
                output_memberships[consequent],
                firing_strength
            )
        
        return dict(output_memberships)
    
    def defuzzify(self, output_memberships: Dict[str, float]) -> float:
        """
        Defuzzify using centroid method.
        
        Creates aggregated output fuzzy set and computes centroid.
        """
        # Create universe
        x = np.linspace(0, 1, self.universe_points)
        aggregated = np.zeros(self.universe_points)
        
        # Aggregate all output membership functions
        for term, strength in output_memberships.items():
            if term in self.RELEVANCE_MF and strength > 0:
                mf_type, params = self.RELEVANCE_MF[term]
                for i, xi in enumerate(x):
                    mu = self._compute_membership(xi, mf_type, params)
                    # Implication (min) then Aggregation (max)
                    aggregated[i] = max(aggregated[i], min(mu, strength))
        
        # Centroid defuzzification
        if np.sum(aggregated) > 0:
            centroid = np.sum(x * aggregated) / np.sum(aggregated)
            return float(centroid)
        
        return 0.5  # Default if no rules fire
    
    def infer(
        self,
        recency_days: float,
        completeness: float,
        resource_count: int,
        similarity: float
    ) -> Tuple[float, Dict[str, FuzzyMembership]]:
        """
        Perform complete fuzzy inference.
        
        Args:
            recency_days: Days since last modification
            completeness: Metadata completeness ratio (0-1)
            resource_count: Number of resources
            similarity: Query-document similarity (0-1)
            
        Returns:
            Tuple of (relevance_score, fuzzified_inputs)
        """
        # Fuzzify inputs
        fuzzified = {
            'recency': self.fuzzify('recency', recency_days),
            'completeness': self.fuzzify('completeness', completeness),
            'resources': self.fuzzify('resources', resource_count),
            'similarity': self.fuzzify('similarity', similarity)
        }
        
        # Evaluate rules
        output_memberships = self.evaluate_rules(fuzzified)
        
        # Defuzzify
        relevance = self.defuzzify(output_memberships)
        
        return relevance, fuzzified


# ============================================================================
# METADATA QUALITY ANALYZER
# ============================================================================

class MetadataAnalyzer:
    """
    Analyzes dataset metadata quality according to DCAT-AP CH standards.
    
    Computes completeness scores based on presence and quality of
    required and recommended metadata fields.
    """
    
    # DCAT-AP CH required fields
    REQUIRED_FIELDS = [
        'title', 'description', 'publisher', 'contact_point',
        'theme', 'access_rights', 'issued'
    ]
    
    # DCAT-AP CH recommended fields
    RECOMMENDED_FIELDS = [
        'keyword', 'spatial', 'temporal', 'accrual_periodicity',
        'language', 'documentation', 'relation'
    ]
    
    # Field weights for completeness score
    FIELD_WEIGHTS = {
        'title': 1.0,
        'description': 1.0,
        'organization': 0.8,
        'resources': 0.9,
        'tags': 0.6,
        'groups': 0.7,
        'license': 0.5,
        'metadata_modified': 0.4
    }
    
    def compute_completeness(self, dataset: Dict) -> float:
        """
        Compute metadata completeness score.
        
        Args:
            dataset: Raw dataset dictionary from CKAN
            
        Returns:
            Completeness ratio (0-1)
        """
        total_weight = sum(self.FIELD_WEIGHTS.values())
        achieved_weight = 0.0
        
        for field, weight in self.FIELD_WEIGHTS.items():
            value = dataset.get(field)
            
            if field == 'title':
                # Check multilingual titles
                if isinstance(value, dict):
                    achieved_weight += weight if any(value.values()) else 0
                elif value:
                    achieved_weight += weight
                    
            elif field == 'description':
                if isinstance(value, dict):
                    desc = ' '.join(str(v) for v in value.values() if v)
                else:
                    desc = str(value) if value else ''
                # Quality: at least 50 characters
                if len(desc) >= 50:
                    achieved_weight += weight
                elif len(desc) >= 20:
                    achieved_weight += weight * 0.5
                    
            elif field == 'resources':
                resources = value or []
                if len(resources) >= 3:
                    achieved_weight += weight
                elif len(resources) >= 1:
                    achieved_weight += weight * 0.7
                    
            elif field == 'tags':
                tags = value or []
                if len(tags) >= 3:
                    achieved_weight += weight
                elif len(tags) >= 1:
                    achieved_weight += weight * 0.5
                    
            elif field == 'groups':
                if value:
                    achieved_weight += weight
                    
            elif field == 'organization':
                org = value or {}
                if org.get('name') or org.get('title'):
                    achieved_weight += weight
                    
            elif value:
                achieved_weight += weight
        
        return achieved_weight / total_weight if total_weight > 0 else 0.0


# ============================================================================
# QUERY PROCESSOR WITH MULTILINGUAL SUPPORT
# ============================================================================

class MultilingualQueryProcessor:
    """
    Processes search queries with support for Swiss national languages.
    
    Features:
    - Language detection
    - Stopword removal (DE, FR, IT, EN)
    - Vague predicate detection ("recent", "complete", etc.)
    - Query expansion
    """
    
    # Stopwords for each language
    STOPWORDS = {
        'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
               'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
               'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
               'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
               'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'},
        'de': {'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer',
               'einem', 'einen', 'und', 'oder', 'aber', 'in', 'auf', 'an', 'zu',
               'für', 'von', 'mit', 'bei', 'nach', 'über', 'unter', 'vor', 'durch',
               'ist', 'sind', 'war', 'waren', 'sein', 'hat', 'haben', 'wird',
               'werden', 'kann', 'können', 'muss', 'müssen', 'soll', 'sollen',
               'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr'},
        'fr': {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'dans',
               'sur', 'à', 'de', 'pour', 'par', 'avec', 'sans', 'sous', 'vers',
               'est', 'sont', 'était', 'être', 'avoir', 'a', 'ont', 'fait',
               'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
               'ce', 'cette', 'ces', 'qui', 'que', 'quoi', 'dont', 'où'},
        'it': {'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'e', 'o',
               'ma', 'in', 'su', 'a', 'di', 'da', 'per', 'con', 'tra', 'fra',
               'è', 'sono', 'era', 'essere', 'avere', 'ha', 'hanno', 'fatto',
               'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro', 'che', 'chi'}
    }
    
    # Vague predicates in multiple languages
    VAGUE_PREDICATES = {
        'recency': {
            'recent', 'new', 'latest', 'fresh', 'current', 'updated',
            'neu', 'aktuell', 'neueste', 'frisch',
            'récent', 'nouveau', 'actuel', 'dernier',
            'recente', 'nuovo', 'attuale', 'ultimo'
        },
        'completeness': {
            'complete', 'comprehensive', 'full', 'detailed', 'documented',
            'well-documented', 'thorough',
            'vollständig', 'komplett', 'vollumfänglich', 'dokumentiert',
            'complet', 'détaillé', 'documenté',
            'completo', 'dettagliato', 'documentato'
        },
        'quality': {
            'quality', 'good', 'reliable', 'accurate', 'verified',
            'qualität', 'gut', 'zuverlässig', 'genau',
            'qualité', 'bon', 'fiable', 'précis',
            'qualità', 'buono', 'affidabile', 'preciso'
        }
    }
    
    def __init__(self):
        """Initialize the query processor."""
        self.all_stopwords = set()
        for sw_set in self.STOPWORDS.values():
            self.all_stopwords.update(sw_set)
    
    def detect_language(self, query: str) -> str:
        """
        Detect the primary language of the query.
        
        Uses stopword frequency analysis.
        """
        words = set(query.lower().split())
        
        scores = {}
        for lang, stopwords in self.STOPWORDS.items():
            overlap = len(words & stopwords)
            scores[lang] = overlap
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'en'  # Default to English
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        # Tokenize
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Remove stopwords
        keywords = [w for w in words if w not in self.all_stopwords and len(w) > 2]
        
        return keywords
    
    def detect_vague_predicates(self, query: str) -> Dict[str, bool]:
        """
        Detect vague predicates in the query.
        
        Returns which types of vague terms are present.
        """
        query_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        detected = {}
        for predicate_type, terms in self.VAGUE_PREDICATES.items():
            detected[predicate_type] = bool(words & terms)
        
        return detected
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Fully process a search query.
        
        Returns:
            Dictionary with language, keywords, vague predicates, etc.
        """
        return {
            'original': query,
            'language': self.detect_language(query),
            'keywords': self.extract_keywords(query),
            'vague_predicates': self.detect_vague_predicates(query),
            'has_vague_terms': any(self.detect_vague_predicates(query).values())
        }


# ============================================================================
# SIMILARITY CALCULATOR
# ============================================================================

class SimilarityCalculator:
    """
    Calculate query-document similarity using TF-IDF inspired approach.
    """
    
    def calculate(
        self,
        query_keywords: List[str],
        dataset: Dict
    ) -> float:
        """
        Calculate similarity between query and dataset.
        
        Args:
            query_keywords: Extracted query keywords
            dataset: Dataset dictionary
            
        Returns:
            Similarity score (0-1)
        """
        if not query_keywords:
            return 0.5  # Neutral if no keywords
        
        # Build document text
        doc_parts = []
        
        # Title
        title = dataset.get('title', {})
        if isinstance(title, dict):
            doc_parts.extend(str(v).lower() for v in title.values() if v)
        elif title:
            doc_parts.append(str(title).lower())
        
        # Description
        desc = dataset.get('description', {})
        if isinstance(desc, dict):
            doc_parts.extend(str(v).lower() for v in desc.values() if v)
        elif desc:
            doc_parts.append(str(desc).lower())
        
        # Tags
        tags = dataset.get('tags', [])
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, dict):
                    doc_parts.append(str(tag.get('name', '')).lower())
                else:
                    doc_parts.append(str(tag).lower())
        
        # Groups/Themes
        groups = dataset.get('groups', [])
        for group in groups:
            if isinstance(group, dict):
                doc_parts.append(str(group.get('name', '')).lower())
        
        doc_text = ' '.join(doc_parts)
        
        # Calculate match score
        matches = 0
        weighted_matches = 0
        
        for keyword in query_keywords:
            kw_lower = keyword.lower()
            
            # Exact match
            if kw_lower in doc_text:
                matches += 1
                
                # Weight by where it appears
                title_text = ' '.join(str(v).lower() for v in 
                    (title.values() if isinstance(title, dict) else [title]) if v)
                
                if kw_lower in title_text:
                    weighted_matches += 2.0  # Title match is worth more
                else:
                    weighted_matches += 1.0
        
        if not query_keywords:
            return 0.5
        
        # Normalize
        base_score = matches / len(query_keywords)
        weighted_score = weighted_matches / (len(query_keywords) * 2)
        
        # Combine
        final_score = 0.6 * weighted_score + 0.4 * base_score
        
        return min(1.0, final_score)


# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================

class ExplanationGenerator:
    """
    Generates human-readable explanations for ranking decisions.
    
    This is key for addressing RQ3: explainability and user trust.
    """
    
    RECENCY_EXPLANATIONS = {
        'very_recent': "📅 Very recent data (updated within the last month)",
        'recent': "📅 Recently updated data (within the past year)",
        'moderate': "📅 Moderately recent data (1-3 years old)",
        'old': "📅 Older dataset (more than 3 years since update)",
        'very_old': "📅 Historical data (not recently updated)"
    }
    
    COMPLETENESS_EXPLANATIONS = {
        'complete': "📋 Excellent metadata documentation",
        'high': "📋 Well-documented with comprehensive metadata",
        'medium': "📋 Adequately documented metadata",
        'partial': "📋 Partial metadata documentation",
        'low': "📋 Minimal metadata available"
    }
    
    RESOURCE_EXPLANATIONS = {
        'comprehensive': "📁 Extensive resource collection with multiple formats",
        'rich': "📁 Multiple resources and formats available",
        'moderate': "📁 Several resources provided",
        'limited': "📁 Limited resources available",
        'minimal': "📁 Single resource only"
    }
    
    SIMILARITY_EXPLANATIONS = {
        'exact_match': "🎯 Excellent match to your search terms",
        'highly_relevant': "🎯 Highly relevant to your query",
        'relevant': "🎯 Relevant to your search",
        'somewhat_relevant': "🎯 Partially matches your query",
        'not_relevant': "🎯 Limited relevance to search terms"
    }
    
    def generate(
        self,
        factors: RankingFactors,
        vague_predicates: Dict[str, bool]
    ) -> str:
        """
        Generate natural language explanation.
        
        Args:
            factors: RankingFactors with scores and terms
            vague_predicates: Which vague terms were in the query
            
        Returns:
            Human-readable explanation string
        """
        lines = []
        
        # Main relevance statement
        score_pct = int(factors.fuzzy_relevance * 100)
        if score_pct >= 80:
            lines.append(f"**Excellent Match** (Relevance: {score_pct}%)")
        elif score_pct >= 60:
            lines.append(f"**Good Match** (Relevance: {score_pct}%)")
        elif score_pct >= 40:
            lines.append(f"**Moderate Match** (Relevance: {score_pct}%)")
        else:
            lines.append(f"**Lower Relevance** (Score: {score_pct}%)")
        
        lines.append("")
        
        # Factor breakdowns
        factor_lines = []
        
        # Similarity
        sim_exp = self.SIMILARITY_EXPLANATIONS.get(
            factors.similarity_term, 
            f"🎯 Similarity: {int(factors.similarity_score * 100)}%"
        )
        factor_lines.append(sim_exp)
        
        # Recency - especially if user asked for "recent"
        rec_exp = self.RECENCY_EXPLANATIONS.get(
            factors.recency_term,
            f"📅 Recency score: {int(factors.recency_score * 100)}%"
        )
        if vague_predicates.get('recency'):
            factor_lines.insert(0, f"**{rec_exp}** ← You asked for recent data")
        else:
            factor_lines.append(rec_exp)
        
        # Completeness - especially if user asked for "complete"
        comp_exp = self.COMPLETENESS_EXPLANATIONS.get(
            factors.completeness_term,
            f"📋 Completeness: {int(factors.completeness_score * 100)}%"
        )
        if vague_predicates.get('completeness'):
            factor_lines.insert(0, f"**{comp_exp}** ← You asked for complete data")
        else:
            factor_lines.append(comp_exp)
        
        # Resources
        res_exp = self.RESOURCE_EXPLANATIONS.get(
            factors.resource_term,
            f"📁 Resources: {int(factors.resource_score * 100)}%"
        )
        factor_lines.append(res_exp)
        
        lines.extend(factor_lines)
        
        return '\n'.join(lines)


# ============================================================================
# BASELINE RANKING SYSTEMS (FOR COMPARISON)
# ============================================================================

def dataset_identity_key(dataset: Dict) -> str:
    """Build a stable identity key used to remove duplicate datasets."""
    name = str(dataset.get('name') or '').strip().lower()
    if name:
        return f"name:{name}"

    dataset_id = str(dataset.get('id') or '').strip().lower()
    if dataset_id:
        return f"id:{dataset_id}"

    title = dataset.get('title', {})
    if isinstance(title, dict):
        title_text = str(
            title.get('en')
            or title.get('de')
            or title.get('fr')
            or title.get('it')
            or next(iter(title.values()), '')
        ).strip().lower()
    else:
        title_text = str(title or '').strip().lower()

    org = dataset.get('organization', {}) or {}
    if isinstance(org, dict):
        org_name = str(org.get('name') or org.get('title') or '').strip().lower()
    else:
        org_name = str(org).strip().lower()

    return f"fallback:{title_text}|{org_name}"


def deduplicate_datasets(datasets: List[Dict]) -> List[Dict]:
    """Remove duplicate datasets while preserving first-seen order."""
    unique: List[Dict] = []
    seen_keys: set[str] = set()

    for dataset in datasets:
        key = dataset_identity_key(dataset)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(dataset)

    return unique


def dataset_display_key(dataset: Dict) -> str:
    """Build a display-level key to collapse repeated title rows in UI."""
    title = dataset.get('title', {})
    if isinstance(title, dict):
        title_text = str(
            title.get('en')
            or title.get('de')
            or title.get('fr')
            or title.get('it')
            or next(iter(title.values()), '')
        ).strip().lower()
    else:
        title_text = str(title or '').strip().lower()

    org = dataset.get('organization', {}) or {}
    if isinstance(org, dict):
        org_text = str(org.get('title') or org.get('name') or '').strip().lower()
    else:
        org_text = str(org).strip().lower()

    return f"{title_text}|{org_text}"


def deduplicate_display_datasets(datasets: List[Dict]) -> List[Dict]:
    """Deduplicate raw datasets by visible title and organization for cleaner output."""
    unique: List[Dict] = []
    seen_keys: set[str] = set()

    for dataset in datasets:
        key = dataset_display_key(dataset)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(dataset)

    return unique


def ranked_result_display_key(result: DatasetResult) -> str:
    """Build a display-level key for ranked results."""
    title_text = str(
        result.title.get('en')
        or result.title.get('de')
        or result.title.get('fr')
        or result.title.get('it')
        or next(iter(result.title.values()), '')
    ).strip().lower()
    org_text = str(result.organization or '').strip().lower()
    return f"{title_text}|{org_text}"


def deduplicate_ranked_results(results: List[DatasetResult]) -> List[DatasetResult]:
    """Deduplicate ranked outputs by visible title and organization."""
    unique: List[DatasetResult] = []
    seen_keys: set[str] = set()

    for result in results:
        key = ranked_result_display_key(result)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(result)

    for index, result in enumerate(unique, start=1):
        result.rank = index

    return unique

class PortalDefaultRanker:
    """
    Simulates the portal's default ranking (as returned by API).
    Uses the order returned by opendata.swiss CKAN search.
    """
    
    def rank(self, datasets: List[Dict], query: str) -> List[Dict]:
        """Return datasets in their original API order."""
        datasets = deduplicate_datasets(datasets)
        for i, ds in enumerate(datasets):
            ds['_relevance_score'] = 1.0 - (i * 0.02)  # Decreasing by position
            ds['_ranking_method'] = 'portal_default'
        return datasets


class BM25Ranker:
    """
    BM25 probabilistic ranking.
    
    Reference: Robertson et al. (1994) "Okapi at TREC-3"
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _get_doc_text(self, dataset: Dict) -> str:
        """Extract searchable text from dataset."""
        parts = []
        
        title = dataset.get('title', {})
        if isinstance(title, dict):
            parts.extend(str(v) for v in title.values() if v)
        elif title:
            parts.append(str(title))
        
        desc = dataset.get('description', {})
        if isinstance(desc, dict):
            parts.extend(str(v) for v in desc.values() if v)
        elif desc:
            parts.append(str(desc))
        
        tags = dataset.get('tags', [])
        for tag in tags:
            if isinstance(tag, dict):
                parts.append(str(tag.get('name', '')))
            else:
                parts.append(str(tag))
        
        return ' '.join(parts)
    
    def rank(self, datasets: List[Dict], query: str) -> List[Dict]:
        """Rank datasets using BM25."""
        datasets = deduplicate_datasets(datasets)
        query_terms = self._tokenize(query)
        
        if not query_terms or not datasets:
            return datasets
        
        # Build corpus
        docs = [self._tokenize(self._get_doc_text(ds)) for ds in datasets]
        
        # Compute average document length
        avg_dl = sum(len(d) for d in docs) / len(docs) if docs else 1
        
        # Compute IDF for query terms
        N = len(docs)
        idf = {}
        for term in query_terms:
            df = sum(1 for doc in docs if term in doc)
            idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1) if df > 0 else 0
        
        # Score each document
        scores = []
        for doc in docs:
            score = 0.0
            dl = len(doc)
            term_freq = defaultdict(int)
            for term in doc:
                term_freq[term] += 1
            
            for term in query_terms:
                if term in term_freq:
                    tf = term_freq[term]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                    score += idf.get(term, 0) * numerator / denominator
            
            scores.append(score)
        
        # Normalize scores
        max_score = max(scores) if scores and max(scores) > 0 else 1
        
        # Assign scores
        for i, ds in enumerate(datasets):
            ds['_relevance_score'] = scores[i] / max_score if max_score > 0 else 0
            ds['_ranking_method'] = 'bm25'
        
        # Sort by score
        return sorted(datasets, key=lambda x: x.get('_relevance_score', 0), reverse=True)


# ============================================================================
# MAIN FUZZY RANKING SYSTEM
# ============================================================================

class FuzzyHCIRRanker:
    """
    Main fuzzy human-centered ranking system.
    
    Combines:
    - Query processing with vague predicate detection
    - Metadata quality analysis
    - Fuzzy inference for relevance scoring
    - Explainable results
    """
    
    def __init__(self):
        self.fuzzy_engine = CalibratedFuzzyEngine()
        self.metadata_analyzer = MetadataAnalyzer()
        self.query_processor = MultilingualQueryProcessor()
        self.similarity_calculator = SimilarityCalculator()
        self.explanation_generator = ExplanationGenerator()
    
    def rank(
        self,
        datasets: List[Dict],
        query: str
    ) -> List[DatasetResult]:
        """
        Rank datasets using fuzzy inference.
        
        Args:
            datasets: Raw datasets from CKAN API
            query: User search query
            
        Returns:
            List of DatasetResult with rankings and explanations
        """
        datasets = deduplicate_datasets(datasets)

        # Process query
        query_info = self.query_processor.process(query)
        keywords = query_info['keywords']
        vague_predicates = query_info['vague_predicates']
        
        results = []
        
        for ds in datasets:
            # Compute factors
            
            # 1. Recency
            try:
                mod_str = ds.get('metadata_modified', '')
                if mod_str:
                    mod_date = datetime.fromisoformat(mod_str.replace('Z', '+00:00'))
                    days_since = (datetime.now(mod_date.tzinfo) - mod_date).days
                else:
                    days_since = 365
            except:
                days_since = 365
            
            # 2. Completeness
            completeness = self.metadata_analyzer.compute_completeness(ds)
            
            # 3. Resource count
            resources = ds.get('resources', [])
            resource_count = len(resources) if resources else 0
            
            # 4. Similarity
            similarity = self.similarity_calculator.calculate(keywords, ds)
            
            # Fuzzy inference
            relevance, fuzzified = self.fuzzy_engine.infer(
                recency_days=days_since,
                completeness=completeness,
                resource_count=resource_count,
                similarity=similarity
            )
            
            # Build ranking factors
            factors = RankingFactors(
                recency_score=1.0 - min(days_since / 3650, 1.0),  # Normalize to 0-1
                completeness_score=completeness,
                resource_score=min(resource_count / 10, 1.0),
                similarity_score=similarity,
                fuzzy_relevance=relevance,
                recency_term=fuzzified['recency'].dominant_term[0],
                completeness_term=fuzzified['completeness'].dominant_term[0],
                resource_term=fuzzified['resources'].dominant_term[0],
                similarity_term=fuzzified['similarity'].dominant_term[0]
            )
            
            # Generate explanation
            explanation = self.explanation_generator.generate(factors, vague_predicates)
            
            # Build result
            title = ds.get('title', {})
            if isinstance(title, str):
                title = {'en': title}
            
            description = ds.get('description', {})
            if isinstance(description, dict):
                desc_text = description.get('en') or description.get('de') or \
                           description.get('fr') or next(iter(description.values()), '')
            else:
                desc_text = str(description) if description else ''
            
            org = ds.get('organization', {})
            org_name = org.get('title') or org.get('name') or 'Unknown' if org else 'Unknown'
            
            groups = ds.get('groups', [])
            themes = [g.get('name', '') for g in groups if isinstance(g, dict)]
            
            tags = ds.get('tags', [])
            tag_names = []
            for tag in tags:
                if isinstance(tag, dict):
                    tag_names.append(tag.get('name', ''))
                else:
                    tag_names.append(str(tag))
            
            result = DatasetResult(
                id=ds.get('name', ds.get('id', '')),
                title=title,
                description=desc_text[:500],
                organization=org_name,
                resources=resources,
                themes=themes,
                tags=tag_names[:10],
                modified=ds.get('metadata_modified', ''),
                created=ds.get('metadata_created', ''),
                license=ds.get('license_id', 'Unknown'),
                url=f"https://opendata.swiss/en/dataset/{ds.get('name', '')}",
                relevance_score=relevance,
                factors=factors,
                explanation=explanation
            )
            
            results.append(result)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Assign ranks
        for i, r in enumerate(results):
            r.rank = i + 1
        
        return results


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def render_header():
    """Render Swiss government style header."""
    st.markdown(SWISS_GOV_CSS, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="swiss-header">
        <div style="display: flex; align-items: center;">
            <div class="swiss-cross"></div>
            <div>
                <h1>🔍 opendata.swiss | Fuzzy Search</h1>
                <p>Research Prototype - Human-Centered Information Retrieval</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with options."""
    with st.sidebar:
        st.markdown("### 🔧 Search Settings")
        
        # Data source
        data_source = st.radio(
            "Data Source",
            ["🌐 Live API (opendata.swiss)", "📦 Demo Data"],
            index=0
        )
        
        st.markdown("---")
        
        # Ranking method
        st.markdown("### 📊 Ranking Method")
        ranking_method = st.selectbox(
            "Select ranking algorithm",
            [
                "Fuzzy HCIR (Research)",
                "Portal Default",
                "BM25 Keyword",
                "Compare All"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 🏷️ Filters")
        
        # Theme filter
        themes = [
            "All Themes",
            "environment", "mobility", "economy", "population",
            "health", "energy", "education", "agriculture",
            "geography", "culture", "politics", "crime",
            "construction", "finances"
        ]
        selected_theme = st.selectbox("Theme", themes)
        
        # Language filter
        languages = ["All", "English", "German", "French", "Italian"]
        selected_lang = st.selectbox("Interface Language", languages)
        
        st.markdown("---")
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            num_results = st.slider("Results per page", 10, 50, 20)
            show_explanations = st.checkbox("Show ranking explanations", value=True)
            show_factors = st.checkbox("Show factor breakdown", value=True)
        
        st.markdown("---")
        
        # About
        st.markdown("""
        ### 📖 About This Prototype
        
        This is a research prototype for the Master Thesis:
        
        *"Improving Access to Swiss Open Government Data through 
        Fuzzy Human-Centered Information Retrieval"*
        
        **University of Fribourg**  
        Human-IST Institute
        
        **Author:** Deep Shukla  
        **Supervisor:** Janick Spycher  
        **Examiner:** Prof. Dr. Edy Portmann
        """)
        
        return {
            'data_source': data_source,
            'ranking_method': ranking_method,
            'theme': selected_theme if selected_theme != "All Themes" else None,
            'language': selected_lang,
            'num_results': num_results if 'num_results' in dir() else 20,
            'show_explanations': show_explanations if 'show_explanations' in dir() else True,
            'show_factors': show_factors if 'show_factors' in dir() else True
        }


def get_format_badge_class(fmt: str) -> str:
    """Get CSS class for format badge."""
    fmt_upper = fmt.upper()
    format_classes = {
        'CSV': 'format-csv',
        'JSON': 'format-json',
        'XML': 'format-xml',
        'PDF': 'format-pdf',
        'API': 'format-api',
        'GEOJSON': 'format-geojson',
        'XLSX': 'format-xlsx',
        'XLS': 'format-xlsx'
    }
    return format_classes.get(fmt_upper, 'format-default')


def render_result_card(result: DatasetResult, settings: Dict):
    """Render a single result card."""
    # Determine relevance class
    score_pct = int(result.relevance_score * 100)
    if score_pct >= 70:
        relevance_class = "high-relevance"
        badge_class = "score-excellent"
    elif score_pct >= 50:
        relevance_class = "medium-relevance"
        badge_class = "score-good"
    elif score_pct >= 30:
        relevance_class = "medium-relevance"
        badge_class = "score-moderate"
    else:
        relevance_class = "low-relevance"
        badge_class = "score-low"
    
    # Get title
    title = result.title.get('en') or result.title.get('de') or \
            result.title.get('fr') or next(iter(result.title.values()), 'Untitled')
    
    # Format badges HTML
    format_badges = ' '.join([
        f'<span class="format-badge {get_format_badge_class(fmt)}">{fmt}</span>'
        for fmt in result.format_list
    ])
    
    # Tags HTML
    tags_html = ' '.join([
        f'<span class="tag-pill">{tag}</span>'
        for tag in result.tags[:5]
    ])
    
    # Card HTML
    card_html = f"""
    <div class="result-card {relevance_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem; font-weight: 600; color: #333;">#{result.rank}</span>
                    <span class="score-badge {badge_class}">{score_pct}% Relevant</span>
                    {format_badges}
                </div>
                <h3 style="margin: 0 0 8px 0; font-size: 1.15rem;">
                    <a href="{result.url}" target="_blank" style="color: #006699; text-decoration: none;">
                        {title}
                    </a>
                </h3>
                <p style="color: #555; font-size: 0.9rem; margin: 0 0 10px 0; line-height: 1.5;">
                    {result.description[:250]}{'...' if len(result.description) > 250 else ''}
                </p>
                <div style="display: flex; flex-wrap: wrap; gap: 15px; font-size: 0.85rem; color: #666;">
                    <span class="org-badge">
                        <svg width="14" height="14" fill="currentColor" viewBox="0 0 16 16">
                            <path d="M8.707 1.5a1 1 0 0 0-1.414 0L.646 8.146a.5.5 0 0 0 .708.708L8 2.207l6.646 6.647a.5.5 0 0 0 .708-.708L13 5.793V2.5a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5v1.293L8.707 1.5Z"/>
                            <path d="m8 3.293 6 6V13.5a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 13.5V9.293l6-6Z"/>
                        </svg>
                        {result.organization}
                    </span>
                    <span>📅 Modified: {result.days_since_modified} days ago</span>
                    <span>📁 {len(result.resources)} resources</span>
                </div>
                <div style="margin-top: 8px;">
                    {tags_html}
                </div>
            </div>
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Show explanation if enabled
    if settings.get('show_explanations') and result.factors:
        with st.expander("📊 Why this ranking?", expanded=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Factor Scores:**")
                
                # Similarity
                st.markdown(f"🎯 **Similarity:** {result.factors.similarity_term}")
                st.progress(result.factors.similarity_score)
                
                # Recency
                st.markdown(f"📅 **Recency:** {result.factors.recency_term}")
                st.progress(result.factors.recency_score)
            
            with col2:
                # Completeness
                st.markdown(f"📋 **Completeness:** {result.factors.completeness_term}")
                st.progress(result.factors.completeness_score)
                
                # Resources
                st.markdown(f"📁 **Resources:** {result.factors.resource_term}")
                st.progress(result.factors.resource_score)
            
            st.markdown("---")
            st.markdown(result.explanation)


def render_comparison_view(
    results_fuzzy: List[DatasetResult],
    results_portal: List[Dict],
    results_bm25: List[Dict],
    query: str
):
    """Render comparison view of all ranking methods."""
    st.markdown("### 📊 Ranking Comparison")
    st.markdown("Compare how different algorithms rank the same datasets.")
    
    # Build comparison table
    comparison_data = []
    
    # Create ID to result mapping
    fuzzy_ranks = {r.id: r.rank for r in results_fuzzy}
    portal_ranks = {r.get('name', ''): i+1 for i, r in enumerate(results_portal)}
    bm25_ranks = {r.get('name', ''): i+1 for i, r in enumerate(results_bm25)}
    
    for result in results_fuzzy[:10]:
        title = result.title.get('en') or result.title.get('de') or 'Untitled'
        comparison_data.append({
            'Dataset': title[:50] + ('...' if len(title) > 50 else ''),
            'Fuzzy HCIR': fuzzy_ranks.get(result.id, '-'),
            'Portal Default': portal_ranks.get(result.id, '-'),
            'BM25': bm25_ranks.get(result.id, '-'),
            'Fuzzy Score': f"{int(result.relevance_score * 100)}%"
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    # Metrics comparison
    st.markdown("### 📈 Method Characteristics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #4A9F35;">Fuzzy HCIR</div>
            <div class="metric-label">Handles vague queries<br/>Explainable rankings<br/>Quality-aware</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #006699;">Portal Default</div>
            <div class="metric-label">Solr-based scoring<br/>Optimized for keywords<br/>Fast retrieval</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #F29400;">BM25</div>
            <div class="metric-label">Probabilistic model<br/>Term frequency based<br/>Document length normalized</div>
        </div>
        """, unsafe_allow_html=True)


def get_demo_data() -> List[Dict]:
    """Get demo datasets for offline testing."""
    return [
        {
            "name": "air-quality-zurich-2024",
            "title": {"en": "Air Quality Measurements Zurich 2024", "de": "Luftqualitätsmessungen Zürich 2024"},
            "description": {"en": "Daily air quality measurements including PM2.5, PM10, NO2, and O3 levels across monitoring stations in the Zurich metropolitan area. Updated regularly with quality-controlled data."},
            "tags": [{"name": "air quality"}, {"name": "pollution"}, {"name": "environment"}, {"name": "PM2.5"}, {"name": "zurich"}],
            "groups": [{"name": "environment"}],
            "resources": [{"format": "CSV"}, {"format": "JSON"}, {"format": "API"}],
            "organization": {"name": "Stadt Zürich", "title": "City of Zurich"},
            "metadata_modified": (datetime.now() - timedelta(days=3)).isoformat(),
            "metadata_created": (datetime.now() - timedelta(days=400)).isoformat(),
            "license_id": "cc-by-4.0"
        },
        {
            "name": "traffic-volume-swiss-highways-2023",
            "title": {"en": "Traffic Volume Swiss Highways 2023", "de": "Verkehrsaufkommen Schweizer Autobahnen 2023"},
            "description": {"en": "Annual traffic volume statistics for Swiss national highways and main roads. Includes vehicle counts, peak hours, and seasonal variations."},
            "tags": [{"name": "traffic"}, {"name": "transport"}, {"name": "mobility"}, {"name": "highways"}],
            "groups": [{"name": "mobility"}],
            "resources": [{"format": "CSV"}, {"format": "PDF"}],
            "organization": {"name": "ASTRA", "title": "Federal Roads Office"},
            "metadata_modified": (datetime.now() - timedelta(days=45)).isoformat(),
            "metadata_created": (datetime.now() - timedelta(days=800)).isoformat(),
            "license_id": "cc-by-4.0"
        },
        {
            "name": "water-quality-swiss-rivers-2024",
            "title": {"en": "Water Quality Swiss Rivers 2024", "de": "Wasserqualität Schweizer Flüsse 2024"},
            "description": {"en": "Comprehensive water quality measurements for major Swiss rivers including pollution indicators, chemical parameters, and biological quality indices."},
            "tags": [{"name": "water quality"}, {"name": "rivers"}, {"name": "environment"}, {"name": "pollution"}],
            "groups": [{"name": "environment"}],
            "resources": [{"format": "CSV"}, {"format": "JSON"}, {"format": "GeoJSON"}],
            "organization": {"name": "BAFU", "title": "Federal Office for the Environment"},
            "metadata_modified": (datetime.now() - timedelta(days=10)).isoformat(),
            "metadata_created": (datetime.now() - timedelta(days=200)).isoformat(),
            "license_id": "cc-by-4.0"
        },
        {
            "name": "public-transport-punctuality-sbb",
            "title": {"en": "Public Transport Punctuality Statistics", "de": "Pünktlichkeitsstatistik öffentlicher Verkehr"},
            "description": {"en": "Statistics on train and bus punctuality across the Swiss public transport network. Monthly aggregated data."},
            "tags": [{"name": "public transport"}, {"name": "trains"}, {"name": "mobility"}, {"name": "punctuality"}],
            "groups": [{"name": "mobility"}],
            "resources": [{"format": "CSV"}],
            "organization": {"name": "SBB", "title": "Swiss Federal Railways"},
            "metadata_modified": (datetime.now() - timedelta(days=180)).isoformat(),
            "metadata_created": (datetime.now() - timedelta(days=1500)).isoformat(),
            "license_id": "cc-by-4.0"
        },
        {
            "name": "noise-pollution-cities-switzerland",
            "title": {"en": "Noise Pollution Urban Areas Switzerland", "de": "Lärmbelastung Städte Schweiz"},
            "description": {"en": "Noise level measurements and mapping data for Swiss urban areas. Includes road traffic noise, railway noise, and aircraft noise exposure data."},
            "tags": [{"name": "noise"}, {"name": "pollution"}, {"name": "environment"}, {"name": "cities"}, {"name": "health"}],
            "groups": [{"name": "environment"}],
            "resources": [{"format": "CSV"}, {"format": "GeoJSON"}, {"format": "PDF"}],
            "organization": {"name": "BAFU", "title": "Federal Office for the Environment"},
            "metadata_modified": (datetime.now() - timedelta(days=90)).isoformat(),
            "metadata_created": (datetime.now() - timedelta(days=600)).isoformat(),
            "license_id": "cc-by-4.0"
        },
        {
            "name": "forest-inventory-switzerland",
            "title": {"en": "Swiss National Forest Inventory", "de": "Schweizerisches Landesforstinventar"},
            "description": {"en": "Complete forest inventory data including forest coverage, tree species distribution, forest health indicators, and biomass estimates."},
            "tags": [{"name": "forest"}, {"name": "environment"}, {"name": "biodiversity"}, {"name": "vegetation"}],
            "groups": [{"name": "environment"}],
            "resources": [{"format": "CSV"}, {"format": "JSON"}, {"format": "GeoJSON"}, {"format": "PDF"}, {"format": "XLSX"}],
            "organization": {"name": "WSL", "title": "Swiss Federal Research Institute"},
            "metadata_modified": (datetime.now() - timedelta(days=30)).isoformat(),
            "metadata_created": (datetime.now() - timedelta(days=2000)).isoformat(),
            "license_id": "cc-by-4.0"
        },
        {
            "name": "bicycle-infrastructure-network",
            "title": {"en": "Swiss Bicycle Infrastructure Network", "de": "Veloinfrastruktur Schweiz"},
            "description": {"en": "Cycling routes, bike lanes, and bicycle parking facilities across Switzerland. Complete network data with quality attributes."},
            "tags": [{"name": "bicycle"}, {"name": "cycling"}, {"name": "mobility"}, {"name": "infrastructure"}],
            "groups": [{"name": "mobility"}],
            "resources": [{"format": "GeoJSON"}, {"format": "CSV"}],
            "organization": {"name": "ARE", "title": "Federal Office for Spatial Development"},
            "metadata_modified": (datetime.now() - timedelta(days=60)).isoformat(),
            "metadata_created": (datetime.now() - timedelta(days=500)).isoformat(),
            "license_id": "cc-by-4.0"
        },
        {
            "name": "climate-data-meteoswiss",
            "title": {"en": "Climate Data Switzerland", "de": "Klimadaten Schweiz"},
            "description": {"en": "Historical and current climate data from MeteoSwiss including temperature, precipitation, and other meteorological parameters."},
            "tags": [{"name": "climate"}, {"name": "weather"}, {"name": "temperature"}, {"name": "environment"}],
            "groups": [{"name": "environment"}],
            "resources": [{"format": "CSV"}, {"format": "JSON"}, {"format": "API"}, {"format": "NetCDF"}],
            "organization": {"name": "MeteoSwiss", "title": "Federal Office of Meteorology"},
            "metadata_modified": (datetime.now() - timedelta(days=1)).isoformat(),
            "metadata_created": (datetime.now() - timedelta(days=3000)).isoformat(),
            "license_id": "cc-by-4.0"
        }
    ]


def main():
    """Main application entry point."""
    render_header()
    settings = render_sidebar()
    
    # Initialize clients and rankers
    api_client = OpenDataSwissClient()
    fuzzy_ranker = FuzzyHCIRRanker()
    portal_ranker = PortalDefaultRanker()
    bm25_ranker = BM25Ranker()
    
    # Search interface
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Search Swiss Open Government Data",
            placeholder="e.g., 'recent air quality data', 'complete transport statistics'",
            key="search_query"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("Search", type="primary", use_container_width=True)
    
    # Example queries
    st.markdown("**Try these:**")
    example_cols = st.columns(4)
    example_queries = [
        "recent air quality data",
        "complete transport statistics",
        "environment pollution monitoring",
        "well-documented climate data"
    ]
    
    for col, example in zip(example_cols, example_queries):
        with col:
            if st.button(f"📝 {example}", key=f"example_{example}"):
                query = example
                search_clicked = True
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Perform search
    if query and search_clicked:
        with st.spinner("Searching opendata.swiss..."):
            # Get data
            if "Demo" in settings['data_source']:
                raw_results = get_demo_data()
                total_count = len(raw_results)
            else:
                # Build filter query
                fq = f"groups:{settings['theme']}" if settings['theme'] else None
                raw_results, total_count = api_client.search(
                    query, 
                    rows=settings.get('num_results', 20),
                    fq=fq
                )

            raw_results = deduplicate_datasets(raw_results)
            raw_results = deduplicate_display_datasets(raw_results)
            
            if raw_results:
                # Process query for info display
                query_processor = MultilingualQueryProcessor()
                query_info = query_processor.process(query)
                
                # Display query analysis
                if query_info['has_vague_terms']:
                    vague_found = [k for k, v in query_info['vague_predicates'].items() if v]
                    st.info(f"🧠 **Fuzzy query detected!** Found vague terms: {', '.join(vague_found)}. "
                           f"The fuzzy system will interpret these contextually.")
                
                # Apply ranking
                if settings['ranking_method'] == "Compare All":
                    results_fuzzy = deduplicate_ranked_results(fuzzy_ranker.rank(raw_results.copy(), query))
                    results_portal = deduplicate_display_datasets(portal_ranker.rank(raw_results.copy(), query))
                    results_bm25 = deduplicate_display_datasets(bm25_ranker.rank(raw_results.copy(), query))
                    
                    render_comparison_view(results_fuzzy, results_portal, results_bm25, query)
                    
                    st.markdown("---")
                    st.markdown("### 🏆 Fuzzy HCIR Results")
                    for result in results_fuzzy[:10]:
                        render_result_card(result, settings)
                
                elif settings['ranking_method'] == "Fuzzy HCIR (Research)":
                    results = deduplicate_ranked_results(fuzzy_ranker.rank(raw_results, query))
                    
                    st.markdown(f"### 📊 Found {total_count} datasets • Showing top {len(results)}")
                    
                    for result in results:
                        render_result_card(result, settings)
                
                elif settings['ranking_method'] == "Portal Default":
                    raw_results = deduplicate_display_datasets(portal_ranker.rank(raw_results, query))
                    
                    st.markdown(f"### 📊 Found {total_count} datasets (Portal Default Ranking)")
                    
                    for i, ds in enumerate(raw_results):
                        # Quick card for portal results
                        title = ds.get('title', {})
                        if isinstance(title, dict):
                            title_str = title.get('en') or title.get('de') or 'Untitled'
                        else:
                            title_str = str(title)
                        
                        st.markdown(f"**#{i+1}** [{title_str}](https://opendata.swiss/en/dataset/{ds.get('name', '')})")
                        
                        desc = ds.get('description', {})
                        if isinstance(desc, dict):
                            desc_str = desc.get('en') or desc.get('de') or ''
                        else:
                            desc_str = str(desc) if desc else ''
                        st.markdown(f"<small>{desc_str[:200]}...</small>", unsafe_allow_html=True)
                        st.markdown("---")
                
                else:  # BM25
                    raw_results = deduplicate_display_datasets(bm25_ranker.rank(raw_results, query))
                    st.markdown(f"### 📊 Found {total_count} datasets (BM25 Ranking)")
                    
                    for i, ds in enumerate(raw_results):
                        title = ds.get('title', {})
                        if isinstance(title, dict):
                            title_str = title.get('en') or title.get('de') or 'Untitled'
                        else:
                            title_str = str(title)
                        
                        score = ds.get('_relevance_score', 0)
                        st.markdown(f"**#{i+1}** [{title_str}](https://opendata.swiss/en/dataset/{ds.get('name', '')}) - Score: {score:.2f}")
                        st.markdown("---")
            else:
                st.warning("No results found. Try a different search query.")
    
    # Footer
    st.markdown("""
    <div class="swiss-footer">
        <p>🇨🇭 Research Prototype | University of Fribourg - Human-IST Institute</p>
        <p>Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR" | Deep Shukla | 2026</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
