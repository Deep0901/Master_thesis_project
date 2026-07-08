#!/usr/bin/env python3
"""
Phase 2: Metadata Collection and Empirical Analysis

This module provides comprehensive tools for collecting and analyzing
metadata from the Swiss OGD portal (opendata.swiss).

Implements:
- Systematic metadata collection via CKAN API
- Statistical analysis of metadata quality metrics
- Visualization of metadata distributions
- Export of processed analysis results

Author: Deep Shukla
Thesis: Improving Access to Swiss OGD through Fuzzy HCIR
University of Fribourg, Human-IST Institute
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DatasetMetrics:
    """Computed metrics for a single dataset."""
    dataset_id: str
    dataset_name: str
    
    # Basic metadata
    title_length: int
    description_length: int
    tag_count: int
    resource_count: int
    
    # Quality metrics
    completeness_score: float
    documentation_score: float
    
    # Temporal metrics
    days_since_modified: int
    days_since_created: Optional[int]
    
    # Organizational
    organization: str
    organization_type: str
    
    # Multilingual
    languages_available: List[str]
    primary_language: str
    
    # Resource diversity
    format_count: int
    formats_available: List[str]
    
    # Categorization
    themes: List[str]
    spatial_coverage: Optional[str]


@dataclass
class PortalStatistics:
    """Aggregated statistics for the entire portal."""
    total_datasets: int
    total_organizations: int
    total_themes: int
    
    # Distribution statistics
    description_length_stats: Dict[str, float]
    tag_count_stats: Dict[str, float]
    resource_count_stats: Dict[str, float]
    completeness_stats: Dict[str, float]
    recency_stats: Dict[str, float]
    
    # Percentiles for calibration
    recency_percentiles: Dict[str, float]
    completeness_percentiles: Dict[str, float]
    resources_percentiles: Dict[str, float]
    
    # Organization distribution
    top_organizations: List[Tuple[str, int]]
    organization_type_distribution: Dict[str, int]
    
    # Theme distribution
    theme_distribution: Dict[str, int]
    
    # Format distribution
    format_distribution: Dict[str, int]
    
    # Temporal analysis
    update_frequency_distribution: Dict[str, int]
    
    collection_timestamp: str


# ============================================================================
# CKAN API CLIENT
# ============================================================================

class SwissOGDCollector:
    """
    Collector for Swiss Open Government Data metadata.
    
    Interfaces with the CKAN API at opendata.swiss to retrieve
    comprehensive dataset metadata for analysis.
    """
    
    BASE_URL = "https://opendata.swiss/api/3/action/"
    
    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize the collector.
        
        Args:
            rate_limit_delay: Seconds to wait between API requests
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UniFR-Thesis-Research/1.0'
        })
        self.rate_limit_delay = rate_limit_delay
        self.cache = {}
    
    def _request(self, action: str, params: Dict = None) -> Dict:
        """Make a rate-limited API request."""
        url = f"{self.BASE_URL}{action}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get('success'):
                return data.get('result', {})
            else:
                logger.error(f"API error: {data.get('error', 'Unknown')}")
                return {}
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {}
        finally:
            time.sleep(self.rate_limit_delay)
    
    def get_portal_info(self) -> Dict:
        """Get basic portal statistics."""
        info = {}
        
        # Total dataset count
        result = self._request('package_search', {'rows': 0})
        info['total_datasets'] = result.get('count', 0)
        
        # Organization count
        orgs = self._request('organization_list', {'all_fields': True})
        info['organizations'] = orgs if isinstance(orgs, list) else []
        info['total_organizations'] = len(info['organizations'])
        
        # Theme/group count
        groups = self._request('group_list', {'all_fields': True})
        info['themes'] = groups if isinstance(groups, list) else []
        info['total_themes'] = len(info['themes'])
        
        return info
    
    def collect_datasets(self, 
                        n_datasets: int = 500,
                        strategy: str = 'representative') -> List[Dict]:
        """
        Collect dataset metadata.
        
        Args:
            n_datasets: Number of datasets to collect
            strategy: 'all', 'recent', 'representative', or 'random'
        
        Returns:
            List of dataset metadata dictionaries
        """
        logger.info(f"Collecting {n_datasets} datasets using '{strategy}' strategy")
        
        if strategy == 'recent':
            return self._collect_recent(n_datasets)
        elif strategy == 'representative':
            return self._collect_representative(n_datasets)
        elif strategy == 'random':
            return self._collect_random(n_datasets)
        else:
            return self._collect_all(n_datasets)
    
    def _collect_recent(self, n: int) -> List[Dict]:
        """Collect most recently modified datasets."""
        result = self._request('package_search', {
            'rows': n,
            'sort': 'metadata_modified desc'
        })
        return result.get('results', [])
    
    def _collect_representative(self, n: int) -> List[Dict]:
        """
        Collect representative sample across organizations and themes.
        
        Ensures coverage across:
        - Different organization types (federal, cantonal, municipal)
        - Different themes
        - Different time periods
        """
        datasets = []
        
        # Get all themes
        themes = self._request('group_list')
        if not themes:
            themes = []
        
        # Allocate datasets per theme
        per_theme = max(n // len(themes), 10) if themes else n
        
        for theme in themes[:14]:  # Limit to 14 main themes
            result = self._request('package_search', {
                'fq': f'groups:{theme}',
                'rows': per_theme,
                'sort': 'metadata_modified desc'
            })
            theme_datasets = result.get('results', [])
            datasets.extend(theme_datasets)
            logger.info(f"  Collected {len(theme_datasets)} from theme: {theme}")
        
        # Remove duplicates
        seen_ids = set()
        unique = []
        for ds in datasets:
            if ds.get('id') not in seen_ids:
                seen_ids.add(ds.get('id'))
                unique.append(ds)
        
        logger.info(f"Total unique datasets collected: {len(unique)}")
        return unique[:n]
    
    def _collect_random(self, n: int) -> List[Dict]:
        """Collect random sample of datasets."""
        # Get total count
        result = self._request('package_search', {'rows': 0})
        total = result.get('count', 0)
        
        if total == 0:
            return []
        
        # Collect from random offsets
        datasets = []
        batch_size = 100
        
        import random
        offsets = random.sample(range(0, total, batch_size), 
                               min(n // batch_size + 1, total // batch_size))
        
        for offset in offsets:
            result = self._request('package_search', {
                'rows': batch_size,
                'start': offset
            })
            datasets.extend(result.get('results', []))
            if len(datasets) >= n:
                break
        
        return datasets[:n]
    
    def _collect_all(self, n: int) -> List[Dict]:
        """Collect datasets sequentially."""
        datasets = []
        batch_size = 100
        offset = 0
        
        while len(datasets) < n:
            result = self._request('package_search', {
                'rows': min(batch_size, n - len(datasets)),
                'start': offset
            })
            batch = result.get('results', [])
            if not batch:
                break
            datasets.extend(batch)
            offset += batch_size
            logger.info(f"  Collected {len(datasets)}/{n} datasets")
        
        return datasets


# ============================================================================
# METADATA ANALYZER
# ============================================================================

class MetadataAnalyzer:
    """
    Analyzes collected metadata to compute quality metrics
    and statistical distributions.
    """
    
    # DCAT-AP CH fields used for completeness scoring
    COMPLETENESS_FIELDS = [
        'title', 'description', 'resources', 'tags', 'organization',
        'groups', 'license_id', 'issued', 'modified', 'publisher', 
        'contact_point', 'temporal', 'spatial', 'accrual_periodicity'
    ]
    
    def __init__(self):
        self.datasets = []
        self.metrics = []
    
    def analyze(self, datasets: List[Dict]) -> List[DatasetMetrics]:
        """
        Analyze a collection of datasets.
        
        Args:
            datasets: List of dataset metadata from CKAN API
            
        Returns:
            List of computed DatasetMetrics
        """
        self.datasets = datasets
        self.metrics = []
        
        for ds in datasets:
            try:
                metrics = self._compute_dataset_metrics(ds)
                self.metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Error analyzing dataset {ds.get('name')}: {e}")
        
        return self.metrics
    
    def _compute_dataset_metrics(self, ds: Dict) -> DatasetMetrics:
        """Compute all metrics for a single dataset."""
        now = datetime.now()
        
        # Basic metadata lengths
        title = ds.get('title', {})
        description = ds.get('description', {})
        
        # Handle multilingual fields
        if isinstance(title, dict):
            title_length = max(len(v) for v in title.values()) if title else 0
            languages = list(title.keys())
            primary_lang = max(title.keys(), key=lambda k: len(title.get(k, ''))) if title else 'de'
        else:
            title_length = len(str(title)) if title else 0
            languages = ['de']
            primary_lang = 'de'
        
        if isinstance(description, dict):
            desc_length = max(len(v) for v in description.values()) if description else 0
        else:
            desc_length = len(str(description)) if description else 0
        
        # Tags and resources
        tags = ds.get('tags', []) or ds.get('keywords', []) or []
        resources = ds.get('resources', []) or []
        
        # Temporal metrics
        modified_str = ds.get('metadata_modified', ds.get('modified', ''))
        created_str = ds.get('metadata_created', ds.get('issued', ''))
        
        days_modified = self._compute_days_since(modified_str, now)
        days_created = self._compute_days_since(created_str, now) if created_str else None
        
        # Organization
        org = ds.get('organization', {}) or {}
        org_name = org.get('title', org.get('name', 'Unknown'))
        org_type = self._classify_organization(org_name)
        
        # Resource analysis
        formats = []
        for r in resources:
            fmt = r.get('format', '').upper()
            if fmt:
                formats.append(fmt)
        
        # Themes
        groups = ds.get('groups', []) or []
        themes = [g.get('name', g) if isinstance(g, dict) else g for g in groups]
        
        # Completeness and documentation scores
        completeness = self._compute_completeness(ds)
        documentation = self._compute_documentation_score(ds, desc_length, len(tags))
        
        # Spatial coverage
        spatial = ds.get('spatial', ds.get('coverage', None))
        if isinstance(spatial, dict):
            spatial = spatial.get('value', str(spatial))
        
        return DatasetMetrics(
            dataset_id=ds.get('id', ''),
            dataset_name=ds.get('name', ''),
            title_length=title_length,
            description_length=desc_length,
            tag_count=len(tags),
            resource_count=len(resources),
            completeness_score=completeness,
            documentation_score=documentation,
            days_since_modified=days_modified,
            days_since_created=days_created,
            organization=org_name,
            organization_type=org_type,
            languages_available=languages,
            primary_language=primary_lang,
            format_count=len(set(formats)),
            formats_available=list(set(formats)),
            themes=themes,
            spatial_coverage=spatial
        )
    
    def _compute_days_since(self, date_str: str, now: datetime) -> int:
        """Compute days since a date string."""
        if not date_str:
            return 1000  # Default for missing dates
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return (now - dt.replace(tzinfo=None)).days
        except:
            return 1000
    
    def _classify_organization(self, org_name: str) -> str:
        """Classify organization type."""
        name_lower = org_name.lower()
        
        federal_indicators = ['bundesamt', 'federal', 'eidgen', 'confédération', 
                             'confederazione', 'swiss', 'schweiz', 'suisse']
        cantonal_indicators = ['kanton', 'canton', 'cantone', 'kt.', 'kantonale']
        municipal_indicators = ['stadt', 'ville', 'città', 'gemeinde', 'commune']
        
        if any(ind in name_lower for ind in federal_indicators):
            return 'federal'
        elif any(ind in name_lower for ind in cantonal_indicators):
            return 'cantonal'
        elif any(ind in name_lower for ind in municipal_indicators):
            return 'municipal'
        else:
            return 'other'
    
    def _compute_completeness(self, ds: Dict) -> float:
        """Compute DCAT-AP CH completeness score."""
        filled = 0
        for field in self.COMPLETENESS_FIELDS:
            value = ds.get(field)
            if value:
                if isinstance(value, (list, dict)):
                    if len(value) > 0:
                        filled += 1
                else:
                    filled += 1
        return filled / len(self.COMPLETENESS_FIELDS)
    
    def _compute_documentation_score(self, ds: Dict, desc_len: int, tag_count: int) -> float:
        """Compute documentation quality score."""
        # Based on description length, tags, and resource descriptions
        desc_score = min(desc_len / 500, 1.0)  # Normalize to 500 chars
        tag_score = min(tag_count / 5, 1.0)    # Normalize to 5 tags
        
        # Check resource descriptions
        resources = ds.get('resources', []) or []
        res_desc_count = sum(1 for r in resources if r.get('description'))
        res_score = res_desc_count / len(resources) if resources else 0
        
        return 0.5 * desc_score + 0.3 * tag_score + 0.2 * res_score
    
    def compute_statistics(self) -> PortalStatistics:
        """Compute aggregated portal statistics."""
        if not self.metrics:
            raise ValueError("No metrics computed. Call analyze() first.")
        
        df = pd.DataFrame([asdict(m) for m in self.metrics])
        
        # Distribution statistics
        desc_stats = self._compute_dist_stats(df['description_length'])
        tag_stats = self._compute_dist_stats(df['tag_count'])
        res_stats = self._compute_dist_stats(df['resource_count'])
        comp_stats = self._compute_dist_stats(df['completeness_score'])
        recency_stats = self._compute_dist_stats(df['days_since_modified'])
        
        # Percentiles for fuzzy calibration
        recency_pctls = {
            'p10': df['days_since_modified'].quantile(0.10),
            'p25': df['days_since_modified'].quantile(0.25),
            'p50': df['days_since_modified'].quantile(0.50),
            'p75': df['days_since_modified'].quantile(0.75),
            'p90': df['days_since_modified'].quantile(0.90)
        }
        
        comp_pctls = {
            'p10': df['completeness_score'].quantile(0.10),
            'p25': df['completeness_score'].quantile(0.25),
            'p50': df['completeness_score'].quantile(0.50),
            'p75': df['completeness_score'].quantile(0.75),
            'p90': df['completeness_score'].quantile(0.90)
        }
        
        res_pctls = {
            'p10': df['resource_count'].quantile(0.10),
            'p25': df['resource_count'].quantile(0.25),
            'p50': df['resource_count'].quantile(0.50),
            'p75': df['resource_count'].quantile(0.75),
            'p90': df['resource_count'].quantile(0.90)
        }
        
        # Organization distribution
        org_counts = df['organization'].value_counts().head(20)
        top_orgs = list(zip(org_counts.index, org_counts.values))
        
        org_type_dist = df['organization_type'].value_counts().to_dict()
        
        # Theme distribution
        all_themes = []
        for themes in df['themes']:
            all_themes.extend(themes if themes else [])
        theme_dist = pd.Series(all_themes).value_counts().to_dict()
        
        # Format distribution
        all_formats = []
        for formats in df['formats_available']:
            all_formats.extend(formats if formats else [])
        format_dist = pd.Series(all_formats).value_counts().to_dict()
        
        # Update frequency
        def categorize_recency(days):
            if days <= 30:
                return 'Last month'
            elif days <= 90:
                return 'Last quarter'
            elif days <= 365:
                return 'Last year'
            elif days <= 730:
                return 'Last 2 years'
            else:
                return 'Older'
        
        df['update_category'] = df['days_since_modified'].apply(categorize_recency)
        update_dist = df['update_category'].value_counts().to_dict()
        
        return PortalStatistics(
            total_datasets=len(self.metrics),
            total_organizations=df['organization'].nunique(),
            total_themes=len(theme_dist),
            description_length_stats=desc_stats,
            tag_count_stats=tag_stats,
            resource_count_stats=res_stats,
            completeness_stats=comp_stats,
            recency_stats=recency_stats,
            recency_percentiles=recency_pctls,
            completeness_percentiles=comp_pctls,
            resources_percentiles=res_pctls,
            top_organizations=top_orgs,
            organization_type_distribution=org_type_dist,
            theme_distribution=theme_dist,
            format_distribution=format_dist,
            update_frequency_distribution=update_dist,
            collection_timestamp=datetime.now().isoformat()
        )
    
    def _compute_dist_stats(self, series: pd.Series) -> Dict[str, float]:
        """Compute distribution statistics."""
        return {
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'q25': float(series.quantile(0.25)),
            'q75': float(series.quantile(0.75))
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame."""
        return pd.DataFrame([asdict(m) for m in self.metrics])
    
    def export_results(self, output_dir: str = "data/processed"):
        """Export analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export metrics as CSV
        df = self.to_dataframe()
        csv_path = os.path.join(output_dir, f"metadata_metrics_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported metrics to {csv_path}")
        
        # Export statistics as JSON
        if self.metrics:
            stats = self.compute_statistics()
            stats_path = os.path.join(output_dir, f"portal_statistics_{timestamp}.json")
            with open(stats_path, 'w') as f:
                json.dump(asdict(stats), f, indent=2, default=str)
            logger.info(f"Exported statistics to {stats_path}")
        
        return csv_path


# ============================================================================
# MAIN COLLECTION WORKFLOW
# ============================================================================

def run_collection_phase(n_datasets: int = 600) -> Tuple[List[Dict], PortalStatistics]:
    """
    Execute the complete Phase 2 collection and analysis workflow.
    
    Args:
        n_datasets: Number of datasets to collect
        
    Returns:
        Tuple of (raw datasets, computed statistics)
    """
    logger.info("="*60)
    logger.info("PHASE 2: METADATA COLLECTION AND EMPIRICAL ANALYSIS")
    logger.info("="*60)
    
    # Step 1: Initialize collector
    collector = SwissOGDCollector(rate_limit_delay=0.3)
    
    # Step 2: Get portal overview
    logger.info("\nStep 1: Retrieving portal information...")
    portal_info = collector.get_portal_info()
    logger.info(f"  Total datasets on portal: {portal_info.get('total_datasets', 'Unknown')}")
    logger.info(f"  Total organizations: {portal_info.get('total_organizations', 'Unknown')}")
    logger.info(f"  Total themes: {portal_info.get('total_themes', 'Unknown')}")
    
    # Step 3: Collect representative sample
    logger.info(f"\nStep 2: Collecting {n_datasets} representative datasets...")
    datasets = collector.collect_datasets(n_datasets, strategy='representative')
    logger.info(f"  Collected {len(datasets)} datasets")
    
    # Step 4: Analyze metadata
    logger.info("\nStep 3: Analyzing metadata quality...")
    analyzer = MetadataAnalyzer()
    metrics = analyzer.analyze(datasets)
    logger.info(f"  Analyzed {len(metrics)} datasets")
    
    # Step 5: Compute statistics
    logger.info("\nStep 4: Computing portal statistics...")
    statistics = analyzer.compute_statistics()
    
    # Step 6: Export results
    logger.info("\nStep 5: Exporting results...")
    analyzer.export_results("data/processed")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("COLLECTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Datasets analyzed: {statistics.total_datasets}")
    logger.info(f"Organizations covered: {statistics.total_organizations}")
    logger.info(f"Themes covered: {statistics.total_themes}")
    logger.info(f"\nRecency (days since modified):")
    logger.info(f"  Mean: {statistics.recency_stats['mean']:.0f}")
    logger.info(f"  Median: {statistics.recency_stats['median']:.0f}")
    logger.info(f"  P25-P75: {statistics.recency_percentiles['p25']:.0f} - {statistics.recency_percentiles['p75']:.0f}")
    logger.info(f"\nCompleteness:")
    logger.info(f"  Mean: {statistics.completeness_stats['mean']:.2%}")
    logger.info(f"  Median: {statistics.completeness_stats['median']:.2%}")
    logger.info(f"\nResources per dataset:")
    logger.info(f"  Mean: {statistics.resource_count_stats['mean']:.1f}")
    logger.info(f"  Median: {statistics.resource_count_stats['median']:.0f}")
    
    return datasets, statistics


if __name__ == "__main__":
    datasets, stats = run_collection_phase(600)
