"""
Comprehensive Data Collection Pipeline for Swiss OGD Research

This module collects ALL metadata from opendata.swiss for statistical
analysis to calibrate the fuzzy ranking system.

Research Purpose:
- Understand actual metadata quality distributions across the portal
- Calibrate membership function parameters based on real data
- Identify patterns in metadata completeness, recency, and resource availability
- Support RQ1: How can fuzzy logic model vagueness in OGD metadata?

Author: Deep Shukla
Thesis: Improving Access to Swiss OGD through Fuzzy HCIR
"""

import requests
import json
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import concurrent.futures
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadataRecord:
    """
    Complete metadata record for a single dataset.
    Designed for statistical analysis and fuzzy system calibration.
    """
    # Identification
    id: str
    name: str
    title: Dict[str, str]  # Multilingual titles
    
    # Temporal Information
    metadata_created: Optional[str] = None
    metadata_modified: Optional[str] = None
    temporal_coverage_start: Optional[str] = None
    temporal_coverage_end: Optional[str] = None
    accrual_periodicity: Optional[str] = None
    
    # Organization & Attribution  
    organization_name: Optional[str] = None
    organization_title: Optional[str] = None
    publisher: Optional[str] = None
    contact_points: List[str] = field(default_factory=list)
    
    # Content Description
    description: Dict[str, str] = field(default_factory=dict)
    keywords: List[Dict[str, str]] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    
    # Resources
    num_resources: int = 0
    resource_formats: List[str] = field(default_factory=list)
    has_api: bool = False
    has_download: bool = False
    
    # Quality Indicators
    license_id: Optional[str] = None
    spatial_coverage: Optional[str] = None
    language: List[str] = field(default_factory=list)
    
    # Computed Metrics (for analysis)
    days_since_modified: Optional[int] = None
    title_length: int = 0
    description_length: int = 0
    num_keywords: int = 0
    completeness_score: float = 0.0
    
    def compute_metrics(self):
        """Compute derived metrics for analysis."""
        # Days since modified
        if self.metadata_modified:
            try:
                modified_date = datetime.fromisoformat(
                    self.metadata_modified.replace('Z', '+00:00')
                )
                self.days_since_modified = (datetime.now(modified_date.tzinfo) - modified_date).days
            except:
                self.days_since_modified = None
        
        # Title length (max across languages)
        if self.title:
            self.title_length = max(len(str(v)) for v in self.title.values() if v)
        
        # Description length (max across languages)
        if self.description:
            self.description_length = max(
                len(str(v)) for v in self.description.values() if v
            ) if any(self.description.values()) else 0
        
        # Number of keywords
        self.num_keywords = len(self.keywords)
        
        # Completeness score
        self.completeness_score = self._calculate_completeness()
    
    def _calculate_completeness(self) -> float:
        """
        Calculate metadata completeness score based on DCAT-AP CH requirements.
        
        Weights based on importance for discoverability:
        - Mandatory fields: Higher weight
        - Recommended fields: Medium weight
        - Optional fields: Lower weight
        """
        checks = {
            # Mandatory (weight: 2)
            'title': (bool(self.title and any(self.title.values())), 2),
            'description': (bool(self.description and any(self.description.values())), 2),
            'publisher': (bool(self.publisher or self.organization_name), 2),
            
            # Recommended (weight: 1.5)
            'keywords': (self.num_keywords >= 1, 1.5),
            'themes': (len(self.themes) >= 1, 1.5),
            'license': (bool(self.license_id), 1.5),
            'temporal_coverage': (bool(self.temporal_coverage_start or self.temporal_coverage_end), 1.5),
            
            # Optional (weight: 1)
            'contact': (len(self.contact_points) >= 1, 1),
            'spatial': (bool(self.spatial_coverage), 1),
            'language': (len(self.language) >= 1, 1),
            'accrual': (bool(self.accrual_periodicity), 1),
            'resources': (self.num_resources >= 1, 1),
            'multiple_formats': (len(set(self.resource_formats)) >= 2, 1),
        }
        
        weighted_sum = sum(score * weight for (score, weight) in checks.values() if score)
        max_possible = sum(weight for (_, weight) in checks.values())
        
        return weighted_sum / max_possible if max_possible > 0 else 0.0


class OpenDataSwissCollector:
    """
    Comprehensive data collector for opendata.swiss portal.
    
    Designed for research-grade data collection:
    - Handles pagination for large-scale collection
    - Implements rate limiting to respect API
    - Provides progress tracking
    - Supports incremental collection
    """
    
    BASE_URL = "https://opendata.swiss/api/3/action"
    
    def __init__(self, cache_dir: str = "data/raw"):
        """
        Initialize the collector.
        
        Args:
            cache_dir: Directory for caching raw API responses
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OGD-Research-Thesis/1.0 (University of Fribourg)'
        })
        
        # Rate limiting
        self.request_delay = 0.2  # seconds between requests
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()
    
    def _api_call(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a rate-limited API call."""
        self._rate_limit()
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def get_portal_statistics(self) -> Dict:
        """
        Get comprehensive portal statistics.
        
        Returns:
            Dictionary with portal-wide statistics
        """
        logger.info("Collecting portal statistics...")
        
        stats = {}
        
        # Total datasets
        result = self._api_call('package_search', {'rows': 0})
        stats['total_datasets'] = result['result']['count']
        
        # Themes/groups
        result = self._api_call('group_list', {'all_fields': True})
        stats['themes'] = [
            {'name': g['name'], 'title': g.get('title', g['name']), 
             'count': g.get('package_count', 0)}
            for g in result['result']
        ] if isinstance(result['result'], list) and isinstance(result['result'][0], dict) else result['result']
        
        # Organizations
        result = self._api_call('organization_list', {'all_fields': True})
        if isinstance(result['result'], list):
            if len(result['result']) > 0 and isinstance(result['result'][0], dict):
                stats['organizations'] = [
                    {'name': o['name'], 'title': o.get('title', o['name']),
                     'count': o.get('package_count', 0)}
                    for o in result['result']
                ]
            else:
                stats['organizations'] = result['result']
        
        stats['collection_timestamp'] = datetime.now().isoformat()
        
        return stats
    
    def collect_all_datasets(
        self,
        max_datasets: int = None,
        batch_size: int = 100,
        themes: List[str] = None,
        save_raw: bool = True
    ) -> List[DatasetMetadataRecord]:
        """
        Collect metadata for all datasets in the portal.
        
        Args:
            max_datasets: Maximum datasets to collect (None = all)
            batch_size: Number of datasets per API request
            themes: Filter by specific themes (None = all)
            save_raw: Whether to save raw API responses
            
        Returns:
            List of DatasetMetadataRecord objects
        """
        logger.info("Starting comprehensive data collection...")
        
        # Get total count
        query_params = {'rows': 0}
        if themes:
            query_params['fq'] = ' OR '.join([f'groups:{t}' for t in themes])
        
        result = self._api_call('package_search', query_params)
        total_available = result['result']['count']
        
        total_to_collect = min(total_available, max_datasets) if max_datasets else total_available
        logger.info(f"Collecting {total_to_collect} of {total_available} available datasets")
        
        all_records = []
        collected = 0
        
        while collected < total_to_collect:
            # Calculate batch
            remaining = total_to_collect - collected
            current_batch = min(batch_size, remaining)
            
            # Fetch batch
            params = {
                'rows': current_batch,
                'start': collected,
                'sort': 'metadata_modified desc'
            }
            if themes:
                params['fq'] = ' OR '.join([f'groups:{t}' for t in themes])
            
            try:
                result = self._api_call('package_search', params)
                datasets = result['result']['results']
                
                # Parse each dataset
                for ds in datasets:
                    record = self._parse_dataset(ds)
                    record.compute_metrics()
                    all_records.append(record)
                
                collected += len(datasets)
                
                # Progress logging
                if collected % 500 == 0 or collected >= total_to_collect:
                    logger.info(f"Progress: {collected}/{total_to_collect} ({100*collected/total_to_collect:.1f}%)")
                
                # Safety check
                if len(datasets) == 0:
                    logger.warning("Empty batch received, stopping collection")
                    break
                    
            except Exception as e:
                logger.error(f"Error at offset {collected}: {e}")
                time.sleep(2)  # Wait before retry
                continue
        
        logger.info(f"Collection complete: {len(all_records)} datasets")
        
        # Save raw data
        if save_raw:
            self._save_collection(all_records)
        
        return all_records
    
    def _parse_dataset(self, raw_data: Dict) -> DatasetMetadataRecord:
        """Parse raw CKAN dataset into structured record."""
        
        # Handle multilingual fields
        def get_multilingual(field_name: str) -> Dict[str, str]:
            result = {}
            for lang in ['en', 'de', 'fr', 'it']:
                key = f"{field_name}_{lang}" if lang != 'en' else field_name
                if key in raw_data and raw_data[key]:
                    result[lang] = raw_data[key]
            # Also check nested structure
            if field_name in raw_data and isinstance(raw_data[field_name], dict):
                result.update(raw_data[field_name])
            return result
        
        # Extract resources info
        resources = raw_data.get('resources', [])
        resource_formats = [r.get('format', '').upper() for r in resources if r.get('format')]
        has_api = any('api' in r.get('protocol', '').lower() or 
                      r.get('format', '').upper() in ['API', 'WMS', 'WFS', 'SPARQL']
                      for r in resources)
        
        # Extract themes from groups
        themes = [g.get('name', '') for g in raw_data.get('groups', []) if isinstance(g, dict)]
        
        # Extract keywords
        keywords = []
        for tag in raw_data.get('tags', []) or raw_data.get('keywords', []):
            if isinstance(tag, dict):
                keywords.append(tag)
            else:
                keywords.append({'name': str(tag)})
        
        record = DatasetMetadataRecord(
            id=raw_data.get('id', ''),
            name=raw_data.get('name', ''),
            title=get_multilingual('title') or {'en': raw_data.get('title', '')},
            
            metadata_created=raw_data.get('metadata_created'),
            metadata_modified=raw_data.get('metadata_modified'),
            temporal_coverage_start=raw_data.get('temporal_coverage_start_date') or 
                                    raw_data.get('temporalCoverageStart'),
            temporal_coverage_end=raw_data.get('temporal_coverage_end_date') or
                                  raw_data.get('temporalCoverageEnd'),
            accrual_periodicity=raw_data.get('accrual_periodicity') or 
                                raw_data.get('frequency'),
            
            organization_name=raw_data.get('organization', {}).get('name') if raw_data.get('organization') else None,
            organization_title=raw_data.get('organization', {}).get('title') if raw_data.get('organization') else None,
            publisher=raw_data.get('publisher') or 
                      (raw_data.get('organization', {}).get('title') if raw_data.get('organization') else None),
            contact_points=raw_data.get('contact_points', []) or 
                          ([raw_data.get('maintainer_email')] if raw_data.get('maintainer_email') else []),
            
            description=get_multilingual('description') or get_multilingual('notes'),
            keywords=keywords,
            themes=themes,
            
            num_resources=len(resources),
            resource_formats=resource_formats,
            has_api=has_api,
            has_download=len(resources) > 0,
            
            license_id=raw_data.get('license_id') or raw_data.get('license_title'),
            spatial_coverage=raw_data.get('spatial') or raw_data.get('coverage'),
            language=raw_data.get('language', []) if isinstance(raw_data.get('language'), list) else []
        )
        
        return record
    
    def _save_collection(self, records: List[DatasetMetadataRecord]):
        """Save collected data for analysis."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_path = self.cache_dir / f"ogd_metadata_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON: {json_path}")
        
        # Save as CSV for analysis
        csv_path = self.cache_dir / f"ogd_metadata_{timestamp}.csv"
        self._save_as_csv(records, csv_path)
        logger.info(f"Saved CSV: {csv_path}")
    
    def _save_as_csv(self, records: List[DatasetMetadataRecord], path: Path):
        """Save records as CSV for statistical analysis."""
        import csv
        
        # Flatten fields for CSV
        fieldnames = [
            'id', 'name', 'title_en', 'title_de',
            'metadata_created', 'metadata_modified', 'days_since_modified',
            'organization_name', 'publisher',
            'description_length', 'num_keywords', 'num_themes',
            'num_resources', 'resource_formats', 'has_api',
            'license_id', 'completeness_score',
            'title_length'
        ]
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in records:
                row = {
                    'id': r.id,
                    'name': r.name,
                    'title_en': r.title.get('en', ''),
                    'title_de': r.title.get('de', ''),
                    'metadata_created': r.metadata_created,
                    'metadata_modified': r.metadata_modified,
                    'days_since_modified': r.days_since_modified,
                    'organization_name': r.organization_name,
                    'publisher': r.publisher,
                    'description_length': r.description_length,
                    'num_keywords': r.num_keywords,
                    'num_themes': len(r.themes),
                    'num_resources': r.num_resources,
                    'resource_formats': ','.join(r.resource_formats),
                    'has_api': r.has_api,
                    'license_id': r.license_id,
                    'completeness_score': round(r.completeness_score, 4),
                    'title_length': r.title_length
                }
                writer.writerow(row)


def main():
    """Main collection routine."""
    print("=" * 70)
    print("SWISS OGD COMPREHENSIVE DATA COLLECTION")
    print("For Master Thesis Research - University of Fribourg")
    print("=" * 70)
    
    collector = OpenDataSwissCollector(cache_dir="data/raw")
    
    # Get portal statistics first
    print("\n[1] Collecting portal statistics...")
    stats = collector.get_portal_statistics()
    print(f"    Total datasets: {stats['total_datasets']:,}")
    
    # Collect sample for initial analysis (full collection takes ~30 minutes)
    print("\n[2] Collecting dataset metadata (sample for testing)...")
    print("    Note: For full thesis, collect ALL datasets")
    
    records = collector.collect_all_datasets(
        max_datasets=500,  # Start with sample; use None for full collection
        batch_size=100,
        save_raw=True
    )
    
    # Quick statistics
    print("\n[3] Quick Analysis of Collected Data:")
    print(f"    Datasets collected: {len(records)}")
    
    if records:
        # Completeness distribution
        completeness_scores = [r.completeness_score for r in records]
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        print(f"    Average completeness: {avg_completeness:.2%}")
        
        # Recency distribution
        recency_values = [r.days_since_modified for r in records if r.days_since_modified is not None]
        if recency_values:
            avg_recency = sum(recency_values) / len(recency_values)
            print(f"    Average days since modified: {avg_recency:.0f}")
            print(f"    Min: {min(recency_values)}, Max: {max(recency_values)}")
        
        # Resource distribution
        resource_counts = [r.num_resources for r in records]
        avg_resources = sum(resource_counts) / len(resource_counts)
        print(f"    Average resources per dataset: {avg_resources:.1f}")
    
    print("\n" + "=" * 70)
    print("Collection complete! Data saved to data/raw/")
    print("=" * 70)


if __name__ == "__main__":
    main()
