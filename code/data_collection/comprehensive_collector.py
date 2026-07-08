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
import hashlib
import shutil
import time
import logging
import os
import platform
import subprocess
import statistics
import sys
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

COLLECTOR_VERSION = "1.1"


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

        self.portal_statistics: Dict[str, Any] = {}
        self.collection_context: Dict[str, Any] = {}
        self.collection_started_at: Optional[datetime] = None
        self.collection_finished_at: Optional[datetime] = None
        self.api_request_count: int = 0
        
        # Rate limiting
        self.request_delay = 0.2  # seconds between requests
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()
    
    def _api_call(self, endpoint: str, params: Dict = None, retries: int = 3) -> Dict:
        """Make a rate-limited API call with basic retry handling."""
        url = f"{self.BASE_URL}/{endpoint}"
        last_error = None

        for attempt in range(1, retries + 1):
            self._rate_limit()
            self.api_request_count += 1
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                last_error = e
                logger.warning(f"API call failed (attempt {attempt}/{retries}) for {endpoint}: {e}")
                if attempt < retries:
                    time.sleep(min(2.0 * attempt, 5.0))

        logger.error(f"API call failed after {retries} attempts: {last_error}")
        raise last_error if last_error else requests.RequestException(f"API call failed: {endpoint}")
    
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
        groups = result.get('result', [])
        stats['themes'] = [
            {'name': g.get('name', ''), 'title': g.get('title', g.get('name', '')),
             'count': g.get('package_count', 0)}
            for g in groups
        ] if isinstance(groups, list) and (not groups or isinstance(groups[0], dict)) else groups
        
        # Organizations
        result = self._api_call('organization_list', {'all_fields': True})
        organizations = result.get('result', [])
        if isinstance(organizations, list):
            if len(organizations) > 0 and isinstance(organizations[0], dict):
                stats['organizations'] = [
                    {'name': o.get('name', ''), 'title': o.get('title', o.get('name', '')),
                     'count': o.get('package_count', 0)}
                    for o in organizations
                ]
            else:
                stats['organizations'] = organizations
        else:
            stats['organizations'] = organizations
        
        stats['collection_timestamp'] = datetime.now().isoformat()
        self.portal_statistics = stats
        
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
        self.collection_started_at = datetime.now()
        self.collection_finished_at = None
        self.collection_context = {
            'batch_size': batch_size,
            'api_endpoint': f"{self.BASE_URL}/package_search",
            'requested_max_datasets': max_datasets,
            'themes': themes or []
        }
        
        # Get total count
        query_params = {'rows': 0}
        if themes:
            query_params['fq'] = ' OR '.join([f'groups:{t}' for t in themes])
        
        result = self._api_call('package_search', query_params)
        total_available = result['result']['count']
        self.collection_context['portal_total_datasets'] = total_available
        
        total_to_collect = min(total_available, max_datasets) if max_datasets else total_available
        self.collection_context['datasets_downloaded_target'] = total_to_collect
        logger.info(f"Portal reports {total_available:,} datasets")
        logger.info(f"Collecting {total_to_collect:,} of {total_available:,} available datasets")
        
        all_records = []
        collected = 0
        batch_number = 0
        
        while collected < total_to_collect:
            # Calculate batch
            remaining = total_to_collect - collected
            current_batch = min(batch_size, remaining)
            batch_number += 1
            logger.info(
                f"Downloading batch {batch_number}: offset={collected}, size={current_batch}"
            )
            
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
                progress = 100 * collected / total_to_collect if total_to_collect else 100.0
                logger.info(f"Progress: {collected:,}/{total_to_collect:,} ({progress:.1f}%)")
                
                # Safety check
                if len(datasets) == 0:
                    logger.warning("Empty batch received, stopping collection")
                    break
                    
            except Exception as e:
                logger.error(f"Error at offset {collected}: {e}")
                time.sleep(2)  # Wait before retry
                continue
        
        logger.info(f"Collection complete: {len(all_records)} datasets")
        self.collection_finished_at = datetime.now()
        self.collection_context['collection_start_time'] = self.collection_started_at.isoformat() if self.collection_started_at else None
        self.collection_context['collection_end_time'] = self.collection_finished_at.isoformat() if self.collection_finished_at else None
        if self.collection_started_at and self.collection_finished_at:
            self.collection_context['collection_duration_seconds'] = (self.collection_finished_at - self.collection_started_at).total_seconds()
        
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
        """Save collected data and generate reproducible snapshot artifacts."""
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

        immutable_dir = Path("data/snapshots") / timestamp
        latest_dir = Path("data/snapshots/latest")
        immutable_dir.mkdir(parents=True, exist_ok=True)
        latest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving snapshot artifacts to: {immutable_dir}")

        snapshot_payload = [asdict(r) for r in records]
        snapshot_json_path = immutable_dir / "snapshot.json"
        statistics_path = immutable_dir / "statistics.json"
        with open(snapshot_json_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Saving snapshot JSON: {snapshot_json_path}")

        statistics_payload = self._build_snapshot_statistics(records, timestamp)
        with open(statistics_path, 'w', encoding='utf-8') as f:
            json.dump(statistics_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Saving statistics: {statistics_path}")

        metadata_payload: Dict[str, Any]
        readme_text: str
        try:
            metadata_payload = self._build_snapshot_metadata(
                timestamp=timestamp,
                raw_json_path=json_path,
                raw_csv_path=csv_path,
                snapshot_json_path=snapshot_json_path,
                statistics_path=statistics_path,
                record_count=len(records),
                statistics_payload=statistics_payload,
            )

            readme_text = self._build_snapshot_readme(metadata_payload, statistics_payload)

            metadata_path = immutable_dir / "snapshot_metadata.json"
            readme_path = immutable_dir / "README.md"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_payload, f, ensure_ascii=False, indent=2)
            logger.info(f"Saving metadata: {metadata_path}")

            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_text)
            logger.info(f"Saving README: {readme_path}")

            self._copy_snapshot_to_latest(immutable_dir, latest_dir)
        except Exception as e:
            logger.exception(f"Snapshot metadata generation failed; preserving immutable snapshot anyway: {e}")
            self._copy_snapshot_to_latest(immutable_dir, latest_dir, include_metadata=False)

        logger.info("Finished successfully")

    def _copy_snapshot_to_latest(self, source_dir: Path, latest_dir: Path, include_metadata: bool = True):
        """Copy snapshot artifacts from an immutable folder into the latest folder."""
        latest_dir.mkdir(parents=True, exist_ok=True)
        filenames = ["snapshot.json", "statistics.json"]
        if include_metadata:
            filenames.extend(["snapshot_metadata.json", "README.md"])

        for filename in filenames:
            source = source_dir / filename
            if source.exists():
                shutil.copy2(source, latest_dir / filename)

    def _sha256_file(self, path: Path) -> str:
        """Compute the SHA256 checksum for a file."""
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _get_git_info(self) -> Dict[str, Optional[str]]:
        """Return git branch and commit hash if the repository is inside a git checkout."""
        try:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path.cwd(),
            ).stdout.strip()
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path.cwd(),
            ).stdout.strip()
            return {
                'git_branch': branch or None,
                'git_commit_hash': commit or None,
            }
        except Exception:
            return {
                'git_branch': None,
                'git_commit_hash': None,
            }

    def _build_snapshot_statistics(self, records: List[DatasetMetadataRecord], collection_timestamp: str) -> Dict[str, Any]:
        """Build snapshot-level statistics from collected dataset records."""
        def mean(values: List[float]) -> Optional[float]:
            return statistics.fmean(values) if values else None

        def median(values: List[float]) -> Optional[float]:
            return statistics.median(values) if values else None

        def percentile(values: List[float], pct: float) -> Optional[float]:
            if not values:
                return None
            sorted_values = sorted(values)
            if len(sorted_values) == 1:
                return float(sorted_values[0])
            position = (len(sorted_values) - 1) * pct / 100.0
            lower = int(position)
            upper = min(lower + 1, len(sorted_values) - 1)
            weight = position - lower
            return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

        def stddev(values: List[float]) -> Optional[float]:
            return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None

        def distribution_counter(values: List[Any]) -> Dict[str, Dict[str, float]]:
            counter = Counter(str(v) for v in values if v not in (None, '', []))
            total = sum(counter.values())
            return {
                key: {
                    'count': count,
                    'percentage': round((count / total) * 100, 2) if total else 0.0
                }
                for key, count in counter.most_common()
            }

        def top_items(values: List[Any], limit: int = 20) -> List[Dict[str, Any]]:
            counter = Counter(str(v) for v in values if v not in (None, '', []))
            total = sum(counter.values())
            return [
                {
                    'name': name,
                    'count': count,
                    'percentage': round((count / total) * 100, 2) if total else 0.0,
                }
                for name, count in counter.most_common(limit)
            ]

        def count_ratio(condition_count: int, total_count: int) -> Dict[str, float]:
            return {
                'count': condition_count,
                'percentage': round((condition_count / total_count) * 100, 2) if total_count else 0.0,
            }

        total = len(records)
        days_values = [r.days_since_modified for r in records if isinstance(r.days_since_modified, int)]
        completeness_values = [r.completeness_score for r in records]
        title_lengths = [r.title_length for r in records]
        description_lengths = [r.description_length for r in records]
        keyword_counts = [r.num_keywords for r in records]
        resource_counts = [r.num_resources for r in records]
        dataset_ids = [r.id for r in records]
        duplicate_counts = Counter(dataset_ids)
        duplicate_dataset_ids = [dataset_id for dataset_id, count in duplicate_counts.items() if count > 1]
        duplicate_dataset_count = sum(count - 1 for count in duplicate_counts.values() if count > 1)

        missing_title_count = sum(1 for r in records if not (r.title and any(v for v in r.title.values())))
        missing_description_count = sum(1 for r in records if not (r.description and any(v for v in r.description.values())))
        missing_publisher_count = sum(1 for r in records if not (r.publisher or r.organization_name))
        missing_license_count = sum(1 for r in records if not r.license_id)
        missing_keywords_count = sum(1 for r in records if not r.keywords)
        missing_language_count = sum(1 for r in records if not r.language)
        missing_resources_count = sum(1 for r in records if r.num_resources == 0)

        completeness_distribution = Counter(round(r.completeness_score, 2) for r in records)
        completeness_distribution_payload = {
            str(bucket): {
                'count': count,
                'percentage': round((count / total) * 100, 2) if total else 0.0,
            }
            for bucket, count in sorted(completeness_distribution.items(), key=lambda item: float(item[0]))
        }

        statistics_payload = {
            'collection_timestamp': collection_timestamp,
            'portal_total_datasets': self.collection_context.get('portal_total_datasets', len(records)),
            'datasets_downloaded': total,
            'download_duration_seconds': self.collection_context.get('collection_duration_seconds'),
            'batch_size': self.collection_context.get('batch_size'),
            'api_endpoint': self.collection_context.get('api_endpoint', f"{self.BASE_URL}/package_search"),
            'collector_version': COLLECTOR_VERSION,
            'average_completeness': mean(completeness_values),
            'average_title_length': mean(title_lengths),
            'average_description_length': mean(description_lengths),
            'average_resources_per_dataset': mean(resource_counts),
            'average_keywords': mean(keyword_counts),
            'average_days_since_modified': mean(days_values),
            'median_days_since_modified': median(days_values),
            'minimum_days_since_modified': min(days_values) if days_values else None,
            'maximum_days_since_modified': max(days_values) if days_values else None,
            'duplicate_dataset_count': duplicate_dataset_count,
            'duplicate_dataset_ids': duplicate_dataset_ids,
            'datasets_with_api': count_ratio(sum(1 for r in records if r.has_api), total),
            'datasets_with_download': count_ratio(sum(1 for r in records if r.has_download), total),
            'datasets_without_resources': count_ratio(sum(1 for r in records if r.num_resources == 0), total),
            'missing_title_count': missing_title_count,
            'missing_description_count': missing_description_count,
            'missing_publisher_count': missing_publisher_count,
            'missing_license_count': missing_license_count,
            'missing_keywords_count': missing_keywords_count,
            'missing_language_count': missing_language_count,
            'missing_resources_count': missing_resources_count,
            'languages_distribution': distribution_counter(language for record in records for language in record.language),
            'resource_format_distribution': distribution_counter(format_name for record in records for format_name in record.resource_formats),
            'theme_distribution': distribution_counter(theme for record in records for theme in record.themes),
            'organization_distribution': distribution_counter(record.organization_name or record.organization_title for record in records),
            'license_distribution': distribution_counter(record.license_id for record in records),
            'resource_count_distribution': distribution_counter(resource_counts),
            'top_20_publishers': top_items(record.publisher or record.organization_title or record.organization_name for record in records),
            'top_20_resource_formats': top_items(format_name for record in records for format_name in record.resource_formats),
            'top_20_themes': top_items(theme for record in records for theme in record.themes),
            'top_20_languages': top_items(language for record in records for language in record.language),
            'metadata_completeness_distribution': completeness_distribution_payload,
            'summary_statistics': {
                'days_since_modified': {
                    'count': len(days_values),
                    'mean': mean(days_values),
                    'median': median(days_values),
                    'std_dev': stddev(days_values),
                    'min': min(days_values) if days_values else None,
                    'max': max(days_values) if days_values else None,
                    'p10': percentile(days_values, 10),
                    'p25': percentile(days_values, 25),
                    'p50': percentile(days_values, 50),
                    'p75': percentile(days_values, 75),
                    'p90': percentile(days_values, 90),
                },
                'completeness_score': {
                    'count': len(completeness_values),
                    'mean': mean(completeness_values),
                    'median': median(completeness_values),
                    'std_dev': stddev(completeness_values),
                    'min': min(completeness_values) if completeness_values else None,
                    'max': max(completeness_values) if completeness_values else None,
                    'p25': percentile(completeness_values, 25),
                    'p75': percentile(completeness_values, 75),
                    'p95': percentile(completeness_values, 95),
                },
                'title_length': {
                    'count': len(title_lengths),
                    'mean': mean(title_lengths),
                    'median': median(title_lengths),
                    'std_dev': stddev(title_lengths),
                    'min': min(title_lengths) if title_lengths else None,
                    'max': max(title_lengths) if title_lengths else None,
                },
                'description_length': {
                    'count': len(description_lengths),
                    'mean': mean(description_lengths),
                    'median': median(description_lengths),
                    'std_dev': stddev(description_lengths),
                    'min': min(description_lengths) if description_lengths else None,
                    'max': max(description_lengths) if description_lengths else None,
                },
                'keywords_per_dataset': {
                    'count': len(keyword_counts),
                    'mean': mean(keyword_counts),
                    'median': median(keyword_counts),
                    'std_dev': stddev(keyword_counts),
                    'min': min(keyword_counts) if keyword_counts else None,
                    'max': max(keyword_counts) if keyword_counts else None,
                },
                'resources_per_dataset': {
                    'count': len(resource_counts),
                    'mean': mean(resource_counts),
                    'median': median(resource_counts),
                    'std_dev': stddev(resource_counts),
                    'min': min(resource_counts) if resource_counts else None,
                    'max': max(resource_counts) if resource_counts else None,
                }
            }
        }

        return statistics_payload

    def _build_snapshot_metadata(
        self,
        *,
        timestamp: str,
        raw_json_path: Path,
        raw_csv_path: Path,
        snapshot_json_path: Path,
        statistics_path: Path,
        record_count: int,
        statistics_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build descriptive metadata for the generated snapshot."""
        created_at = self.collection_finished_at.isoformat() if self.collection_finished_at else timestamp
        collection_duration = self.collection_context.get('collection_duration_seconds')

        return {
            'snapshot_name': 'latest',
            'created_at': created_at,
            'snapshot_timestamp': created_at,
            'portal_name': 'opendata.swiss',
            'api_endpoint': self.collection_context.get('api_endpoint', f"{self.BASE_URL}/package_search"),
            'collection_method': 'CKAN package_search',
            'collector_class': self.__class__.__name__,
            'repository': 'Master_thesis_project',
            'research_project': 'Improving Access to Swiss Open Government Data through Human-Centered Information Retrieval Using Fuzzy Logic-Based Ranking',
            'thesis': 'Improving Access to Swiss Open Government Data through Human-Centered Information Retrieval Using Fuzzy Logic-Based Ranking',
            'university': 'University of Fribourg',
            'batch_size': self.collection_context.get('batch_size'),
            'pagination_supported': True,
            'rate_limiting': f'{self.request_delay:.1f}s between requests',
            'total_api_requests': self.api_request_count,
            'datasets_downloaded': record_count,
            'collection_duration_seconds': collection_duration,
            'collection_start_time': self.collection_context.get('collection_start_time'),
            'collection_end_time': self.collection_context.get('collection_end_time'),
            'python_version': sys.version,
            'platform': platform.platform(),
            'timestamped_raw_json': str(raw_json_path),
            'timestamped_raw_csv': str(raw_csv_path),
            'snapshot_json': str(snapshot_json_path),
            'statistics_json': str(statistics_path),
            'snapshot_json_sha256': self._sha256_file(snapshot_json_path) if snapshot_json_path.exists() else None,
            'statistics_json_sha256': self._sha256_file(statistics_path) if statistics_path.exists() else None,
            **self._get_git_info(),
            'collector_version': COLLECTOR_VERSION,
            'portal_total_datasets': statistics_payload.get('portal_total_datasets'),
        }

    def _build_snapshot_readme(self, metadata: Dict[str, Any], statistics_payload: Dict[str, Any]) -> str:
        """Build a README describing the reproducible snapshot folder."""
        collection_duration = metadata.get('collection_duration_seconds')
        expected_runtime = 'roughly 30 minutes to several hours depending on portal speed and rate limiting'

        return f"""# Swiss OGD Snapshot

## Purpose
This folder contains a reproducible snapshot of the opendata.swiss portal collected for the Master's thesis research project.

## Collection Details
- Collection date: {metadata.get('created_at')}
- Snapshot timestamp: {metadata.get('snapshot_timestamp')}
- Portal: {metadata.get('portal_name')}
- API endpoint: {metadata.get('api_endpoint')}
- Collector: {metadata.get('collector_class')}
- Dataset count: {metadata.get('datasets_downloaded')}
- Portal total datasets reported: {metadata.get('portal_total_datasets')}
- Collector version: {metadata.get('collector_version')}
- Python version: {metadata.get('python_version')}
- Total API requests: {metadata.get('total_api_requests')}
- Git branch: {metadata.get('git_branch')}
- Git commit: {metadata.get('git_commit_hash')}

## Research Context
- Research project: {metadata.get('research_project')}
- Thesis: {metadata.get('thesis')}
- University: {metadata.get('university')}

## Folder Contents
- `snapshot.json`: every collected DatasetMetadataRecord serialized as UTF-8 JSON with indent=2.
- `statistics.json`: automatically computed portal and dataset statistics.
- `snapshot_metadata.json`: technical metadata about how the snapshot was produced.
- `README.md`: this documentation file.
- Timestamped raw exports are stored separately under `data/raw/` and are preserved.

## Integrity Checks
- SHA256 snapshot.json: {metadata.get('snapshot_json_sha256')}
- SHA256 statistics.json: {metadata.get('statistics_json_sha256')}

## How the Snapshot Was Produced
The collector uses CKAN `package_search` pagination with rate limiting and batches of {metadata.get('batch_size')} datasets. The default collection target is the full portal snapshot (`max_datasets=None`). The timestamped raw JSON and CSV exports are still written first, then the snapshot artifacts in `data/snapshots/latest/` are refreshed.

## How to Reproduce
1. Run `python -m code.data_collection.comprehensive_collector`.
2. The collector will fetch the full portal snapshot and overwrite this `latest` directory.
3. Review `snapshot.json`, `statistics.json`, and `snapshot_metadata.json` for the generated outputs.

## Runtime Notes
- Expected runtime: {expected_runtime}
- Measured collection duration seconds: {collection_duration}
- Rate limiting: {metadata.get('rate_limiting')}

## Reproducibility Statement
This snapshot is intended for thesis publication and long-term reproducibility. The raw timestamped exports remain available alongside the latest overwriteable snapshot so that future readers can compare the exact frozen snapshot against prior runs.

## Example Structure
```text
data/snapshots/latest/
  README.md
  snapshot.json
  snapshot_metadata.json
  statistics.json
```
"""
    
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
        max_datasets=None,
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
