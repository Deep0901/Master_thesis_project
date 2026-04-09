"""
CKAN API Client for Swiss Open Government Data Portal (opendata.swiss)

This module provides a client interface to interact with the CKAN API
for retrieving dataset metadata from opendata.swiss.

Research Context:
- Part of Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR"
- Student: Deep Shukla
- University of Fribourg, Human-IST
"""

import requests
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Structured representation of dataset metadata."""
    id: str
    title: Dict[str, str]  # Multilingual titles {de, fr, it, en}
    description: Dict[str, str]  # Multilingual descriptions
    keywords: List[str]
    organization: str
    created: datetime
    modified: datetime
    resources: List[Dict]
    themes: List[str]
    temporal_coverage: Optional[str]
    spatial_coverage: Optional[str]
    license: Optional[str]
    contact_point: Optional[str]
    publisher: Optional[str]
    
    @property
    def completeness_score(self) -> float:
        """Calculate basic metadata completeness (0-1)."""
        fields = [
            bool(self.title),
            bool(self.description),
            len(self.keywords) > 0,
            bool(self.organization),
            bool(self.resources),
            bool(self.themes),
            bool(self.temporal_coverage),
            bool(self.license)
        ]
        return sum(fields) / len(fields)
    
    @property
    def days_since_modified(self) -> int:
        """Calculate days since last modification."""
        return (datetime.now() - self.modified).days


class CKANClient:
    """
    Client for interacting with the opendata.swiss CKAN API.
    
    Implements data collection functionality for the thesis prototype.
    """
    
    BASE_URL = "https://opendata.swiss/api/3/action"
    
    def __init__(self, base_url: str = None, rate_limit: float = 0.5):
        """
        Initialize the CKAN client.
        
        Args:
            base_url: Optional custom API base URL
            rate_limit: Minimum seconds between API calls
        """
        self.base_url = base_url or self.BASE_URL
        self.rate_limit = rate_limit
        self._last_request_time = 0
        self.session = requests.Session()
        
    def _rate_limit_wait(self):
        """Ensure rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make a rate-limited request to the CKAN API.
        
        Args:
            endpoint: API endpoint name
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        self._rate_limit_wait()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("success"):
                raise ValueError(f"API returned error: {data.get('error')}")
                
            return data["result"]
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def search_datasets(
        self,
        query: str = "*:*",
        rows: int = 100,
        start: int = 0,
        filters: Dict[str, str] = None,
        sort: str = "metadata_modified desc"
    ) -> Dict[str, Any]:
        """
        Search for datasets using CKAN package_search API.
        
        Args:
            query: Search query string (Solr syntax)
            rows: Number of results to return
            start: Offset for pagination
            filters: Filter queries (fq parameter)
            sort: Sort order
            
        Returns:
            Search results with count and dataset list
        """
        params = {
            "q": query,
            "rows": rows,
            "start": start,
            "sort": sort
        }
        
        if filters:
            fq_parts = [f'{k}:"{v}"' for k, v in filters.items()]
            params["fq"] = " AND ".join(fq_parts)
        
        result = self._make_request("package_search", params)
        
        return {
            "count": result.get("count", 0),
            "datasets": result.get("results", [])
        }
    
    def get_dataset(self, dataset_id: str) -> Dict:
        """
        Retrieve full metadata for a specific dataset.
        
        Args:
            dataset_id: Dataset ID or name
            
        Returns:
            Complete dataset metadata
        """
        return self._make_request("package_show", {"id": dataset_id})
    
    def get_organization_list(self) -> List[Dict]:
        """Get list of all organizations publishing data."""
        return self._make_request("organization_list", {"all_fields": True})
    
    def get_group_list(self) -> List[Dict]:
        """Get list of thematic groups/categories."""
        return self._make_request("group_list", {"all_fields": True})
    
    def get_tag_list(self) -> List[str]:
        """Get list of all tags used in the portal."""
        return self._make_request("tag_list")
    
    def collect_all_datasets(
        self,
        batch_size: int = 100,
        max_datasets: int = None,
        themes: List[str] = None
    ) -> List[Dict]:
        """
        Collect all datasets from the portal with pagination.
        
        Args:
            batch_size: Number of datasets per API call
            max_datasets: Maximum total datasets to collect
            themes: Filter by specific themes (e.g., ["environment", "mobility"])
            
        Returns:
            List of all collected dataset metadata
        """
        all_datasets = []
        start = 0
        
        # Build filter for themes if specified
        filters = None
        if themes:
            filters = {"groups": " OR ".join(themes)}
        
        while True:
            logger.info(f"Fetching datasets {start} to {start + batch_size}...")
            
            result = self.search_datasets(
                rows=batch_size,
                start=start,
                filters=filters
            )
            
            datasets = result["datasets"]
            if not datasets:
                break
                
            all_datasets.extend(datasets)
            
            if max_datasets and len(all_datasets) >= max_datasets:
                all_datasets = all_datasets[:max_datasets]
                break
                
            start += batch_size
            
            if len(datasets) < batch_size:
                break
        
        logger.info(f"Collected {len(all_datasets)} datasets")
        return all_datasets
    
    def parse_dataset_metadata(self, raw_data: Dict) -> DatasetMetadata:
        """
        Parse raw API response into structured DatasetMetadata.
        
        Args:
            raw_data: Raw dataset dictionary from API
            
        Returns:
            Structured DatasetMetadata object
        """
        def extract_multilingual(field_data):
            """Extract multilingual content from various formats."""
            if isinstance(field_data, dict):
                return {
                    "de": field_data.get("de", ""),
                    "fr": field_data.get("fr", ""),
                    "it": field_data.get("it", ""),
                    "en": field_data.get("en", "")
                }
            elif isinstance(field_data, str):
                return {"de": field_data, "fr": "", "it": "", "en": ""}
            return {"de": "", "fr": "", "it": "", "en": ""}
        
        def parse_datetime(date_str: str) -> datetime:
            """Parse ISO datetime string."""
            if not date_str:
                return datetime.now()
            try:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except:
                return datetime.now()
        
        # Extract keywords from tags
        keywords = [tag["name"] for tag in raw_data.get("tags", [])]
        
        # Extract themes from groups
        themes = [group["name"] for group in raw_data.get("groups", [])]
        
        # Extract organization
        org = raw_data.get("organization", {})
        org_title = org.get("title", "") if org else ""
        if isinstance(org_title, dict):
            org_title = org_title.get("en") or org_title.get("de") or ""
        
        return DatasetMetadata(
            id=raw_data.get("id", ""),
            title=extract_multilingual(raw_data.get("title")),
            description=extract_multilingual(raw_data.get("notes")),
            keywords=keywords,
            organization=org_title,
            created=parse_datetime(raw_data.get("metadata_created")),
            modified=parse_datetime(raw_data.get("metadata_modified")),
            resources=raw_data.get("resources", []),
            themes=themes,
            temporal_coverage=raw_data.get("temporalCoverage"),
            spatial_coverage=raw_data.get("spatialCoverage"),
            license=raw_data.get("license_id"),
            contact_point=raw_data.get("contact_point"),
            publisher=raw_data.get("publisher")
        )


# Convenience functions for quick access
def get_client() -> CKANClient:
    """Get a configured CKAN client instance."""
    return CKANClient()


def collect_environment_datasets(max_datasets: int = 500) -> List[DatasetMetadata]:
    """Collect datasets from Environment theme."""
    client = get_client()
    raw_datasets = client.collect_all_datasets(
        max_datasets=max_datasets,
        themes=["territory-and-environment"]
    )
    return [client.parse_dataset_metadata(d) for d in raw_datasets]


def collect_mobility_datasets(max_datasets: int = 500) -> List[DatasetMetadata]:
    """Collect datasets from Mobility theme."""
    client = get_client()
    raw_datasets = client.collect_all_datasets(
        max_datasets=max_datasets,
        themes=["mobility-and-transport"]
    )
    return [client.parse_dataset_metadata(d) for d in raw_datasets]


if __name__ == "__main__":
    # Demo: Collect sample datasets
    client = CKANClient()
    
    print("=" * 60)
    print("Swiss Open Government Data - CKAN API Client Demo")
    print("=" * 60)
    
    # Get portal statistics
    result = client.search_datasets(rows=0)
    print(f"\nTotal datasets in portal: {result['count']}")
    
    # Collect sample
    sample = client.search_datasets(rows=10)
    print(f"\nSample of {len(sample['datasets'])} datasets:")
    
    for dataset in sample['datasets']:
        metadata = client.parse_dataset_metadata(dataset)
        title = metadata.title.get('en') or metadata.title.get('de') or 'No title'
        print(f"  - {title[:60]}... (completeness: {metadata.completeness_score:.0%})")
