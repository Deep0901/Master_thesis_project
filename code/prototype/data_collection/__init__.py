"""
Data Collection Package

Modules for collecting metadata from opendata.swiss CKAN API.
"""

from .ckan_api_client import (
    CKANClient,
    DatasetMetadata,
    get_client,
    collect_environment_datasets,
    collect_mobility_datasets
)

__all__ = [
    'CKANClient',
    'DatasetMetadata',
    'get_client',
    'collect_environment_datasets',
    'collect_mobility_datasets'
]
