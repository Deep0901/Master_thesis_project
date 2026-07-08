from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import requests


ErrorHandler = Callable[[str], None]


@dataclass
class CKANSearchResponse:
    results: List[Dict]
    count: int


class OpenDataSwissClient:
    """Client for the opendata.swiss CKAN API.

    This module is intentionally Streamlit-free to keep the client easy to unit test.
    Callers can pass an optional `error_handler` (e.g., `st.error`) to surface errors.
    """

    BASE_URL = "https://opendata.swiss/api/3/action"

    def __init__(self, *, error_handler: Optional[ErrorHandler] = None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Swiss-OGD-Fuzzy-HCIR-Research/1.0"
        })
        self._last_request_time = 0.0
        self._min_request_interval = 0.2  # 200ms between requests
        self._error_handler = error_handler

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _on_error(self, message: str) -> None:
        if self._error_handler is not None:
            try:
                self._error_handler(message)
            except Exception:
                pass

    def search(
        self,
        query: str,
        *,
        rows: int = 30,
        start: int = 0,
        fq: Optional[str] = None,
        sort: str = "score desc",
    ) -> Tuple[List[Dict], int]:
        """Search datasets on opendata.swiss."""
        self._rate_limit()

        params: Dict[str, object] = {
            "q": query,
            "rows": rows,
            "start": start,
            "sort": sort,
        }
        if fq:
            params["fq"] = fq

        try:
            response = self.session.get(
                f"{self.BASE_URL}/package_search",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("success"):
                result = data.get("result") or {}
                return (result.get("results") or []), int(result.get("count") or 0)

            return [], 0

        except requests.RequestException as e:
            self._on_error(f"API Error: {str(e)}")
            return [], 0

    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Get a single dataset by ID."""
        self._rate_limit()

        try:
            response = self.session.get(
                f"{self.BASE_URL}/package_show",
                params={"id": dataset_id},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                return data.get("result")
            return None
        except requests.RequestException:
            return None

    def get_organizations(self) -> List[Dict]:
        """Get all organizations."""
        self._rate_limit()

        try:
            response = self.session.get(
                f"{self.BASE_URL}/organization_list",
                params={"all_fields": True},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                return data.get("result") or []
            return []
        except requests.RequestException:
            return []

    def get_themes(self) -> List[Dict]:
        """Get all thematic categories (groups)."""
        self._rate_limit()

        try:
            response = self.session.get(
                f"{self.BASE_URL}/group_list",
                params={"all_fields": True},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                return data.get("result") or []
            return []
        except requests.RequestException:
            return []
