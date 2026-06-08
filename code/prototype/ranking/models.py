from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def safe_html_text(value: Any, fallback: str = "") -> str:
    """Return a safe, non-null text value for HTML rendering."""
    if value is None:
        return fallback
    if isinstance(value, dict):
        value = (
            value.get("en")
            or value.get("de")
            or value.get("fr")
            or value.get("it")
            or next(iter(value.values()), fallback)
        )
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value if item is not None)
    return html.escape(str(value)) if str(value) != "" else fallback


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
            mod_date = datetime.fromisoformat(self.modified.replace("Z", "+00:00"))
            return (datetime.now(mod_date.tzinfo) - mod_date).days
        except Exception:
            return 365

    @property
    def format_list(self) -> List[str]:
        formats: List[str] = []
        for resource in self.resources:
            fmt = str(resource.get("format", "Unknown") or "").upper()
            if fmt and fmt not in formats:
                formats.append(fmt)
        return formats[:5]
