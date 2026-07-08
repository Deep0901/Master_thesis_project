from __future__ import annotations

import ast
import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def normalize_text(value: Any, fallback: str = "") -> str:
    """Normalize metadata values into clean plain text."""
    if value is None:
        return fallback

    if isinstance(value, dict):
        for lang in ("en", "de", "fr", "it"):
            candidate = value.get(lang)
            if candidate:
                normalized = normalize_text(candidate, fallback)
                if normalized:
                    return normalized
        for candidate in value.values():
            normalized = normalize_text(candidate, fallback)
            if normalized:
                return normalized
        return fallback

    if isinstance(value, (list, tuple)):
        normalized_items = [normalize_text(item, fallback) for item in value if item is not None]
        joined = " ".join(item for item in normalized_items if item)
        return joined.strip() or fallback

    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return fallback

        if (trimmed.startswith("{") and trimmed.endswith("}")) or (
            trimmed.startswith("[") and trimmed.endswith("]")
        ):
            try:
                parsed = json.loads(trimmed)
                return normalize_text(parsed, fallback)
            except Exception:
                try:
                    parsed = ast.literal_eval(trimmed)
                    return normalize_text(parsed, fallback)
                except Exception:
                    pass

        unescaped = html.unescape(trimmed)
        cleaned = re.sub(r"<[^>]+>", "", unescaped)
        return cleaned.strip() or fallback

    return str(value).strip() or fallback


def safe_html_text(value: Any, fallback: str = "") -> str:
    """Return a safe, escaped text value for HTML rendering."""
    text = normalize_text(value, fallback)
    return html.escape(text) if text != "" else fallback


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
    title: Any
    description: str
    organization: Any
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
