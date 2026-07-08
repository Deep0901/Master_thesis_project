"""Implicit feedback logging for the Streamlit prototype.

Stores per-result positive/negative judgments for later evaluation (e.g., Precision@K, NDCG).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FeedbackEvent:
    timestamp_utc: str
    query: str
    dataset_id: str
    rank: int
    helpful: bool
    ranking_method: str
    data_source: str
    relevance_score: float
    metadata_scores: Dict[str, float]
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "query": self.query,
            "dataset_id": self.dataset_id,
            "rank": self.rank,
            "helpful": self.helpful,
            "ranking_method": self.ranking_method,
            "data_source": self.data_source,
            "relevance_score": self.relevance_score,
            "metadata_scores": self.metadata_scores,
            "extra": self.extra,
        }


def default_feedback_path() -> Path:
    # Keep results under evaluation/ for easy inclusion in thesis artifacts.
    return Path(__file__).resolve().parents[2] / "evaluation" / "results" / "implicit_feedback.jsonl"


def default_feedback_csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "evaluation" / "results" / "implicit_feedback.csv"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _feedback_csv_headers() -> List[str]:
    return [
        "timestamp_utc",
        "query",
        "dataset_id",
        "rank",
        "helpful",
        "ranking_method",
        "data_source",
        "relevance_score",
        "recency_score",
        "completeness_score",
        "resource_score",
        "similarity_score",
        "license_id",
        "organization",
    ]


def _feedback_event_to_csv_row(event: FeedbackEvent) -> Dict[str, Any]:
    return {
        "timestamp_utc": event.timestamp_utc,
        "query": event.query,
        "dataset_id": event.dataset_id,
        "rank": event.rank,
        "helpful": event.helpful,
        "ranking_method": event.ranking_method,
        "data_source": event.data_source,
        "relevance_score": event.relevance_score,
        "recency_score": float(event.metadata_scores.get("recency", 0.0)),
        "completeness_score": float(event.metadata_scores.get("completeness", 0.0)),
        "resource_score": float(event.metadata_scores.get("resources", 0.0)),
        "similarity_score": float(event.metadata_scores.get("similarity", 0.0)),
        "license_id": event.extra.get("license_id", ""),
        "organization": event.extra.get("organization", ""),
    }


def append_feedback_csv(event: FeedbackEvent, path: Optional[Path] = None) -> Path:
    target = path or default_feedback_csv_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists()

    with target.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_feedback_csv_headers())
        if write_header:
            writer.writeheader()
        writer.writerow(_feedback_event_to_csv_row(event))

    return target


def append_feedback_event(event: FeedbackEvent, path: Optional[Path] = None) -> Path:
    """Append an event as JSONL and CSV, then return the JSONL path used."""
    target = path or default_feedback_path()
    csv_target = default_feedback_csv_path() if path is None else path.with_suffix(".csv")
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    append_feedback_csv(event, csv_target)
    return target


def build_event(
    *,
    query: str,
    dataset_id: str,
    rank: int,
    helpful: bool,
    ranking_method: str,
    data_source: str,
    relevance_score: float,
    metadata_scores: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> FeedbackEvent:
    return FeedbackEvent(
        timestamp_utc=utc_now_iso(),
        query=query,
        dataset_id=dataset_id,
        rank=rank,
        helpful=helpful,
        ranking_method=ranking_method,
        data_source=data_source,
        relevance_score=float(relevance_score),
        metadata_scores=dict(metadata_scores or {}),
        extra=dict(extra or {}),
    )
