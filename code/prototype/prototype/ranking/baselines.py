from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List

from .models import DatasetResult, normalize_text


def dataset_identity_key(dataset: Dict) -> str:
    """Build a stable identity key used to remove duplicate datasets."""
    name = str(dataset.get("name") or "").strip().lower()
    if name:
        return f"name:{name}"

    dataset_id = str(dataset.get("id") or "").strip().lower()
    if dataset_id:
        return f"id:{dataset_id}"

    title = dataset.get("title", {})
    title_text = normalize_text(title, "").strip().lower()

    org = dataset.get("organization", {}) or {}
    org_name = normalize_text(org, "").strip().lower()

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
    title = dataset.get("title", {})
    title_text = normalize_text(title, "").strip().lower()

    org = dataset.get("organization", {}) or {}
    org_text = normalize_text(org, "").strip().lower()

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
    # result.title may be a dict of localized strings or a plain string
    title_text = normalize_text(result.title, "").strip().lower()
    org_text = normalize_text(result.organization, "").strip().lower()
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
    """Simulates the portal's default ranking (API order)."""

    def rank(self, datasets: List[Dict], query: str) -> List[Dict]:
        datasets = deduplicate_datasets(datasets)
        for index, dataset in enumerate(datasets):
            dataset["_relevance_score"] = 1.0 - (index * 0.02)
            dataset["_ranking_method"] = "portal_default"
        return datasets


class BM25Ranker:
    """BM25 probabilistic ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _get_doc_text(self, dataset: Dict) -> str:
        parts: List[str] = []

        title = dataset.get("title", {})
        if isinstance(title, dict):
            parts.extend(str(v) for v in title.values() if v)
        elif title:
            parts.append(str(title))

        desc = dataset.get("description", {})
        if isinstance(desc, dict):
            parts.extend(str(v) for v in desc.values() if v)
        elif desc:
            parts.append(str(desc))

        tags = dataset.get("tags", [])
        for tag in tags:
            if isinstance(tag, dict):
                parts.append(str(tag.get("name", "")))
            else:
                parts.append(str(tag))

        return " ".join(parts)

    def rank(self, datasets: List[Dict], query: str) -> List[Dict]:
        datasets = deduplicate_datasets(datasets)
        query_terms = self._tokenize(query)

        if not query_terms or not datasets:
            return datasets

        docs = [self._tokenize(self._get_doc_text(ds)) for ds in datasets]
        avg_dl = sum(len(d) for d in docs) / len(docs) if docs else 1

        N = len(docs)
        idf: Dict[str, float] = {}
        for term in query_terms:
            df = sum(1 for doc in docs if term in doc)
            idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1) if df > 0 else 0

        scores: List[float] = []
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

        max_score = max(scores) if scores and max(scores) > 0 else 1

        for i, dataset in enumerate(datasets):
            dataset["_relevance_score"] = scores[i] / max_score if max_score > 0 else 0
            dataset["_ranking_method"] = "bm25"

        return sorted(datasets, key=lambda x: x.get("_relevance_score", 0), reverse=True)
