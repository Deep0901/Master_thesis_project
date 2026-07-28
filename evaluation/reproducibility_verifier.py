"""Lightweight reproducibility verification helpers.

These helpers do not change the main ranking logic. They provide a small
check that the committed artifacts still match the expected hashes when a
reproducibility report is present.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_artifacts(
    corpus_path: Optional[Path | str] = None,
    ground_truth_path: Optional[Path | str] = None,
    report_path: Optional[Path | str] = None,
) -> List[str]:
    """Return a list of reproducibility issues for the provided artifact paths."""
    issues: List[str] = []

    corpus_path = Path(corpus_path or "data/raw/ogd_metadata_20260306_183841.json")
    ground_truth_path = Path(ground_truth_path or "evaluation/ground_truth_final.json")
    report_path = Path(report_path or "evaluation/results/reproducibility_report.json")

    if not corpus_path.exists():
        issues.append(f"Missing corpus snapshot: {corpus_path}")
    if not ground_truth_path.exists():
        issues.append(f"Missing ground truth file: {ground_truth_path}")
    if not report_path.exists():
        issues.append(f"Missing reproducibility report: {report_path}")
        return issues

    if corpus_path.exists() and ground_truth_path.exists():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(f"Invalid JSON in reproducibility report: {exc}")
            return issues

        expected_snapshot_hash = payload.get("snapshot_hash")
        expected_ground_truth_hash = payload.get("ground_truth_hash")

        if expected_snapshot_hash and _sha256(corpus_path) != expected_snapshot_hash:
            issues.append("Corpus snapshot hash does not match the reproducibility report")
        if expected_ground_truth_hash and _sha256(ground_truth_path) != expected_ground_truth_hash:
            issues.append("Ground-truth hash does not match the reproducibility report")

    return issues
