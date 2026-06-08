"""Dynamic calibration of fuzzy membership functions using live CKAN distributions.

This script queries the opendata.swiss CKAN API, computes robust percentiles for
key metadata factors, and writes membership-function boundaries to a JSON file
consumed by the Streamlit prototype (`CalibratedFuzzyEngine`).

Typical usage (one-shot):
    c:/thesis/.venv/Scripts/python.exe analytics/dynamic_calibration.py --sample 2000

Periodic usage (e.g., every 24h):
    c:/thesis/.venv/Scripts/python.exe analytics/dynamic_calibration.py --sample 2000 --interval-minutes 1440

Output:
    analytics/fuzzy_calibration_live.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# When executed as a script, ensure repo root is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.prototype.api.client import OpenDataSwissClient


@dataclass
class CalibrationSummary:
    timestamp_utc: str
    sample_size: int
    portal_count_estimate: int
    sampling: str
    page_size: int


def _safe_days_since_modified(dataset: Dict[str, Any]) -> int:
    mod_str = dataset.get("metadata_modified") or ""
    if not mod_str:
        return 365
    try:
        mod_dt = datetime.fromisoformat(mod_str.replace("Z", "+00:00"))
        now = datetime.now(mod_dt.tzinfo)
        return max(0, int((now - mod_dt).days))
    except Exception:
        return 365


def _resource_count(dataset: Dict[str, Any]) -> int:
    resources = dataset.get("resources") or []
    return int(len(resources))


def _completeness_ratio(dataset: Dict[str, Any]) -> float:
    # Lightweight completeness proxy aligned with the prototype's MetadataAnalyzer.
    # (Keeps calibration script independent of Streamlit UI code.)
    weights = {
        "title": 1.0,
        "description": 1.0,
        "organization": 0.8,
        "resources": 0.9,
        "tags": 0.6,
        "groups": 0.7,
        "license": 0.5,
        "metadata_modified": 0.4,
    }

    achieved = 0.0
    total = float(sum(weights.values()))

    for field, w in weights.items():
        value = dataset.get(field)

        if field == "title":
            if isinstance(value, dict):
                achieved += w if any(value.values()) else 0.0
            elif value:
                achieved += w
            continue

        if field == "description":
            if isinstance(value, dict):
                desc = " ".join(str(v) for v in value.values() if v)
            else:
                desc = str(value) if value else ""
            if len(desc) >= 50:
                achieved += w
            elif len(desc) >= 20:
                achieved += w * 0.5
            continue

        if field == "resources":
            resources = value or []
            if len(resources) >= 3:
                achieved += w
            elif len(resources) >= 1:
                achieved += w * 0.7
            continue

        if field == "tags":
            tags = value or []
            if len(tags) >= 3:
                achieved += w
            elif len(tags) >= 1:
                achieved += w * 0.5
            continue

        if field == "groups":
            achieved += w if value else 0.0
            continue

        if field == "organization":
            org = value or {}
            if isinstance(org, dict) and (org.get("name") or org.get("title")):
                achieved += w
            elif org:
                achieved += w
            continue

        achieved += w if value else 0.0

    return float(achieved / total) if total > 0 else 0.0


def _percentiles(values: List[float], ps: List[int]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    out: Dict[str, float] = {}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out


def _non_decreasing(params: List[float]) -> List[float]:
    """Ensure MF params are non-decreasing to avoid invalid shapes."""
    out: List[float] = []
    last = -float("inf")
    for x in params:
        fx = float(x)
        if fx < last:
            fx = last
        out.append(fx)
        last = fx
    return out


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _collect_sample(
    client: OpenDataSwissClient,
    *,
    sample_size: int,
    page_size: int,
    sampling: str,
    seed: Optional[int],
    sort: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Collect a sample of datasets using one or more CKAN pages."""
    # 1) Get portal count estimate.
    _, portal_count = client.search("*:*", rows=1, start=0, sort=sort)
    portal_count = int(portal_count or 0)
    if portal_count <= 0:
        return ([], 0)

    page_size = max(10, min(int(page_size), 200))
    target = max(10, min(int(sample_size), 5000))

    max_start = max(0, portal_count - page_size)
    if sampling not in {"recent", "random_pages"}:
        sampling = "random_pages"

    if sampling == "recent":
        starts = list(range(0, min(max_start, target), page_size))
        if not starts:
            starts = [0]
    else:
        rng = random.Random(seed)
        # Choose page-aligned starts to cover the distribution.
        possible = list(range(0, max_start + 1, page_size))
        rng.shuffle(possible)
        pages_needed = int(math.ceil(target / page_size))
        starts = possible[: max(1, pages_needed)]

    datasets: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for start in starts:
        rows = min(page_size, target - len(datasets))
        if rows <= 0:
            break

        chunk, _ = client.search("*:*", rows=rows, start=int(start), sort=sort)
        for ds in chunk or []:
            ds_id = str(ds.get("id") or ds.get("name") or "")
            if not ds_id or ds_id in seen_ids:
                continue
            seen_ids.add(ds_id)
            datasets.append(ds)
            if len(datasets) >= target:
                break

        if len(datasets) >= target:
            break

    return (datasets, portal_count)


def _build_membership_functions(
    *,
    days: List[int],
    completeness: List[float],
    resources: List[int],
) -> Dict[str, Dict[str, List[float]]]:
    """Generate MF boundaries from percentiles with basic hardening."""
    day_p = {
        10: float(np.percentile(days, 10)),
        25: float(np.percentile(days, 25)),
        50: float(np.percentile(days, 50)),
        75: float(np.percentile(days, 75)),
        90: float(np.percentile(days, 90)),
    }
    day_min, day_max = float(min(days)), float(max(days))

    comp_p = {
        10: float(np.percentile(completeness, 10)),
        25: float(np.percentile(completeness, 25)),
        50: float(np.percentile(completeness, 50)),
        75: float(np.percentile(completeness, 75)),
        90: float(np.percentile(completeness, 90)),
    }

    res_p = {
        25: float(np.percentile(resources, 25)),
        50: float(np.percentile(resources, 50)),
        75: float(np.percentile(resources, 75)),
        90: float(np.percentile(resources, 90)),
        95: float(np.percentile(resources, 95)),
    }
    res_min, res_max = float(min(resources)), float(max(resources))

    # Recency: lower is better (days since modified).
    recency = {
        "very_recent": _non_decreasing([day_min, day_min, day_p[10]]),
        "recent": _non_decreasing([day_min, day_p[10], day_p[25], day_p[50]]),
        "moderate": _non_decreasing([day_p[25], day_p[50], day_p[75], day_p[90]]),
        "old": _non_decreasing([day_p[50], day_p[75], day_p[90], day_max]),
        "very_old": _non_decreasing([day_p[90], day_max, day_max]),
    }

    # Completeness: 0..1, higher is better.
    c10, c25, c50, c75, c90 = (comp_p[10], comp_p[25], comp_p[50], comp_p[75], comp_p[90])
    completeness_mf = {
        "low": _non_decreasing([0.0, 0.0, _clamp(c10, 0.0, 1.0)]),
        "partial": _non_decreasing([
            _clamp(c10, 0.0, 1.0),
            _clamp(c25, 0.0, 1.0),
            _clamp(c50, 0.0, 1.0),
        ]),
        "medium": _non_decreasing([
            _clamp(c25, 0.0, 1.0),
            _clamp(c50, 0.0, 1.0),
            _clamp(c75, 0.0, 1.0),
            _clamp(c90, 0.0, 1.0),
        ]),
        "high": _non_decreasing([
            _clamp(c50, 0.0, 1.0),
            _clamp(c75, 0.0, 1.0),
            _clamp(c90, 0.0, 1.0),
            1.0,
        ]),
        "complete": _non_decreasing([
            _clamp(c90, 0.0, 1.0),
            1.0,
            1.0,
        ]),
    }

    # Resources: non-negative integer-ish.
    resources_mf = {
        "minimal": _non_decreasing([res_min, res_min, res_p[25]]),
        "limited": _non_decreasing([res_min, res_p[25], res_p[50]]),
        "moderate": _non_decreasing([res_p[25], res_p[50], res_p[75]]),
        "rich": _non_decreasing([res_p[50], res_p[75], res_p[90], res_p[95]]),
        "comprehensive": _non_decreasing([res_p[90], res_p[95], res_max]),
    }

    return {
        "recency": recency,
        "completeness": completeness_mf,
        "resources": resources_mf,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=2000, help="Number of datasets to sample")
    parser.add_argument(
        "--sampling",
        type=str,
        default="random_pages",
        choices=["random_pages", "recent"],
        help="Sampling strategy: random pages across the portal, or most-recent datasets only",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Datasets per API call when sampling across pages",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible random sampling",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="metadata_modified desc",
        help="CKAN sort order for package_search",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("analytics") / "fuzzy_calibration_live.json"),
        help="Output calibration JSON path",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=0,
        help="If >0, run calibration repeatedly every N minutes",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Stop after N runs in periodic mode (0 = run forever)",
    )
    parser.add_argument(
        "--jitter-seconds",
        type=int,
        default=0,
        help="Optional random jitter added before each periodic run",
    )
    args = parser.parse_args()

    client = OpenDataSwissClient()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def run_once() -> None:
        datasets, portal_count = _collect_sample(
            client,
            sample_size=int(args.sample),
            page_size=int(args.page_size),
            sampling=str(args.sampling),
            seed=args.seed,
            sort=str(args.sort),
        )

        if not datasets:
            raise RuntimeError("No datasets returned from CKAN; cannot calibrate.")

        days = [_safe_days_since_modified(ds) for ds in datasets]
        completeness = [_completeness_ratio(ds) for ds in datasets]
        resources = [_resource_count(ds) for ds in datasets]

        calibration = {
            "summary": asdict(
                CalibrationSummary(
                    timestamp_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    sample_size=len(datasets),
                    portal_count_estimate=int(portal_count or 0),
                    sampling=str(args.sampling),
                    page_size=int(args.page_size),
                )
            ),
            "recency_days": {
                "percentiles": _percentiles(days, [10, 25, 50, 75, 90]),
                "min": int(min(days)),
                "max": int(max(days)),
            },
            "completeness_ratio": {
                "percentiles": _percentiles(completeness, [10, 25, 50, 75, 90]),
                "min": float(min(completeness)),
                "max": float(max(completeness)),
            },
            "resource_count": {
                "percentiles": _percentiles(resources, [25, 50, 75, 90, 95]),
                "min": int(min(resources)),
                "max": int(max(resources)),
            },
            "membership_functions": _build_membership_functions(
                days=days,
                completeness=completeness,
                resources=resources,
            ),
        }

        out_path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote calibration to: {out_path}")

    interval_minutes = int(args.interval_minutes or 0)
    if interval_minutes <= 0:
        run_once()
        return 0

    runs = 0
    max_runs = int(args.max_runs or 0)
    rng = random.Random(args.seed)

    while True:
        if int(args.jitter_seconds or 0) > 0:
            time.sleep(rng.uniform(0, float(args.jitter_seconds)))

        try:
            run_once()
        except Exception as e:
            # Periodic jobs should be resilient: keep going, but emit a clear error.
            print(f"Calibration run failed: {e}")

        runs += 1
        if max_runs > 0 and runs >= max_runs:
            return 0

        time.sleep(float(interval_minutes) * 60.0)


if __name__ == "__main__":
    raise SystemExit(main())
