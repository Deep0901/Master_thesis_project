import csv
import json
from pathlib import Path

from evaluation.build_ground_truth import build_ground_truth


def test_build_ground_truth_adds_provenance_metadata(tmp_path):
    pooled_csv = tmp_path / "pooled_candidates.csv"
    benchmark_queries = tmp_path / "benchmark_queries.json"
    output_json = tmp_path / "ground_truth_final.json"

    with pooled_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "query_text", "dataset_id", "dataset_title", "single_assessor_grade", "notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "Q1",
                "query_text": "housing statistics",
                "dataset_id": "ds-1",
                "dataset_title": "Housing data",
                "single_assessor_grade": "3",
                "notes": "originally graded 3",
            }
        )

    benchmark_queries.write_text(
        json.dumps([{"query_id": "Q1", "query_text": "housing statistics", "query_language": "en", "domain": "social", "intent": "find housing data"}]),
        encoding="utf-8",
    )

    build_ground_truth(
        pooled_csv=pooled_csv,
        benchmark_queries_json=benchmark_queries,
        output_json=output_json,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    query_meta = payload["Q1"]["query"]
    assert query_meta["grading_scale"]["authoritative_scale"] == "0-2"
    assert query_meta["grading_scale"]["mapping"]["3"] == "2"
    assert payload["Q1"]["judgments"][0]["relevance"] == 2
    assert payload["Q1"]["judgments"][0]["annotator"] == "single_assessor_consolidated"
