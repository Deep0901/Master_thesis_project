"""
Build Authoritative Ground Truth JSON from pooled_candidates.csv

This script reads evaluation/data/pooled_candidates.csv and constructs the canonical,
authoritative evaluation/ground_truth_final.json file using the single-assessor
consolidated grades collapsed to the published 0-2 scale.
"""

import csv
import json
from pathlib import Path

POOLED_CSV = Path("evaluation/data/pooled_candidates.csv")
BENCHMARK_QUERIES_JSON = Path("evaluation/benchmark_queries.json")
OUTPUT_JSON = Path("evaluation/ground_truth_final.json")


def build_ground_truth(
    pooled_csv: Path | str | None = None,
    benchmark_queries_json: Path | str | None = None,
    output_json: Path | str | None = None,
):
    pooled_csv_path = Path(pooled_csv) if pooled_csv is not None else POOLED_CSV
    benchmark_queries_path = Path(benchmark_queries_json) if benchmark_queries_json is not None else BENCHMARK_QUERIES_JSON
    output_path = Path(output_json) if output_json is not None else OUTPUT_JSON

    if not pooled_csv_path.exists():
        raise FileNotFoundError(f"Missing pooled candidates file: {pooled_csv_path}")
    
    queries_metadata = {}
    if benchmark_queries_path.exists():
        with open(benchmark_queries_path, "r", encoding="utf-8") as f:
            q_data = json.load(f)
            if isinstance(q_data, list):
                for q in q_data:
                    queries_metadata[q["query_id"]] = q
            elif isinstance(q_data, dict):
                queries_metadata = q_data

    ground_truth = {}

    with open(pooled_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"]
            if qid not in ground_truth:
                q_meta = queries_metadata.get(qid, {
                    "query_id": qid,
                    "query_text": row["query_text"],
                    "query_language": "de",
                    "domain": "general",
                    "intent": row["query_text"]
                })
                if isinstance(q_meta, dict) and "query" in q_meta:
                    q_meta = q_meta["query"]
                ground_truth[qid] = {
                    "query": {
                        **q_meta,
                        "grading_scale": {
                            "authoritative_scale": "0-2",
                            "mapping": {
                                "3": "2",
                                "2": "2",
                                "1": "1",
                                "0": "0",
                            },
                            "note": "The published final ground truth uses the collapsed 0-2 scale derived from the single-author consolidation process.",
                        },
                    },
                    "judgments": []
                }
            
            # Support multiple CSV fieldname variants: prefer consolidated single-assessor fields
            grade_str = (
                row.get("author_consolidated_grade")
                or row.get("single_assessor_grade")
                or row.get("adjudicated_grade")
                or row.get("judge1_grade")
                or row.get("judge_grade")
                or "0"
            )
            try:
                grade = int(grade_str)
            except ValueError:
                grade = 0
            
            # Ensure grade is in [0, 2]
            grade = max(0, min(2, grade))
            
            judgment = {
                "dataset_id": row["dataset_id"],
                "dataset_title": row.get("dataset_title", ""),
                "relevance": grade,
                "annotator": "single_assessor_consolidated",
                "notes": row.get("notes", "")
            }
            ground_truth[qid]["judgments"].append(judgment)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {output_path} with {len(ground_truth)} queries and {sum(len(v['judgments']) for v in ground_truth.values())} total judgments.")

    # Basic sanity check: ensure at least one non-zero relevance judgment exists.
    total_judgments = sum(len(v['judgments']) for v in ground_truth.values())
    non_zero = sum(1 for v in ground_truth.values() for j in v['judgments'] if j.get('relevance', 0) > 0)

    if non_zero == 0:
        # Preserve the current authoritative file if no positive grades are present in the CSV.
        # This prevents the builder from silently overwriting the existing final ground truth
        # with an empty fallback during incomplete annotation runs.
        if output_path.exists():
            print(f"No positive grades were found in {pooled_csv_path}; preserving existing output at {output_path}")
            return

        raise ValueError(
            f"Constructed {output_path} contains {total_judgments} judgments but none are rated > 0. "
            "Do not publish an all-zero authoritative ground-truth file; ensure pooled candidates "
            "contain assessor grades before generating the final file."
        )

if __name__ == "__main__":
    build_ground_truth()
