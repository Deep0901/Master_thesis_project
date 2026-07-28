import hashlib
import json
from pathlib import Path

import pytest

from evaluation.experiment_runner import ExperimentRunner


class StubSystem:
    name = "stub"

    def __init__(self, ranked_docs):
        self.ranked_docs = ranked_docs

    def search(self, query: str, num_results: int = 10):
        return [(doc_id, 1.0 / rank) for rank, doc_id in enumerate(self.ranked_docs, start=1)]


def write_ground_truth(path: Path, relevant_doc: str = "dataset-1") -> None:
    payload = {
        "Q1": {
            "query": {
                "query_id": "Q1",
                "query_text": "test query",
                "query_language": "en",
                "domain": "test",
                "intent": "exercise evaluation plumbing",
                "ground_truth": [],
            },
            "judgments": [
                {
                    "dataset_id": relevant_doc,
                    "dataset_title": "Relevant dataset",
                    "relevance": 2,
                    "annotator": "single_assessor",
                }
            ],
        }
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_queries(path: Path) -> None:
    payload = [
        {
            "query_id": "Q1",
            "query_text": "test query",
            "query_language": "en",
            "domain": "test",
            "intent": "exercise evaluation plumbing",
            "ground_truth": [],
        }
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_run_experiment_rejects_all_zero_metrics_with_positive_judgments(tmp_path):
    ground_truth = tmp_path / "ground_truth_final.json"
    queries = tmp_path / "benchmark_queries.json"
    write_ground_truth(ground_truth)
    write_queries(queries)

    runner = ExperimentRunner(
        ground_truth_file=str(ground_truth),
        benchmark_queries_file=str(queries),
    )
    runner.add_system(StubSystem(["not-relevant"]))

    with pytest.raises(ValueError, match="all-zero metrics"):
        runner.run_experiment()


def test_run_experiment_accepts_nonzero_metrics(tmp_path):
    ground_truth = tmp_path / "ground_truth_final.json"
    queries = tmp_path / "benchmark_queries.json"
    write_ground_truth(ground_truth)
    write_queries(queries)

    runner = ExperimentRunner(
        ground_truth_file=str(ground_truth),
        benchmark_queries_file=str(queries),
    )
    runner.add_system(StubSystem(["dataset-1"]))

    result = runner.run_experiment()

    assert result["stub"]["MAP"] == pytest.approx(1.0)
    assert result["stub"]["P@5"] == pytest.approx(0.2)
    assert result["stub"]["nDCG@10"] == pytest.approx(1.0)
    assert result["stub"]["MRR"] == pytest.approx(1.0)


def test_resolve_ground_truth_does_not_copy_fallback_files(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    evaluation_dir = project_dir / "evaluation"
    evaluation_dir.mkdir(parents=True)
    write_ground_truth(evaluation_dir / "ground_truth_auto.json")

    monkeypatch.chdir(project_dir)
    runner = ExperimentRunner(
        ground_truth_file="evaluation/missing_ground_truth_final.json",
        benchmark_queries_file="evaluation/missing_queries.json",
    )

    with pytest.raises(FileNotFoundError):
        runner._resolve_ground_truth()

    assert not (evaluation_dir / "ground_truth_final.json").exists()


def test_reproducibility_payload_includes_ground_truth_hash(tmp_path):
    ground_truth = tmp_path / "ground_truth_final.json"
    queries = tmp_path / "benchmark_queries.json"
    write_ground_truth(ground_truth)
    write_queries(queries)

    runner = ExperimentRunner(
        ground_truth_file=str(ground_truth),
        benchmark_queries_file=str(queries),
    )

    payload = runner._reproducibility_payload()

    assert payload["ground_truth_hash"] == hashlib.sha256(ground_truth.read_bytes()).hexdigest()
