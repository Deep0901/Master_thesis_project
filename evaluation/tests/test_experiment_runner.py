from evaluation.experiment_runner import ExperimentRunner


class StubSystem:
    name = "stub"

    def search(self, query: str, num_results: int = 10):
        return [("dataset-1", 0.9)]


def test_run_complete_pipeline_writes_expected_outputs(tmp_path):
    runner = ExperimentRunner(
        ground_truth_file="evaluation/ground_truth_auto.json",
        benchmark_queries_file="evaluation/benchmark_queries_v2.json",
    )
    runner.add_system(StubSystem())
    runner.output_dir = tmp_path / "results"

    result = runner.run_complete_pipeline()

    assert result["evaluation_results"]
    assert (tmp_path / "results" / "query_metrics.csv").exists()
    assert (tmp_path / "results" / "system_summary.csv").exists()
    assert (tmp_path / "results" / "pairwise_statistics.csv").exists()
    assert (tmp_path / "results" / "bootstrap_confidence_intervals.csv").exists()
    assert (tmp_path / "results" / "win_loss_matrix.csv").exists()
    assert (tmp_path / "results" / "publication_tables.md").exists()
    assert (tmp_path / "results" / "figures" / "bar_mean_metrics.png").exists()
