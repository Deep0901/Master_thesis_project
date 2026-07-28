import hashlib
import json
from pathlib import Path

from evaluation.reproducibility_verifier import verify_artifacts


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_verify_artifacts_accepts_matching_hashes(tmp_path):
    corpus = tmp_path / "corpus.json"
    ground_truth = tmp_path / "ground_truth.json"
    report = tmp_path / "reproducibility_report.json"

    write_text(corpus, '{"datasets": [1]}')
    write_text(ground_truth, '{"queries": []}')
    report.write_text(
        json.dumps(
            {
                "snapshot_hash": sha256(corpus),
                "ground_truth_hash": sha256(ground_truth),
            }
        ),
        encoding="utf-8",
    )

    issues = verify_artifacts(
        corpus_path=corpus,
        ground_truth_path=ground_truth,
        report_path=report,
    )

    assert issues == []
