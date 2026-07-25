from pathlib import Path

from evaluation.scripts.export_user_study_audit import build_audit_artifact


def test_build_audit_artifact_writes_instrument_and_status(tmp_path):
    output_path = tmp_path / "user_study_audit.md"

    payload = build_audit_artifact(output_path=output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert "Evaluation Questionnaire Audit" in rendered
    assert "The explanation messages helped me understand" in rendered
    assert "No de-identified item-level response matrix is present" in rendered
    assert payload["item_count"] == 10
