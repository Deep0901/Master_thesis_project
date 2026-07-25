from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


QUESTIONNAIRE_ITEMS = [
    {
        "id": "Q1",
        "text": "The explanation messages helped me understand why a dataset was ranked highly.",
        "scale": "1 = strongly disagree to 5 = strongly agree",
    },
    {
        "id": "Q2",
        "text": "The ranking results were easy to interpret.",
        "scale": "1 = strongly disagree to 5 = strongly agree",
    },
    {
        "id": "Q3",
        "text": "The metadata summaries were useful for judging dataset relevance.",
        "scale": "1 = strongly disagree to 5 = strongly agree",
    },
    {
        "id": "Q4",
        "text": "I trusted the system's ranking decisions.",
        "scale": "1 = strongly disagree to 5 = strongly agree",
    },
    {
        "id": "Q5",
        "text": "I would use this system to find Swiss open government data.",
        "scale": "1 = strongly disagree to 5 = strongly agree",
    },
    {
        "id": "Q6",
        "text": "The interface made it easy to compare candidate datasets.",
        "scale": "1 = strongly disagree to 5 = strongly agree",
    },
    {
        "id": "Q7",
        "text": "The explanations increased my confidence in the results.",
        "scale": "1 = strongly disagree to 5 = strongly agree",
    },
    {
        "id": "Q8",
        "text": "The system helped me understand the role of metadata in ranking.",
        "scale": "1 = strongly disagree to 5 = strongly agree",
    },
    {
        "id": "Q9",
        "text": "The system was useful for completing the retrieval task.",
        "scale": "1 = strongly disagree to 5 = strongly agree",
    },
    {
        "id": "Q10",
        "text": "Please add any additional comments about the experience.",
        "scale": "Open-ended response",
    },
]


def build_audit_artifact(output_path: Path | str | None = None) -> Dict[str, Any]:
    target = Path(output_path) if output_path is not None else Path("evaluation/results/user_study_audit.md")
    target.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Evaluation Questionnaire Audit",
        "",
        "This artifact records the user-study instrument that was actually used for the thesis evaluation.",
        "",
        "## Instrument",
        "",
        "- Participants: 10",
        "- Format: formative questionnaire completed after the retrieval task",
        "- Response scale: 1 = strongly disagree to 5 = strongly agree for closed-ended items",
        "- Open-ended prompt: included for comments",
        "",
        "## Questionnaire items",
        "",
    ]

    for item in QUESTIONNAIRE_ITEMS:
        lines.append(f"- {item['id']}: {item['text']}")
        lines.append(f"  Scale: {item['scale']}")

    lines.extend([
        "",
        "## Audit status",
        "",
        "- No de-identified item-level response matrix is present in the repository snapshot inspected for this revision.",
        "- The questionnaire instrument is therefore recorded here as the authoritative audit artifact for the study design.",
        "- Any future publication of participant-level aggregates should be added alongside this file with the consent and provenance details.",
        "",
    ])

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "path": str(target),
        "item_count": len(QUESTIONNAIRE_ITEMS),
        "participants": 10,
        "status": "questionnaire_instrument_recorded",
    }


if __name__ == "__main__":
    result = build_audit_artifact()
    print(json.dumps(result, indent=2))
