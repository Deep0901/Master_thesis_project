"""
Ground Truth Annotation Tool for Relevance Judgments

This tool helps collect ground truth relevance judgments for evaluation queries.
It searches the actual opendata.swiss portal and presents results for annotation.

Process:
1. Load benchmark queries
2. For each query, search opendata.swiss
3. Present top results for manual relevance annotation
4. Save annotations to JSON for evaluation

Annotation Scale (NIST TREC style):
0 - Not relevant (does not address information need)
1 - Marginally relevant (tangentially related)  
2 - Relevant (addresses need but may not be ideal)
3 - Highly relevant (directly addresses information need)

Author: Deep Shukla
Thesis: Improving Access to Swiss OGD through Fuzzy HCIR
University of Fribourg, Human-IST Institute
"""

import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

from evaluation.evaluation_framework import (
    compute_quadratic_weighted_kappa,
    compute_percentage_agreement,
    compute_disagreement_count,
    landis_koch_category,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetCandidate:
    """Dataset retrieved for annotation."""
    dataset_id: str
    name: str
    title: str
    description: str
    organization: str
    themes: List[str]
    num_resources: int
    formats: List[str]
    metadata_modified: str
    
    # Annotation (to be filled)
    relevance: Optional[int] = None
    notes: str = ""


class AnnotationTool:
    """
    Interactive tool for collecting relevance judgments.
    """

    def __init__(
        self,
        pooled_candidates_file: str = "evaluation/data/pooled_candidates.csv",
    ):
        self.pooled_candidates_file = Path(pooled_candidates_file)
        self.queries: List[Dict[str, Any]] = []
        self.rows: List[Dict[str, str]] = []
        self.annotations: Dict[str, Dict[str, Any]] = {}
        self._load_pooled_candidates()

    def _load_pooled_candidates(self) -> List[Dict[str, str]]:
        """Load the pooled candidate CSV produced by the experiment runner."""
        self.rows = []
        if not self.pooled_candidates_file.exists():
            logger.warning("Pooled candidates file not found: %s", self.pooled_candidates_file)
            return self.rows

        with open(self.pooled_candidates_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
        return self.rows

    def _save_rows(self) -> None:
        """Persist the pooled candidate CSV after each incremental update."""
        if not self.rows:
            return

        fieldnames = list(self.rows[0].keys())
        with open(self.pooled_candidates_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def _group_rows_by_query(self) -> Dict[str, List[Dict[str, str]]]:
        """Group pooled candidates by query identifier."""
        grouped: Dict[str, List[Dict[str, str]]] = {}
        for row in self.rows:
            grouped.setdefault(row.get("query_id", ""), []).append(row)
        return grouped

    def _prompt_grade(self, prompt_text: str, allow_blank: bool = False) -> str:
        """Prompt the annotator for a grade or note string."""
        while True:
            value = input(prompt_text).strip()
            if allow_blank and value == "":
                return ""
            if value in {"0", "1", "2"}:
                return value
            print("  Invalid input. Enter 0, 1, 2, or leave blank.")

    def annotate_pooled_candidates_interactive(self) -> None:
        """Annotate pooled candidates without performing any portal search."""
        if not self.rows:
            print("No pooled candidates loaded.")
            return

        grouped_rows = self._group_rows_by_query()

        for query_id, rows in grouped_rows.items():
            print("\n" + "=" * 70)
            print(f"QUERY: {query_id}")
            print("=" * 70)
            print(f"Text: {rows[0].get('query_text', '')}")

            for index, row in enumerate(rows, 1):
                print(f"\n--- Candidate {index}/{len(rows)} ---")
                print(f"Dataset title: {row.get('dataset_title', '')}")
                print(f"Dataset id: {row.get('dataset_id', '')}")
                print(f"Systems: {row.get('systems_found_in', '')}")
                print(f"Judge1: {row.get('judge1_grade', '')}")
                print(f"Judge2: {row.get('judge2_grade', '')}")
                print(f"Adjudicated: {row.get('adjudicated_grade', '')}")
                print(f"Notes: {row.get('notes', '')}")

                row["judge1_grade"] = self._prompt_grade("Judge 1 (0/1/2, blank to keep): ", allow_blank=True) or row.get("judge1_grade", "")
                row["judge2_grade"] = self._prompt_grade("Judge 2 (0/1/2, blank to keep): ", allow_blank=True) or row.get("judge2_grade", "")
                row["adjudicated_grade"] = self._prompt_grade("Adjudicated (0/1/2, blank to keep): ", allow_blank=True) or row.get("adjudicated_grade", "")
                row["notes"] = input("Notes (optional): ").strip() or row.get("notes", "")

                self._save_rows()

        print(f"\nSaved annotations to {self.pooled_candidates_file}")

    def export_final_ground_truth(
        self,
        output_file: str = "evaluation/ground_truth_final.json",
    ) -> Dict[str, Any]:
        """Export human annotations from pooled candidates to the final ground truth file."""
        grouped: Dict[str, Dict[str, Any]] = {}

        for row in self.rows:
            query_id = row.get("query_id", "").strip()
            query_text = row.get("query_text", "").strip()
            dataset_id = row.get("dataset_id", "").strip()
            if not query_id or not dataset_id:
                continue

            adjudicated = str(row.get("adjudicated_grade", "")).strip()
            if adjudicated != "":
                relevance = int(float(adjudicated))
                annotator = "adjudicated"
            else:
                judge1 = str(row.get("judge1_grade", "")).strip()
                if judge1 == "":
                    continue
                relevance = int(float(judge1))
                annotator = "judge1"

            bucket = grouped.setdefault(
                query_id,
                {
                    "query": {
                        "query_id": query_id,
                        "query_text": query_text,
                        "query_language": "de",
                        "domain": "",
                        "intent": "",
                        "expected_themes": [],
                        "ground_truth": [],
                    },
                    "judgments": [],
                },
            )
            bucket["judgments"].append(
                {
                    "dataset_id": dataset_id,
                    "dataset_title": row.get("dataset_title", "").strip(),
                    "relevance": relevance,
                    "annotator": annotator,
                    "notes": row.get("notes", "").strip(),
                }
            )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(grouped, f, indent=2, ensure_ascii=False)

        return grouped

    def compute_agreement(self) -> Dict[str, Any]:
        """Compute agreement statistics from the current pooled CSV."""
        judge1: List[int] = []
        judge2: List[int] = []

        for row in self.rows:
            left = str(row.get("judge1_grade", "")).strip()
            right = str(row.get("judge2_grade", "")).strip()
            if left == "" or right == "":
                continue
            judge1.append(int(float(left)))
            judge2.append(int(float(right)))

        if not judge1:
            return {
                "quadratic_weighted_kappa": 0.0,
                "percentage_agreement": 0.0,
                "disagreement_count": 0,
                "agreement_category": "poor",
                "num_compared": 0,
            }

        kappa = compute_quadratic_weighted_kappa(judge1, judge2)
        return {
            "quadratic_weighted_kappa": kappa,
            "percentage_agreement": compute_percentage_agreement(judge1, judge2),
            "disagreement_count": compute_disagreement_count(judge1, judge2),
            "agreement_category": landis_koch_category(kappa),
            "num_compared": len(judge1),
        }
    
    def annotate_query_interactive(self, query: Dict) -> List[Dict]:
        """
        Interactive annotation for a single query.
        """
        print("\n" + "=" * 70)
        print(f"QUERY: {query['query_id']}")
        print("=" * 70)
        print(f"Text: {query['query_text']}")
        print(f"Intent: {query['intent']}")
        print(f"Domain: {query['domain']}")
        print(f"Expected themes: {query['expected_themes']}")
        print("-" * 70)

        annotations = []
        for row in self.rows:
            if row.get("query_id", "") != query.get("query_id", ""):
                continue

            print(f"\nDataset title: {row.get('dataset_title', '')}")
            print(f"Dataset id: {row.get('dataset_id', '')}")
            print(f"Systems: {row.get('systems_found_in', '')}")

            relevance = self._prompt_grade("Relevance (0-2, blank to skip): ", allow_blank=True)
            if relevance == "":
                continue

            row["judge1_grade"] = relevance
            row["judge2_grade"] = relevance
            row["adjudicated_grade"] = relevance
            row["notes"] = input("Notes (optional): ").strip()
            annotations.append(
                {
                    'dataset_id': row.get("dataset_id", ""),
                    'dataset_title': row.get("dataset_title", ""),
                    'relevance': int(relevance),
                    'annotator': 'manual',
                    'timestamp': datetime.now().isoformat()
                }
            )
            self._save_rows()

        return annotations
    
    def auto_annotate_query(self, query: Dict) -> List[Dict]:
        """
        Automatic annotation based on heuristics (for testing).
        
        Heuristics:
        - Title contains query keywords: +2
        - Theme matches expected: +1
        - Description contains keywords: +1
        - Recent update: +1
        """
        candidates = [row for row in self.rows if row.get("query_id", "") == query.get("query_id", "")]
        if not candidates:
            return []
        
        keywords = query['query_text'].lower().split()
        expected_themes = set(query.get('expected_themes', []))
        
        annotations = []
        
        for c in candidates:
            score = 0
            
            # Check title match
            title_lower = str(c.get("dataset_title", "")).lower()
            title_matches = sum(1 for kw in keywords if kw in title_lower)
            if title_matches >= 2:
                score += 2
            elif title_matches >= 1:
                score += 1
            
            # Check theme match
            systems_found = set(str(c.get("systems_found_in", "")).split("|"))
            if systems_found:
                score += 1
            
            # Check description
            notes_lower = str(c.get("notes", "")).lower()
            desc_matches = sum(1 for kw in keywords if kw in notes_lower)
            if desc_matches >= 2:
                score += 1
            
            # Cap at 3
            relevance = min(score, 3)
            
            annotations.append({
                'dataset_id': c.get("dataset_id", ""),
                'dataset_title': c.get("dataset_title", ""),
                'relevance': relevance,
                'annotator': 'heuristic',
                'timestamp': datetime.now().isoformat()
            })
        
        return annotations
    
    def run_auto_annotation(self, output_file: str = "evaluation/ground_truth_auto.json"):
        """
        Run automatic annotation for all queries.
        """
        print("=" * 70)
        print("AUTOMATIC RELEVANCE ANNOTATION")
        print("=" * 70)
        
        all_annotations = {}
        
        for query in self.queries:
            print(f"\nProcessing {query['query_id']}: {query['query_text'][:50]}...")
            
            annotations = self.auto_annotate_query(query)
            
            all_annotations[query['query_id']] = {
                'query': query,
                'judgments': annotations
            }
            
            # Rate limiting
            time.sleep(0.5)
            
            relevant_count = sum(1 for a in annotations if a['relevance'] > 0)
            print(f"  -> {len(annotations)} candidates, {relevant_count} marked relevant")
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved annotations to {output_file}")
        
        return all_annotations
    
    def generate_summary(self, annotations: Dict) -> str:
        """Generate summary of annotations."""
        lines = [
            "=" * 70,
            "GROUND TRUTH ANNOTATION SUMMARY",
            "=" * 70,
            ""
        ]
        
        for query_id, data in annotations.items():
            judgments = data['judgments']
            relevant = [j for j in judgments if j['relevance'] > 0]
            highly_rel = [j for j in judgments if j['relevance'] >= 2]
            
            lines.append(f"{query_id}: {data['query']['query_text'][:40]}...")
            lines.append(f"  Total judged: {len(judgments)}")
            lines.append(f"  Relevant (rel>0): {len(relevant)}")
            lines.append(f"  Highly relevant (rel>=2): {len(highly_rel)}")
            lines.append("")
        
        lines.append("=" * 70)
        return "\n".join(lines)


def main():
    """Main entry point."""
    import sys
    
    print("=" * 70)
    print("GROUND TRUTH ANNOTATION TOOL")
    print("For Fuzzy OGD Retrieval Evaluation")
    print("=" * 70)
    
    tool = AnnotationTool()

    if len(sys.argv) > 1 and sys.argv[1] == '--export-final':
        tool.export_final_ground_truth()
        print("\nExported final ground truth from adjudicated grades.")
        return

    if len(sys.argv) > 1 and sys.argv[1] == '--agreement':
        agreement = tool.compute_agreement()
        print(json.dumps(agreement, indent=2, ensure_ascii=False))
        return

    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        print("\nRunning pooled-candidate annotation...")
        print("This will load evaluation/data/pooled_candidates.csv and save incrementally.")
        tool.annotate_pooled_candidates_interactive()
        tool.export_final_ground_truth()
        print("\nFinal ground truth exported.")
        return

    print("\nRun with --interactive to annotate pooled candidates, --agreement to compute agreement, or --export-final to write the adjudicated ground truth.")


if __name__ == "__main__":
    main()
