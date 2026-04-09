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
import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

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
    
    BASE_URL = "https://opendata.swiss/api/3/action"
    
    def __init__(self, queries_file: str = "evaluation/benchmark_queries_v2.json"):
        self.queries_file = Path(queries_file)
        self.queries = []
        self.annotations = {}
        
        # Load queries
        if self.queries_file.exists():
            with open(self.queries_file, 'r', encoding='utf-8') as f:
                self.queries = json.load(f)
    
    def search_portal(self, query_text: str, rows: int = 30) -> List[DatasetCandidate]:
        """
        Search opendata.swiss for candidate datasets.
        """
        params = {
            'q': query_text,
            'rows': rows
        }
        
        try:
            resp = requests.get(f"{self.BASE_URL}/package_search", params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json()['result']['results']
            
            candidates = []
            for ds in results:
                # Get multilingual title
                title = ds.get('title', {})
                if isinstance(title, dict):
                    title_str = title.get('de', '') or title.get('en', '') or title.get('fr', '') or str(title)
                else:
                    title_str = str(title)
                
                # Get description
                desc = ds.get('notes', '') or ds.get('description', '')
                if isinstance(desc, dict):
                    desc_str = desc.get('de', '') or desc.get('en', '') or desc.get('fr', '') or ''
                else:
                    desc_str = str(desc)[:500]
                
                # Get organization
                org = ds.get('organization', {})
                org_name = org.get('title', {}).get('de', org.get('name', 'Unknown')) if isinstance(org.get('title'), dict) else org.get('title', org.get('name', 'Unknown'))
                
                # Get resources
                resources = ds.get('resources', [])
                formats = list(set(r.get('format', '').upper() for r in resources if r.get('format')))
                
                candidates.append(DatasetCandidate(
                    dataset_id=ds.get('id', ''),
                    name=ds.get('name', ''),
                    title=title_str,
                    description=desc_str,
                    organization=str(org_name),
                    themes=[g.get('name', '') for g in ds.get('groups', [])],
                    num_resources=len(resources),
                    formats=formats[:5],
                    metadata_modified=ds.get('metadata_modified', '')
                ))
            
            return candidates
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
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
        
        # Search for candidates
        print(f"\nSearching opendata.swiss for '{query['query_text']}'...")
        candidates = self.search_portal(query['query_text'], rows=20)
        
        if not candidates:
            print("No results found!")
            return []
        
        print(f"\nFound {len(candidates)} candidate datasets. Please annotate each:\n")
        print("Relevance scale:")
        print("  0 = Not relevant")
        print("  1 = Marginally relevant")
        print("  2 = Relevant")
        print("  3 = Highly relevant")
        print("  s = Skip")
        print("  q = Quit annotation")
        print()
        
        annotations = []
        
        for i, c in enumerate(candidates, 1):
            print(f"\n--- Dataset {i}/{len(candidates)} ---")
            print(f"ID: {c.dataset_id[:8]}...")
            print(f"Title: {c.title}")
            print(f"Org: {c.organization}")
            print(f"Themes: {', '.join(c.themes)}")
            print(f"Formats: {', '.join(c.formats)}")
            print(f"Description: {c.description[:200]}...")
            
            while True:
                user_input = input("\nRelevance (0-3, s=skip, q=quit): ").strip().lower()
                
                if user_input == 'q':
                    return annotations
                elif user_input == 's':
                    break
                elif user_input in ['0', '1', '2', '3']:
                    annotations.append({
                        'dataset_id': c.dataset_id,
                        'dataset_title': c.title,
                        'relevance': int(user_input),
                        'annotator': 'manual',
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"  -> Marked as relevance={user_input}")
                    break
                else:
                    print("  Invalid input. Enter 0-3, s, or q.")
        
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
        candidates = self.search_portal(query['query_text'], rows=30)
        
        keywords = query['query_text'].lower().split()
        expected_themes = set(query.get('expected_themes', []))
        
        annotations = []
        
        for c in candidates:
            score = 0
            
            # Check title match
            title_lower = c.title.lower()
            title_matches = sum(1 for kw in keywords if kw in title_lower)
            if title_matches >= 2:
                score += 2
            elif title_matches >= 1:
                score += 1
            
            # Check theme match
            if set(c.themes) & expected_themes:
                score += 1
            
            # Check description
            desc_lower = c.description.lower()
            desc_matches = sum(1 for kw in keywords if kw in desc_lower)
            if desc_matches >= 2:
                score += 1
            
            # Cap at 3
            relevance = min(score, 3)
            
            annotations.append({
                'dataset_id': c.dataset_id,
                'dataset_title': c.title,
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
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        # Interactive mode
        print("\nRunning interactive annotation...")
        print("This will present datasets for manual relevance judgment.")
        
        all_annotations = {}
        
        for query in tool.queries:
            annotations = tool.annotate_query_interactive(query)
            if annotations:
                all_annotations[query['query_id']] = {
                    'query': query,
                    'judgments': annotations
                }
        
        if all_annotations:
            output_file = "evaluation/ground_truth_manual.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_annotations, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to {output_file}")
    
    else:
        # Automatic mode (heuristic-based)
        print("\nRunning automatic annotation (heuristic-based)...")
        print("For production, run with --interactive for manual annotation.\n")
        
        annotations = tool.run_auto_annotation()
        
        print("\n" + tool.generate_summary(annotations))


if __name__ == "__main__":
    main()
