import sys
from pathlib import Path
import json

# Ensure repository root is on path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from evaluation.experiment_runner import FrozenCorpus, _load_query_records, _days_since_modified
from code.ranking.fuzzy_ranker import SimilarityCalculator, MetadataScorer
from code.fuzzy_system.inference_engine import create_inference_engine
from code.query_processing import create_parser


def main():
    corpus = FrozenCorpus.load()

    # Load benchmark queries and pick MOB-04
    queries = _load_query_records(Path('evaluation/benchmark_queries_v2.json'))
    query_record = next((q for q in queries if q.get('query_id') == 'MOB-04'), None)
    if query_record is None:
        print(json.dumps({'error': 'Query MOB-04 not found'}, indent=2))
        return

    query_text = query_record['query_text']

    # Fetch top candidates
    candidates = corpus.search(query_text, rows=10)
    if not candidates:
        print(json.dumps({'error': 'No candidates found in frozen corpus'}, indent=2))
        return

    # Prepare similarity calculator
    sim = SimilarityCalculator()
    sim.fit(candidates)

    parser = create_parser()
    parsed = parser.parse(query_text)
    query_terms = parsed.keywords
    query_themes = parsed.themes

    engine = create_inference_engine(defuzzification='centroid')

    traces = []
    for ds in candidates[:5]:
        ds_id = ds.get('id') or ds.get('name') or ""
        title = ds.get('title')

        thematic_similarity = sim.calculate(query_terms, ds, query_themes=query_themes)
        modified = ds.get('metadata_modified') or ds.get('metadata_modified_date') or ds.get('metadata_created')
        recency_days = _days_since_modified(modified, default=730)
        completeness = MetadataScorer.calculate_completeness(ds)
        resource_avail = MetadataScorer.calculate_resource_availability(ds)

        inputs = {
            'thematic_similarity': float(thematic_similarity),
            'recency': float(recency_days),
            'completeness': float(completeness),
            'resource_availability': float(resource_avail),
        }

        result = engine.infer(inputs)

        trace = {
            'dataset_id': str(ds_id),
            'title': title,
            'inputs': inputs,
            'fuzzification': {
                var: {term: float(mu) for term, mu in fr.memberships.items()}
                for var, fr in result.fuzzification_results.items()
            },
            'rule_activations': [
                {
                    'rule_id': act.rule.id,
                    'firing_strength': float(act.firing_strength),
                    'antecedent_memberships': {f"{var}:{term}": float(mu) for (var, term), mu in act.antecedent_memberships.items()}
                }
                for act in result.rule_activations
            ],
            'dominant_rules': [
                {'rule_id': r.id, 'strength': float(s)} for r, s in result.dominant_rules[:5]
            ],
            'crisp_output': float(result.crisp_output),
            'explanation': result.get_explanation(top_n=5)
        }
        traces.append(trace)

    # Write trace to a UTF-8 JSON file for reliable capture
    output = {
        'query_id': query_record.get('query_id'),
        'query_text': query_text,
        'picked_candidates_traces': traces
    }

    out_path = Path('evaluation/results/mob04_trace.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
