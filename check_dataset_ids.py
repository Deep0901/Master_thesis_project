import json

with open('data/raw/ogd_metadata_20260306_183841.json', 'r', encoding='utf-8', errors='replace') as f:
    corpus = json.load(f)
print(f'Frozen corpus has {len(corpus)} datasets')
print('Sample IDs:', [d.get('dataset_id') for d in corpus[:3]])

# Check one ground truth dataset ID
with open('evaluation/ground_truth_manual.json', 'r', encoding='utf-8') as f:
    gt = json.load(f)
    first_query = list(gt.values())[0]
    first_judgment = first_query.get('judgments', [])[0] if first_query.get('judgments') else {}
    print('First GT dataset ID:', first_judgment.get('dataset_id'))
    print('First GT dataset title:', first_judgment.get('dataset_title'))
    
# Check if IDs match
corpus_ids = set(d.get('dataset_id') for d in corpus)
gt_ids = set()
for query_data in gt.values():
    for judgment in query_data.get('judgments', []):
        gt_ids.add(judgment.get('dataset_id'))
print(f'Corpus has {len(corpus_ids)} unique dataset IDs')
print(f'Ground truth has {len(gt_ids)} unique dataset IDs')
print(f'Overlap: {len(corpus_ids & gt_ids)} datasets')
print(f'GT IDs not in corpus: {len(gt_ids - corpus_ids)} datasets')
if gt_ids - corpus_ids:
    print('Sample missing IDs:', list(gt_ids - corpus_ids)[:3])
