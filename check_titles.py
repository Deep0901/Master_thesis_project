import json

# Load corpus and check for the specific dataset titles from ground truth
with open('data/raw/ogd_metadata_20260306_183841.json', 'r', encoding='utf-8', errors='replace') as f:
    corpus = json.load(f)

print("Checking for ground truth dataset titles in corpus...")
# Check for one specific title: "Statistik der Schweizer Städte 2020"
target_titles = [
    "Statistik der Schweizer Städte 2020",
    "Statistik der Schweizer Städte 2021",
    "Statistik der Schweizer Städte 2022",
]

for dataset in corpus:
    title_de = dataset.get('title', {}).get('de', '') if isinstance(dataset.get('title'), dict) else str(dataset.get('title', ''))
    name = dataset.get('name', '')
    dataset_id = dataset.get('id', '')
    
    for target_title in target_titles:
        if target_title in title_de or target_title == name:
            print(f"Found: {name}")
            print(f"  ID: {dataset_id}")
            print(f"  Title DE: {title_de}")
            print()
            break
