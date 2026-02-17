"""Search for specific claims mentioned by the user."""
import json

# Load all datasets
datasets = {
    'test': '/workspace/data/hoverBench_test.json',
    'train': '/workspace/data/hoverBench_train.json',
    'val': '/workspace/data/hoverBench_val.json',
    'dev': '/workspace/data/hoverBench_dev.json',
}

keywords_to_search = [
    ('Burn Gorman', 'Crimson Peak', 'Guillermo del Toro'),
    ('Godhead', 'Joe Lynn Turner'),
    ('François de Fleury', 'Battle of Monmouth'),
    ('G.I. Joe', 'Channing Tatum'),
]

for dataset_name, file_path in datasets.items():
    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*80}")
    print(f"Searching {dataset_name} set ({len(data)} examples)")
    print(f"{'='*80}")

    for keyword_group in keywords_to_search:
        for i, example in enumerate(data):
            claim = example['claim']
            # Check if any keyword from this group is in the claim
            if any(keyword.lower() in claim.lower() for keyword in keyword_group):
                print(f"\n--- Example {i} (keywords: {keyword_group[0]}) ---")
                print(f"Claim: {claim}")
                print(f"Supporting facts:")
                for sf in example['supporting_facts']:
                    print(f"  - {sf['key']} (sentence {sf['value']})")
