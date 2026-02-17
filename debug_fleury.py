"""Debug the François de Fleury example."""
import json
import dspy
from langProBe.hover.hover_pipeline import HoverMultiHopPipeline
from langProBe.hover.hover_utils import discrete_retrieval_eval

# Load training data
with open('/workspace/data/hoverBench_train.json', 'r') as f:
    train_data = json.load(f)

# Initialize the pipeline
pipeline = HoverMultiHopPipeline()
pipeline.setup_lm("openai/gpt-4o-mini")

example = train_data[55]

print("="*80)
print("FRANÇOIS DE FLEURY EXAMPLE")
print("="*80)

print(f"\nClaim: {example['claim']}")
print(f"Label: {example['label']}")

# Get supporting facts needed
print(f"\nSupporting facts needed:")
gold_titles = []
for sf in example['supporting_facts']:
    title = sf['key']
    sent_id = sf['value']
    gold_titles.append(title)
    print(f"  - {title} (sentence {sent_id})")

# Normalize gold titles
gold_titles_normalized = set(map(dspy.evaluate.normalize_text, gold_titles))

# Run the pipeline
pred = pipeline(claim=example['claim'])

# Get retrieved docs
retrieved_docs = pred.retrieved_docs[:21]

print(f"\nRetrieved {len(retrieved_docs)} documents:")

# Extract and normalize retrieved titles
retrieved_titles_normalized = set()

for j, doc in enumerate(retrieved_docs):
    title = doc.split(" | ")[0]
    normalized_title = dspy.evaluate.normalize_text(title)
    retrieved_titles_normalized.add(normalized_title)

    # Check if this is a gold doc
    is_gold = normalized_title in gold_titles_normalized
    marker = "★" if is_gold else " "
    print(f"  {marker} {j+1}. {title}")

# Check which gold titles are found
print(f"\n\nGold titles (normalized): {gold_titles_normalized}")
print(f"Found in retrieved docs:")
found_count = 0
for gold_title in gold_titles_normalized:
    found = gold_title in retrieved_titles_normalized
    status = "✓ FOUND" if found else "✗ MISSING"
    if found:
        found_count += 1
    print(f"  {status}: {gold_title}")

# Evaluate
score = discrete_retrieval_eval(example, pred)
print(f"\nScore: {score} ({found_count}/{len(gold_titles_normalized)} gold docs found)")

if not score:
    print("\n⚠️  FAILURE ANALYSIS:")
    missing = gold_titles_normalized - retrieved_titles_normalized
    print(f"Missing documents: {missing}")
