"""Debug script to analyze retrieval failures for the first 4 examples."""
import json
import dspy
from langProBe.hover.hover_pipeline import HoverMultiHopPipeline
from langProBe.hover.hover_utils import discrete_retrieval_eval

# Load test data
with open('/workspace/data/hoverBench_test.json', 'r') as f:
    test_data = json.load(f)

# Initialize the pipeline
pipeline = HoverMultiHopPipeline()

# Setup LM (using a default model for testing)
pipeline.setup_lm("openai/gpt-4o-mini")

print("="*80)
print("ANALYZING FIRST 4 EXAMPLES FOR RETRIEVAL FAILURES")
print("="*80)

# Analyze first 4 examples
for i in range(4):
    example = test_data[i]
    print(f"\n{'='*80}")
    print(f"EXAMPLE {i}")
    print(f"{'='*80}")

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
    try:
        # Create DSPy example
        dspy_example = dspy.Example(**example).with_inputs('claim')

        # Run prediction
        pred = pipeline(claim=example['claim'])

        # Get retrieved docs
        retrieved_docs = pred.retrieved_docs[:21]  # Top 21 as per MAX_RETRIEVED_DOCS

        print(f"\nRetrieved {len(retrieved_docs)} documents:")

        # Extract and normalize retrieved titles
        retrieved_titles = []
        retrieved_titles_normalized = set()

        for j, doc in enumerate(retrieved_docs):
            # Docs are in format "Title | Text"
            title = doc.split(" | ")[0]
            retrieved_titles.append(title)
            normalized_title = dspy.evaluate.normalize_text(title)
            retrieved_titles_normalized.add(normalized_title)
            print(f"  {j+1}. {title}")

        # Check which gold titles are found
        print(f"\nGold titles (normalized): {gold_titles_normalized}")
        print(f"Found in retrieved docs:")
        for gold_title in gold_titles_normalized:
            found = gold_title in retrieved_titles_normalized
            status = "✓ FOUND" if found else "✗ MISSING"
            print(f"  {status}: {gold_title}")

        # Evaluate
        score = discrete_retrieval_eval(example, pred)
        print(f"\nScore: {score} (1.0 = success, 0.0 = failure)")

        if not score:
            print("\n⚠️  FAILURE ANALYSIS:")
            missing = gold_titles_normalized - retrieved_titles_normalized
            print(f"Missing documents: {missing}")

    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
