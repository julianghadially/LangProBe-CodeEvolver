"""Comprehensive analysis of retrieval failures across test set."""
import json
import dspy
from langProBe.hover.hover_pipeline import HoverMultiHopPipeline
from langProBe.hover.hover_utils import discrete_retrieval_eval

# Load test data
with open('/workspace/data/hoverBench_test.json', 'r') as f:
    test_data = json.load(f)

# Initialize the pipeline
pipeline = HoverMultiHopPipeline()
pipeline.setup_lm("openai/gpt-4o-mini")

print("="*80)
print("COMPREHENSIVE ANALYSIS OF TEST SET FAILURES")
print("="*80)

# Analyze first 20 examples
num_examples = min(20, len(test_data))
failures = []
successes = []
missing_doc_counts = []

for i in range(num_examples):
    example = test_data[i]

    # Get gold titles
    gold_titles = [sf['key'] for sf in example['supporting_facts']]
    gold_titles_normalized = set(map(dspy.evaluate.normalize_text, gold_titles))

    # Run prediction
    try:
        pred = pipeline(claim=example['claim'])
        retrieved_docs = pred.retrieved_docs[:21]

        # Extract normalized retrieved titles
        retrieved_titles_normalized = set()
        for doc in retrieved_docs:
            title = doc.split(" | ")[0]
            normalized_title = dspy.evaluate.normalize_text(title)
            retrieved_titles_normalized.add(normalized_title)

        # Check score
        score = discrete_retrieval_eval(example, pred)

        # Count found/missing
        found = gold_titles_normalized & retrieved_titles_normalized
        missing = gold_titles_normalized - retrieved_titles_normalized

        result = {
            'index': i,
            'claim': example['claim'][:80] + '...' if len(example['claim']) > 80 else example['claim'],
            'score': score,
            'total_gold': len(gold_titles_normalized),
            'found': len(found),
            'missing': len(missing),
            'missing_docs': missing,
            'gold_docs': gold_titles_normalized,
        }

        if score:
            successes.append(result)
        else:
            failures.append(result)

        missing_doc_counts.append(len(missing))

    except Exception as e:
        print(f"Error on example {i}: {e}")
        continue

# Print summary
print(f"\n{'='*80}")
print(f"SUMMARY (first {num_examples} examples)")
print(f"{'='*80}")
print(f"Total evaluated: {num_examples}")
print(f"Successes: {len(successes)} ({len(successes)/num_examples*100:.1f}%)")
print(f"Failures: {len(failures)} ({len(failures)/num_examples*100:.1f}%)")

# Analyze failure patterns
print(f"\n{'='*80}")
print("FAILURE PATTERNS")
print(f"{'='*80}")

if failures:
    # Count how many docs missing per failure
    from collections import Counter
    missing_counts = Counter([f['missing'] for f in failures])
    print(f"\nDistribution of missing documents:")
    for count, freq in sorted(missing_counts.items()):
        print(f"  {count} missing doc(s): {freq} examples")

    # Show detailed failures
    print(f"\n{'='*80}")
    print("DETAILED FAILURE EXAMPLES")
    print(f"{'='*80}")
    for f in failures:
        print(f"\nExample {f['index']}")
        print(f"Claim: {f['claim']}")
        print(f"Found: {f['found']}/{f['total_gold']} gold docs")
        print(f"Missing: {f['missing_docs']}")
        print(f"All gold docs needed: {f['gold_docs']}")

# Analyze success examples too
if successes:
    print(f"\n{'='*80}")
    print("SUCCESS EXAMPLES (for comparison)")
    print(f"{'='*80}")
    for s in successes[:3]:
        print(f"\nExample {s['index']}")
        print(f"Claim: {s['claim']}")
        print(f"Found: {s['found']}/{s['total_gold']} gold docs (ALL FOUND)")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
