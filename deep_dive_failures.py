"""Deep dive into why specific documents are not retrieved."""
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

# Focus on the most interesting failure cases
failure_indices = [0, 1, 2]  # Examples with different missing docs

for idx in failure_indices:
    example = test_data[idx]

    print(f"\n{'='*80}")
    print(f"DEEP DIVE - EXAMPLE {idx}")
    print(f"{'='*80}")

    print(f"\nClaim: {example['claim']}")

    # Get gold titles
    gold_titles = [sf['key'] for sf in example['supporting_facts']]
    gold_titles_normalized = set(map(dspy.evaluate.normalize_text, gold_titles))

    print(f"\nGold documents needed:")
    for title in gold_titles:
        print(f"  - {title}")

    # Run prediction
    pred = pipeline(claim=example['claim'])
    retrieved_docs = pred.retrieved_docs[:21]

    # Analyze each hop
    print(f"\nHOP-BY-HOP ANALYSIS:")
    hop_size = 7
    hops = [
        retrieved_docs[0:7],
        retrieved_docs[7:14],
        retrieved_docs[14:21]
    ]

    for hop_num, hop_docs in enumerate(hops, 1):
        print(f"\n  Hop {hop_num} (7 documents):")
        hop_titles_normalized = set()

        for j, doc in enumerate(hop_docs):
            title = doc.split(" | ")[0]
            normalized_title = dspy.evaluate.normalize_text(title)
            hop_titles_normalized.add(normalized_title)

            is_gold = normalized_title in gold_titles_normalized
            marker = "★" if is_gold else " "
            print(f"    {marker} {j+1}. {title}")

        # Check what was found in this hop
        found_in_hop = hop_titles_normalized & gold_titles_normalized
        if found_in_hop:
            print(f"    → Found gold docs: {found_in_hop}")

    # Overall analysis
    all_retrieved_normalized = set()
    for doc in retrieved_docs:
        title = doc.split(" | ")[0]
        normalized_title = dspy.evaluate.normalize_text(title)
        all_retrieved_normalized.add(normalized_title)

    found = gold_titles_normalized & all_retrieved_normalized
    missing = gold_titles_normalized - all_retrieved_normalized

    print(f"\n\nOVERALL RESULTS:")
    print(f"  ✓ Found: {found}")
    print(f"  ✗ Missing: {missing}")

    # Analyze what's "close" to the missing docs
    if missing:
        print(f"\n  ANALYSIS OF MISSING DOCUMENTS:")
        for missing_doc in missing:
            print(f"\n    Missing: '{missing_doc}'")
            # Check if there are similar titles
            similar = []
            for retrieved_title in all_retrieved_normalized:
                # Check for partial matches
                missing_words = set(missing_doc.lower().split())
                retrieved_words = set(retrieved_title.lower().split())
                overlap = missing_words & retrieved_words
                if overlap and len(overlap) >= 1:
                    similar.append((retrieved_title, overlap))

            if similar:
                print(f"    Similar retrieved docs (word overlap):")
                for sim_title, overlap in similar[:5]:
                    print(f"      - '{sim_title}' (overlap: {overlap})")
            else:
                print(f"    No similar documents found in retrieved set")
                print(f"    This suggests the query didn't target this topic")

print(f"\n{'='*80}")
print("DEEP DIVE COMPLETE")
print(f"{'='*80}")
