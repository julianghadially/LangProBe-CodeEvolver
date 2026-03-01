#!/usr/bin/env python3
"""
Script to analyze retrieval gaps by running evaluation on a small set
and capturing detailed feedback about what was retrieved vs. what was needed.
"""
import json
import sys
from pathlib import Path

import dspy
from langProBe.hover.hover_program import HoverMultiHop
from langProBe.hover.hover_utils import discrete_retrieval_eval

# Load test data
DATA_DIR = Path(__file__).resolve().parent / "data"

def load_examples(filename, n=10):
    with open(DATA_DIR / filename) as f:
        raw = json.load(f)
    examples = []
    for item in raw[:n]:
        ex = dspy.Example(**item).with_inputs("claim")
        examples.append(ex)
    return examples

def analyze_retrieval_gaps():
    """Analyze failed retrievals to identify patterns."""

    # Load examples
    examples = load_examples("hoverBench_test.json", n=20)

    # Setup
    lm = dspy.LM("openai/gpt-4o-mini")
    rm = dspy.ColBERTv2(url="https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search")

    # Create program
    program = HoverMultiHop()
    program.setup_lm(lm)

    results = []

    print("Running evaluation on examples...")
    with dspy.context(lm=lm, rm=rm):
        for idx, example in enumerate(examples):
            try:
                # Run prediction
                pred = program(claim=example.claim)

                # Evaluate
                score = discrete_retrieval_eval(example, pred)

                # Extract gold and retrieved doc titles
                gold_titles = [doc["key"] for doc in example.supporting_facts]
                retrieved_titles = [
                    c.split(" | ")[0] for c in pred.retrieved_docs[:21]
                ]

                # Store result
                result = {
                    "idx": idx,
                    "claim": example.claim,
                    "gold_docs": gold_titles,
                    "retrieved_docs": retrieved_titles,
                    "score": 1.0 if score else 0.0,
                    "label": example.label if hasattr(example, "label") else None,
                }
                results.append(result)

                print(f"Example {idx}: Score={result['score']}")

            except Exception as e:
                print(f"Error on example {idx}: {e}")
                continue

    # Analyze failed cases
    failed_cases = [r for r in results if r["score"] == 0.0]

    print(f"\n{'='*80}")
    print(f"ANALYSIS: {len(failed_cases)} failed out of {len(results)} total")
    print(f"{'='*80}\n")

    for case in failed_cases:
        print(f"\n{'─'*80}")
        print(f"FAILED Example {case['idx']}")
        print(f"{'─'*80}")
        print(f"Claim: {case['claim'][:200]}...")
        print(f"\nGold documents needed:")
        for doc in case["gold_docs"]:
            print(f"  ✓ {doc}")

        print(f"\nRetrieved documents:")
        for doc in case["retrieved_docs"][:10]:  # Show first 10
            print(f"  • {doc}")

        # Find missing
        gold_set = set(case["gold_docs"])
        retrieved_set = set(case["retrieved_docs"])
        missing = gold_set - retrieved_set

        print(f"\nMissing documents:")
        for doc in missing:
            print(f"  ✗ {doc}")

        print(f"\nLabel: {case['label']}")

    # Save to file
    output_file = Path("/workspace/retrieval_gap_analysis.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    analyze_retrieval_gaps()
