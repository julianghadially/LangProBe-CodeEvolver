"""Demo script to test the Chain-of-Thought Verifier module.

This script demonstrates the new ChainOfThoughtVerifier module that performs
multi-step reasoning on retrieved documents to verify claims.
"""

import json
import dspy
from langProBe.hover.hover_pipeline import HoverMultiHopPredictPipeline

# Configure DSPy with OpenAI
print("Configuring DSPy with GPT-4o-mini...")
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Load a sample from the training data
print("\nLoading sample data...")
with open("/workspace/data/hoverBench_train.json") as f:
    data = json.load(f)

# Take the first example
example = data[0]
print(f"\n{'='*80}")
print("CLAIM TO VERIFY:")
print(f"{'='*80}")
print(f"{example['claim']}")
print(f"\nGround Truth Label: {example['label']} (0=NOT_SUPPORTED, 1=SUPPORTED)")

# Create the pipeline
print(f"\n{'='*80}")
print("INITIALIZING CHAIN-OF-THOUGHT VERIFIER PIPELINE")
print(f"{'='*80}")
pipeline = HoverMultiHopPredictPipeline()

# Run the pipeline
print("\nExecuting multi-hop retrieval and Chain-of-Thought verification...")
print("(This may take a minute as it performs retrieval and reasoning...)\n")

try:
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        result = pipeline(claim=example["claim"])

    # Display results
    print(f"\n{'='*80}")
    print("VERIFICATION RESULTS")
    print(f"{'='*80}")
    print(f"\nPredicted Label: {result.label} (0=NOT_SUPPORTED, 1=SUPPORTED)")
    print(f"Ground Truth: {example['label']}")
    print(f"Correct: {result.label == example['label']}")

    print(f"\n{'-'*80}")
    print("VERIFICATION DECISION:")
    print(f"{'-'*80}")
    print(result.verification_decision)

    print(f"\n{'-'*80}")
    print("EXTRACTED FACTS:")
    print(f"{'-'*80}")
    if isinstance(result.facts, list):
        for i, fact in enumerate(result.facts, 1):
            print(f"{i}. {fact}")
    else:
        print(result.facts)

    print(f"\n{'-'*80}")
    print("REASONING STEPS:")
    print(f"{'-'*80}")
    if isinstance(result.reasoning_steps, list):
        for i, step in enumerate(result.reasoning_steps, 1):
            print(f"{i}. {step}")
    else:
        print(result.reasoning_steps)

    print(f"\n{'-'*80}")
    print("EXPLICIT COMPARISONS:")
    print(f"{'-'*80}")
    if isinstance(result.comparisons, list):
        for i, comparison in enumerate(result.comparisons, 1):
            print(f"{i}. {comparison}")
    else:
        print(result.comparisons)

    print(f"\n{'-'*80}")
    print("RETRIEVED DOCUMENTS (first 3):")
    print(f"{'-'*80}")
    for i, doc in enumerate(result.retrieved_docs[:3], 1):
        print(f"\n{i}. {doc[:200]}...")

    print(f"\n{'='*80}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*80}")
    print("\nThe Chain-of-Thought Verifier successfully:")
    print("✓ Retrieved relevant documents using multi-hop retrieval")
    print("✓ Extracted key facts from the documents")
    print("✓ Performed step-by-step logical reasoning")
    print("✓ Made explicit comparisons and calculations")
    print("✓ Generated a final verification decision with explanation")

except Exception as e:
    print(f"\n{'='*80}")
    print("ERROR OCCURRED")
    print(f"{'='*80}")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print("\nNote: This demo requires:")
    print("1. OPENAI_API_KEY environment variable to be set")
    print("2. Network access to the ColBERTv2 server")
    print("3. DSPy and all dependencies installed")
