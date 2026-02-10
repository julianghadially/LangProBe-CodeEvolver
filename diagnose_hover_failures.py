"""
Diagnostic script to understand why HoVer multi-hop retrieval is failing.
This simulates the three-hop approach and shows what's happening at each step.
"""

import json

# Load a few examples
with open('data/hoverBench_dev.json', 'r') as f:
    data = json.load(f)

print("=" * 100)
print("DIAGNOSTIC ANALYSIS: HoVer Multi-Hop Retrieval Failures")
print("=" * 100)

# Let's analyze the first 3 examples
for example_idx in range(3):
    example = data[example_idx]

    print(f"\n{'=' * 100}")
    print(f"EXAMPLE {example_idx + 1}")
    print(f"{'=' * 100}")

    print(f"\nCLAIM: {example['claim']}")

    # Required documents
    required_docs = [sf['key'] for sf in example['supporting_facts']]
    print(f"\nREQUIRED SUPPORTING DOCUMENTS ({len(set(required_docs))} unique):")
    for doc in set(required_docs):
        print(f"  ✓ {doc}")

    print(f"\n{'─' * 100}")
    print("CURRENT APPROACH ANALYSIS (HoverMultiHopPredict)")
    print(f"{'─' * 100}")

    print("""
The current 3-hop approach:
1. HOP 1: Retrieve(claim) → get k=7 docs → Summarize
2. HOP 2: Generate query from (claim, summary_1) → Retrieve → get k=7 docs → Summarize
3. HOP 3: Generate query from (claim, summary_1, summary_2) → Retrieve → get k=7 docs
4. Return: hop1_docs + hop2_docs + hop3_docs (21 total docs)
    """)

    print("IDENTIFIED PROBLEMS:")
    print()

    # Problem 1: Sequential dependency
    print("1. SEQUENTIAL DEPENDENCY PROBLEM")
    print("   - Each hop depends on the summary of the previous hop")
    print("   - If Hop 1 misses a key document, it won't be in summary_1")
    print("   - Hop 2 query generation won't know to look for it")
    print("   - This creates a cascading failure")
    print()

    # Problem 2: Query generation bottleneck
    print("2. QUERY GENERATION BOTTLENECK")
    print("   - Hops 2 and 3 rely on LLM to generate the 'right' search query")
    print("   - LLM must infer what to search for from incomplete summaries")
    print("   - LLM may not understand the multi-hop reasoning needed")
    print(f"   - For this example, the claim mentions: {example['claim'][:80]}...")

    # Extract key entities from claim
    claim_words = set(example['claim'].split())
    required_words = set()
    for doc in required_docs:
        required_words.update(doc.split())

    # Check overlap
    overlap = claim_words.intersection(required_words)
    print(f"   - Words in claim that overlap with required docs: {len(overlap)} / {len(required_words)} required words")
    print()

    # Problem 3: Summarization loses specificity
    print("3. SUMMARIZATION LOSES SPECIFICITY")
    print("   - After Hop 1, documents are summarized")
    print("   - Summaries may lose specific entity names, dates, or relationships")
    print("   - Hop 2 query generation works from this lossy summary")
    print("   - Critical entities needed for Hop 2/3 retrieval may be lost")
    print()

    # Problem 4: Limited retrieval per hop
    print("4. LIMITED RETRIEVAL PER HOP (k=7)")
    print("   - Each hop retrieves only 7 documents")
    print("   - If target document is ranked 8th or lower, it's missed")
    print("   - With 3 hops × 7 docs = 21 total, but no guarantee of coverage")
    print("   - The retrieval space is huge, but we're only sampling 7 per query")
    print()

    # Problem 5: No explicit entity extraction
    print("5. NO EXPLICIT ENTITY/RELATIONSHIP EXTRACTION")
    print("   - System doesn't explicitly extract entities from the claim")
    print("   - Doesn't decompose multi-hop reasoning chain")

    # Analyze the specific case
    print()
    print(f"{'─' * 100}")
    print("SPECIFIC FAILURE ANALYSIS FOR THIS EXAMPLE")
    print(f"{'─' * 100}")
    print()

    # Check if required docs are mentioned in claim
    print("Document Visibility in Claim:")
    for doc in set(required_docs):
        if doc in example['claim']:
            print(f"  ✓ '{doc}' - DIRECTLY MENTIONED in claim")
        else:
            # Check for partial matches
            doc_parts = doc.lower().split()
            claim_lower = example['claim'].lower()
            partial_match = any(part in claim_lower for part in doc_parts if len(part) > 3)
            if partial_match:
                print(f"  ~ '{doc}' - PARTIALLY mentioned in claim")
            else:
                print(f"  ✗ '{doc}' - NOT mentioned in claim (requires inference)")

    print()
    print("Multi-hop Reasoning Required:")
    print(f"  This is a {len(set(required_docs))}-hop question")
    print(f"  Reasoning chain likely involves:")

    # Try to infer reasoning chain
    for i, doc in enumerate(set(required_docs), 1):
        print(f"    {i}. Need to retrieve: '{doc}'")

    print()

print("\n" + "=" * 100)
print("CORE PROBLEMS SUMMARY")
print("=" * 100)
print()

print("""
1. CASCADING FAILURES
   - Sequential hop structure means early mistakes compound
   - Missing a document in Hop 1 affects all subsequent hops

2. LOSSY SUMMARIZATION
   - Summarizing 7 documents loses critical details
   - Next hop's query generation works from incomplete information

3. IMPLICIT QUERY GENERATION
   - LLM must "guess" what to search for next
   - No explicit decomposition of the reasoning chain
   - No guarantee the right entities/relationships are targeted

4. INSUFFICIENT COVERAGE
   - k=7 per hop is too small for the retrieval space
   - 21 total docs may not cover all required documents
   - No verification that required docs are in the set

5. NO CLAIM DECOMPOSITION
   - System doesn't explicitly break down the claim into sub-queries
   - Doesn't identify entities that need to be retrieved
   - No explicit multi-hop reasoning plan

WHAT'S NEEDED:
- Explicit entity extraction from claims
- Parallel retrieval instead of sequential
- Larger k or multiple retrieval strategies
- Query decomposition (break claim into sub-questions)
- Verification that required entities are retrieved
- Less reliance on LLM query generation, more on structured retrieval
""")

print("\n" + "=" * 100)
