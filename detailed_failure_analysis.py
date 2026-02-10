"""
Detailed failure pattern analysis for HoVer multi-hop retrieval.
Shows concrete examples of why the current approach fails.
"""

import json

# Load examples
with open('data/hoverBench_dev.json', 'r') as f:
    data = json.load(f)

print("=" * 100)
print("DETAILED FAILURE PATTERN ANALYSIS")
print("=" * 100)

examples_to_analyze = [0, 1, 2]  # First 3 examples

for idx in examples_to_analyze:
    example = data[idx]
    required_docs = list(set([sf['key'] for sf in example['supporting_facts']]))

    print(f"\n{'=' * 100}")
    print(f"EXAMPLE {idx + 1}")
    print(f"{'=' * 100}\n")

    print(f"CLAIM: {example['claim']}\n")

    print(f"REQUIRED DOCUMENTS:")
    for i, doc in enumerate(required_docs, 1):
        print(f"  {i}. {doc}")
    print()

    # Analyze the reasoning chain
    print("=" * 100)
    print("REASONING CHAIN ANALYSIS")
    print("=" * 100)

    # Categorize which docs are directly vs indirectly mentioned
    claim_lower = example['claim'].lower()

    direct_docs = []
    indirect_docs = []

    for doc in required_docs:
        doc_lower = doc.lower()
        # Check if doc name or significant parts appear in claim
        if doc_lower in claim_lower:
            direct_docs.append(doc)
        else:
            # Check for partial matches
            doc_words = [w for w in doc_lower.split() if len(w) > 3]
            if any(word in claim_lower for word in doc_words):
                direct_docs.append(doc)
            else:
                indirect_docs.append(doc)

    print(f"\nDirect mentions (can be retrieved from claim): {len(direct_docs)}")
    for doc in direct_docs:
        print(f"  ✓ {doc}")

    print(f"\nIndirect (require multi-hop reasoning): {len(indirect_docs)}")
    for doc in indirect_docs:
        print(f"  → {doc} (not in claim - needs inference)")

    print("\n" + "=" * 100)
    print("SIMULATED EXECUTION WITH CURRENT APPROACH")
    print("=" * 100)

    print("\n--- HOP 1: Retrieve(claim) ---")
    print(f"Query: '{example['claim'][:80]}...'")
    print("Expected behavior:")
    print(f"  - Likely retrieves documents about: {', '.join(direct_docs) if direct_docs else 'general topics'}")
    print(f"  - k=7 documents retrieved")

    if direct_docs:
        print(f"  - May find: {', '.join(direct_docs)}")
    if indirect_docs:
        print(f"  - Will likely MISS: {', '.join(indirect_docs)}")
        print(f"    (These aren't mentioned in the claim)")

    print("\n  Problem: If indirect documents aren't in top-7, they're missed")
    print("  Problem: Retrieval model trained on direct term matching")

    print("\n--- Summarize Hop 1 Results ---")
    print("Expected summary might be:")
    if direct_docs:
        print(f"  'Based on retrieved docs, we found info about {direct_docs[0]}...'")
    print("  Problem: Summary is lossy - specific entity names/relationships may be dropped")

    print("\n--- HOP 2: Generate Query from (claim + summary_1) ---")
    print("LLM must generate a query like:")
    if indirect_docs:
        print(f"  'Who is {indirect_docs[0]}?' or similar")
    print("\n  Problems:")
    print("  - LLM doesn't know what specific entity to search for")
    print("  - Summary may not contain the clue needed for next hop")
    print("  - No explicit reasoning chain or entity extraction")
    print(f"  - Missing entity: {indirect_docs[0] if indirect_docs else 'N/A'}")

    print("\n--- HOP 3: Similar issues compound ---")
    print("  - Depends on success of Hops 1 and 2")
    print("  - If Hop 2 failed, Hop 3 has even less information")

    print("\n" + "=" * 100)
    print("WHY THIS SPECIFIC EXAMPLE FAILS")
    print("=" * 100)

    if idx == 0:
        print("""
Example 1: Trigg Hound and Löwchen
- Claim mentions 'Trigg Hound', 'Löwchen', and 'Foxhounds'
- Hop 1 might retrieve Trigg Hound and Löwchen docs (directly mentioned)
- But needs 'American Foxhound' to verify the claim
- 'American Foxhound' is only partially referenced ('Foxhounds')
- If Hop 1 misses American Foxhound (ranked 8th+ in retrieval), system fails
- Hop 2 query generation: LLM summarizes Trigg/Löwchen info
- LLM must infer: "Need to get general Foxhound info"
- But specific doc name is 'American Foxhound' - may not match generated query
- Result: MISS on required documents → Score: 0.0
        """)

    elif idx == 1:
        print("""
Example 2: River Rat and Coal Miner's Daughter
- Claim mentions both films but NOT the director's name
- Required: 'Thomas Rickman (writer)' - NOT in claim
- Hop 1: Retrieves docs about the two films (direct mentions)
- Summary: "The River Rat and Coal Miner's Daughter are films..."
- Hop 2 query: LLM needs to generate "Who directed The River Rat?"
  * But query might be too generic: "River Rat director"
  * Needs exact match: "Thomas Rickman (writer)"
  * Retrieval model may return "The River Rat" doc again instead of director doc
- Result: MISS on 'Thomas Rickman (writer)' → Score: 0.0
        """)

    elif idx == 2:
        print("""
Example 3: Filbornaskolan and Celtic
- Claim mentions 'Filbornaskolan' and '1977 Scottish League Cup Final'
- Required: 'Henrik Larsson' - NOT in claim
- Hop 1: Retrieves Filbornaskolan and maybe Celtic Cup Final docs
- Summary: "Filbornaskolan is a school... Celtic Cup Final was in 1977..."
- Hop 2 query: LLM must infer the connection (notable alumni)
  * Needs to search for: "notable alumni of Filbornaskolan in Celtic"
  * But doesn't know the specific person is 'Henrik Larsson'
  * Generic query may not retrieve the right person
- Even if some Filbornaskolan doc is retrieved, it might not mention Henrik Larsson
- Result: MISS on 'Henrik Larsson' → Score: 0.0
        """)

    print("\n")

print("=" * 100)
print("ROOT CAUSE: THE BRIDGING ENTITY PROBLEM")
print("=" * 100)
print("""
Multi-hop questions have a common structure:

  Entity A ←→ Bridging Entity ←→ Entity B
  (in claim)   (NOT in claim)    (in claim)

Example:
  "Filbornaskolan" ←→ "Henrik Larsson" ←→ "Celtic Cup Final"
  (mentioned)         (NOT mentioned)      (mentioned)

The BRIDGING ENTITY is the critical missing link.

CURRENT APPROACH FAILS BECAUSE:
1. Direct retrieval on claim gets Entity A and B documents
2. But bridging entity is not in claim, so not retrieved directly
3. Summarization of A/B docs loses specific entity references
4. LLM query generation must "guess" the bridging entity
5. Without explicit entity extraction and reasoning, this guess often fails

WHAT WORKS IN OTHER SYSTEMS:
- IRCoT: Explicitly generates reasoning chain + intermediate queries
- Decompose: Breaks claim into sub-questions for each hop
- ReAct: Uses iterative search and reasoning
- Graph-based: Builds entity graph and traverses it

WHAT'S MISSING HERE:
- No explicit entity extraction
- No sub-question generation
- No verification of retrieval success
- No iterative refinement
- Sequential dependency (can't recover from early mistakes)
""")

print("\n" + "=" * 100)
print("RECOMMENDED FIXES")
print("=" * 100)
print("""
1. ENTITY EXTRACTION + PARALLEL RETRIEVAL
   - Extract all entities from claim upfront
   - Retrieve documents for each entity in parallel
   - Increases k effectively without sequential dependency

2. QUERY DECOMPOSITION
   - Break claim into sub-questions
   - "What is Filbornaskolan?" → "Who are notable alumni?" → "Did any play for Celtic?"
   - Generate specific queries for each sub-question

3. ITERATIVE RETRIEVAL WITH VERIFICATION
   - After each retrieval, check if required entities are found
   - If not, generate refined queries
   - Use feedback loop instead of one-shot approach

4. INCREASE K OR USE ENSEMBLE
   - k=7 is too small - increase to k=15 or k=20
   - Or use multiple retrieval strategies (BM25 + dense + reranker)

5. REDUCE SUMMARIZATION LOSS
   - Don't summarize for next hop
   - Pass raw documents or structured entity lists
   - Use entity-focused summaries instead of content summaries

6. REASONING CHAIN GENERATION
   - Explicitly generate reasoning chain: A → ? → B
   - Identify missing entities in the chain
   - Generate targeted queries for missing entities
""")

print("\n" + "=" * 100)
