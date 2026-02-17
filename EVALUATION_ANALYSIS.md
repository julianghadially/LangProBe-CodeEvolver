# Evaluation Feedback Analysis: Why the System Gets 0.0 Scores

## Executive Summary

The current `HoverMultiHop` implementation is getting **0.0 scores on all test cases** because it fails to retrieve the **specific Wikipedia article titles** required for verification. The system uses a "coverage-driven" approach that analyzes abstract "verification aspects" but does not generate queries that directly target the **named entities** mentioned in the claims.

---

## The Evaluation Metric

The evaluation metric in `/workspace/langProBe/hover/hover_utils.py` is simple and strict:

```python
def discrete_retrieval_eval(example, pred, trace=None):
    gold_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [doc["key"] for doc in example["supporting_facts"]],
        )
    )
    found_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [c.split(" | ")[0] for c in pred.retrieved_docs[:MAX_RETRIEVED_DOCS]],
        )
    )
    return gold_titles.issubset(found_titles)
```

**Success Criteria**: The system must retrieve ALL the gold Wikipedia article titles. If even ONE is missing, the score is 0 (False).

---

## Example Analysis

### Example 1: Trigg Hound and Löwchen
**Claim**: "The variety of animal that the Trigg Hound belongs to and Löwchen are not both types of Foxhounds."

**Required Documents** (supporting_facts):
- American Foxhound
- Löwchen
- Trigg Hound

**Problem**:
- The coverage-driven approach identifies abstract aspects like "entity A existence", "entity B properties", "relationship between A and B"
- Query generation focuses on these abstract aspects rather than directly querying for "Trigg Hound", "Löwchen", or "American Foxhound"
- Result: Retrieved documents about "dog breeds", "hounds", "animals" but NOT the specific Wikipedia articles

### Example 2: River Rat and Coal Miner's Daughter
**Claim**: "The director of the film The River Rat also created the film Coal Miner's Daughter, which won an Academy Award."

**Required Documents**:
- Thomas Rickman (writer)
- Coal Miner's Daughter (film)
- The River Rat

**Problem**:
- Coverage analyzer identifies aspects like "director identity", "film relationships", "Academy Award"
- Queries focus on these aspects: "River Rat film director", "Coal Miner's Daughter Academy Award"
- Result: Retrieved documents about films and awards, but missed the key Wikipedia article "Thomas Rickman (writer)"

### Example 3: Henrik Larsson and Filbornaskolan
**Claim**: "A notable alumni of Filbornaskolan participated in the Celtic 1997 Scottish League Cup Final."

**Required Documents**:
- 1997 Scottish League Cup Final
- Henrik Larsson
- Filbornaskolan

**Problem**:
- Coverage approach identifies "alumni identity", "event details", "school information"
- Queries are abstract: "Filbornaskolan notable alumni", "1997 Celtic match"
- Result: Retrieved documents about Celtic FC, Swedish schools, football finals, but not the specific articles

---

## Root Cause Analysis

### Current Architecture Issues

The current `HoverMultiHop` implementation has these components:

1. **ClaimEntityTrackerSignature**: Extracts abstract "verification aspects"
   - Output: `verification_aspects` (list of abstract aspects)
   - Example: "entity A existence", "entity B properties", "relationship between A and B"

2. **CoverageAnalyzerSignature**: Analyzes coverage of abstract aspects
   - Output: `covered_aspects`, `missing_aspects`, `under_covered_aspects`
   - Problem: Operates on abstract descriptions, not concrete entities

3. **TargetedQueryGeneratorSignature**: Generates queries to fill gaps
   - Input: Abstract aspects like "missing_aspects", "under_covered_aspects"
   - Output: Queries focused on abstract concepts
   - Problem: Does NOT directly target named entities

### Why It Fails

The system follows this failing pattern:

```
Claim: "Boy Hits Car released My Animal and has more albums than The Invisible"
  ↓
Entity Tracker identifies aspects:
  - "Band A album count"
  - "Band B album count"
  - "Album release relationship"
  ↓
Coverage Analyzer assesses abstract coverage:
  - Missing: "Band A identity and discography"
  - Under-covered: "Specific album information"
  ↓
Query Generator creates abstract queries:
  - "Boy Hits Car band discography"
  - "My Animal album release"
  ↓
Retrieval gets semantically related but WRONG documents:
  - ✗ "My Head Is an Animal" (by Of Monsters and Men)
  - ✗ "Animal" (various artists)
  - ✗ "The Animals" (British band)
  - ✗ Documents about music bands and albums
  - ✗ Missing: "Boy Hits Car" Wikipedia article
  - ✗ Missing: "The Invisible (band)" Wikipedia article
  - ✗ Missing: "My Animal" Wikipedia article
```

### The Fundamental Problem

**The system treats entity retrieval as a semantic search problem when it's actually an exact entity lookup problem.**

Multi-hop fact verification requires retrieving **specific Wikipedia articles by their exact titles**, not semantically similar documents. The coverage-driven approach adds layers of abstraction that diffuse the precision needed to retrieve exact entity pages.

---

## Comparison with Baseline

### Baseline Architecture (Score: 34.33%)

```python
# HOP 1
hop1_docs = self.retrieve_k(claim).passages
summary_1 = self.summarize1(claim=claim, passages=hop1_docs).summary

# HOP 2
hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
hop2_docs = self.retrieve_k(hop2_query).passages
summary_2 = self.summarize2(claim=claim, context=summary_1, passages=hop2_docs).summary

# HOP 3
hop3_query = self.create_query_hop3(claim=claim, summary_1=summary_1, summary_2=summary_2).query
hop3_docs = self.retrieve_k(hop3_query).passages
```

**Strengths**:
- Simple, direct queries derived from claim and summaries
- Summarization preserves key named entities
- Queries like "claim, summary_1 -> query" naturally include entity names
- Achieved 34.33% accuracy

**Weaknesses**:
- Information loss in summarization
- Sequential dependencies
- No explicit entity tracking

### Current Coverage-Driven Architecture (Score: 0.0%)

**Strengths** (in theory):
- Explicit aspect tracking
- Systematic coverage analysis
- Targeted gap filling

**Weaknesses** (in practice):
- Abstraction layers hide entity names
- "Verification aspects" are too generic
- Query generation focuses on concepts, not entities
- Coverage analysis operates on abstract aspects, not concrete entity lists
- Achieved **0.0% accuracy** (catastrophic failure)

---

## Why Baseline Performs Better

The baseline's simple approach actually works better because:

1. **Direct Entity Exposure**: Queries are generated with `claim` as input, which contains all entity names
2. **Entity Preservation**: Summaries typically preserve important named entities
3. **Natural Language Queries**: "claim, summary_1 -> query" prompts naturally include entity names

Example baseline flow:
```
Claim: "Boy Hits Car released My Animal"
  ↓
HOP 1: Retrieve(claim) → Gets some documents
  ↓
Summarize → "Boy Hits Car is a band that released My Animal album"
  ↓
HOP 2 Query: "Boy Hits Car My Animal album" (entities preserved!)
  ↓
Retrieval → Higher chance of getting correct articles
```

---

## The Missing Link: Direct Entity Extraction

What the system SHOULD do but DOESN'T:

```python
# What's needed (but not in current code):
entities = extract_entities(claim)
# → ["Boy Hits Car", "My Animal", "The Invisible"]

# Then generate entity-specific queries:
queries = [
    "Boy Hits Car band",
    "My Animal album Boy Hits Car",
    "The Invisible band discography"
]
```

The current `ClaimEntityTrackerSignature` outputs:
- `entities`: "List of key entities mentioned or implied in the claim"
- `key_facts`: "List of specific facts or relationships that need to be verified"
- `verification_aspects`: "List of distinct aspects that need coverage"

But these outputs are **NOT used directly in query generation**! Instead, queries are generated from:
- `missing_aspects` (abstract descriptions)
- `under_covered_aspects` (abstract descriptions)
- `coverage_summary` (narrative text)

The entity list is extracted but then immediately discarded in favor of abstract aspect descriptions.

---

## Key Issues in Current Implementation

### Issue 1: Entity Extraction is Ignored

```python
# Line 72-73: Entities extracted
tracker_output = self.entity_tracker(claim=claim)
verification_aspects = tracker_output.verification_aspects
```

Note: `tracker_output.entities` is extracted but **never used again**!

### Issue 2: Abstract Coverage Analysis

```python
# Lines 82-87: Coverage analysis on abstract aspects
coverage1 = self.coverage_analyzer1(
    claim=claim,
    verification_aspects=verification_aspects,  # Abstract aspects!
    retrieved_titles=hop1_titles,
    passages=hop1_docs
)
```

The coverage analyzer receives `verification_aspects` (abstract descriptions) rather than concrete entity lists.

### Issue 3: Query Generation Without Entities

```python
# Lines 90-96: Query generation from abstract aspects
hop2_query_output = self.query_generator_hop2(
    claim=claim,
    missing_aspects=coverage1.missing_aspects,  # Abstract!
    under_covered_aspects=coverage1.under_covered_aspects,  # Abstract!
    coverage_summary=coverage1.coverage_summary,  # Narrative!
    retrieved_titles=all_retrieved_titles
)
```

No direct entity names are passed to query generation. The query generator must infer entities from abstract descriptions, leading to semantic drift.

---

## Recommended Fixes

### Fix 1: Use Extracted Entities Directly in Queries

```python
# Extract entities
tracker_output = self.entity_tracker(claim=claim)
entities = tracker_output.entities  # USE THIS!

# Generate entity-specific queries
hop2_query = self.query_generator_hop2(
    claim=claim,
    entities=entities,  # Direct entity names
    retrieved_titles=all_retrieved_titles
)
```

### Fix 2: Entity-Centric Coverage Analysis

```python
coverage1 = self.coverage_analyzer1(
    claim=claim,
    required_entities=entities,  # Concrete entity list
    retrieved_titles=hop1_titles,
    passages=hop1_docs
)
# Output: which entities are covered vs missing
```

### Fix 3: Direct Entity Queries for Missing Entities

```python
# If entity "Boy Hits Car" not found in hop1
missing_entities = ["Boy Hits Car", "The Invisible"]

# Generate direct queries
queries = [f"{entity} Wikipedia" for entity in missing_entities]
```

### Fix 4: Hybrid Approach

Combine baseline's directness with explicit entity tracking:

```python
# HOP 1: Direct claim retrieval (like baseline)
hop1_docs = self.retrieve_k(claim).passages

# Extract entities from claim
entities = extract_entity_names(claim)  # Simple, direct extraction

# HOP 2: Query for each missing entity
covered_entities = check_entity_coverage(entities, hop1_docs)
missing = [e for e in entities if e not in covered_entities]
hop2_query = " OR ".join(missing)  # Direct entity names

# HOP 3: Fill remaining gaps
still_missing = check_entity_coverage(entities, hop1_docs + hop2_docs)
hop3_query = " OR ".join(still_missing)
```

---

## Conclusion

The 0.0 score is caused by **over-abstraction**. The system:

1. Extracts entities correctly
2. Immediately converts them to abstract "verification aspects"
3. Analyzes coverage of abstractions
4. Generates queries targeting abstractions
5. Fails to retrieve the specific entity Wikipedia articles needed

**Solution**: Keep it simple. Extract entity names and query for them directly. The baseline's 34.33% score proves that simpler is better for this task.

The coverage-driven approach would work if retrieval were semantic search over diverse documents. But this task requires **exact entity lookup** in a structured knowledge base (Wikipedia). Abstraction hurts rather than helps.

---

## Metrics Summary

| Approach | Architecture | Score | Key Issue |
|----------|-------------|-------|-----------|
| **Baseline** | Summarization-based sequential | **34.33%** | Information loss in summaries |
| **Current** | Coverage-driven with abstraction | **0.0%** | Entity names lost in abstraction |
| **Needed** | Entity-first direct queries | **TBD** | Must preserve entity names throughout |

---

## Next Steps

1. **Immediate**: Revert to baseline or implement entity-first approach
2. **Short-term**: Test hybrid approach (baseline + entity tracking)
3. **Long-term**: Optimize entity extraction and direct entity querying
4. **Validation**: Measure entity coverage at each hop to ensure critical entities are retrieved

---

## Related Files

- `/workspace/langProBe/hover/hover_program.py` - Current failing implementation
- `/workspace/langProBe/hover/hover_utils.py` - Evaluation metric (requires ALL gold titles)
- `/workspace/data/hoverBench_dev.json` - Test data with supporting_facts (gold titles)
- `/workspace/evaluation_hover_baseline/evaluation_results.csv` - Baseline score: 46.67% (note: different from HotpotQA)

---

*Analysis Date: 2026-02-17*
*Current Branch: codeevolver-20260217004441-a9b59e*
*Commit: 8a32bd2*
