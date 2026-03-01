# Retrieval Gap Analysis - HoVer Multi-Hop System

## Overview

This analysis examines the retrieval architecture based on the HoverMultiHop system implementation in `/workspace/langProBe/hover/hover_program.py` and identifies potential retrieval gaps that could lead to evaluation failures.

## System Architecture

### Current HoverMultiHop Pipeline

```
Input: claim
  ↓
Step 1: Entity & Gap Analysis (EntityAndGapAnalyzer)
  - Extracts 2-3 entity chains from claim
  - Generates 2-3 parallel search queries
  ↓
Step 2: Parallel Retrieval (dspy.Retrieve)
  - Retrieves k=23 documents per query
  - Total pool: 46-69 documents (with duplicates)
  ↓
Step 3: Deduplication & Listwise Reranking (ListwiseDocumentReranker)
  - Removes duplicate documents
  - Evaluates all documents together to identify multi-hop relationships
  - Outputs ranked list of document indices based on interdependencies
  - Returns top 21 documents in optimal order
  ↓
Output: retrieved_docs (max 21)
```

### Evaluation Metric

The system is evaluated using `discrete_retrieval_eval`:
- **Gold Standard**: Set of document titles from `supporting_facts`
- **Retrieved Set**: Top 21 document titles from prediction
- **Scoring**: Binary (1.0 if gold ⊆ retrieved, 0.0 otherwise)
- **Constraint**: All-or-nothing metric - missing even one gold document results in failure

## Sample Test Cases Analysis

### Example 1: Bridging Entity Challenge
```
Claim: "Antonis Fotsis is a player for the club who's name has the starting
        letter from an alphabet derived from the Phoenician alphabet."

Gold Documents Needed:
  1. Ilysiakos B.C. (the club)
  2. Greek alphabet (the alphabet connection)
  3. Antonis Fotsis (the player)

Challenge: The "Greek alphabet" is a bridging entity that connects
"Ilysiakos B.C." (which has a Greek name) to "Phoenician alphabet".
A query focused on "Antonis Fotsis" or "Ilysiakos" may not surface
"Greek alphabet" as a relevant document.
```

### Example 2: Indirect Reference Challenge
```
Claim: "I would be more worried about playing a chess game against the
        north Belgian artist that made Prelude to a Broken Arm, than Jiang Wen."

Gold Documents Needed:
  1. Marcel Duchamp (the "north Belgian artist")
  2. Jiang Wen
  3. Prelude to a Broken Arm (the artwork)

Challenge: The claim uses "north Belgian artist" as an indirect reference
to Marcel Duchamp. The query generator must either:
  - Resolve "north Belgian artist" → "Marcel Duchamp" explicitly
  - Generate a query like "north Belgian artist Prelude to a Broken Arm"
  - If the query just says "chess player" it will miss Duchamp entirely
```

### Example 3: Multi-Context Entity Challenge
```
Claim: "Jacob 'Jack' Kevorkian, born in 1977, is best known for publicly
        championing a terminal patient's right to die via physician-assisted
        suicide. Not the Hall of Fame porn star that replaced Juli Ashton
        on Playboy Radio."

Gold Documents Needed:
  1. Christy Canyon (the porn star)
  2. Jack Kevorkian (physician, assisted suicide context)
  3. Jack Kevorkian (birth year context - note: appears twice!)
  4. Playboy Radio (the radio show)

Challenge: This claim actually contains FALSE information (Kevorkian was
born in 1928, not 1977). The gold docs include Jack Kevorkian TWICE,
suggesting different contexts/sections are needed. Retrieving just one
"Jack Kevorkian" document may not be sufficient.
```

## Identified Retrieval Gap Patterns

### Pattern 1: Missing Bridging Entities

**Problem**: Gold documents that connect two entities in the claim are not directly mentioned.

**Examples**:
- "Greek alphabet" connects "Ilysiakos B.C." (Greek name) to "Phoenician alphabet"
- Concept documents that explain relationships

**Why it fails**:
- Entity extraction focuses on explicit entities ("Antonis Fotsis", "Ilysiakos")
- Queries generated from explicit entities don't retrieve conceptual bridges
- Reranking may deprioritize abstract concepts in favor of entity-specific docs

**Architectural insight**: Need explicit "concept extraction" step that identifies:
- Descriptive phrases that imply entities ("derived from", "is a type of")
- Categorical/taxonomic relationships
- Historical/linguistic connections

---

### Pattern 2: Indirect References and Paraphrases

**Problem**: Entities described indirectly are not resolved to canonical names.

**Examples**:
- "north Belgian artist" → Marcel Duchamp
- "conductor for Where Are You My Brothers?" → Constantine Orbelian
- "the club who's name has the starting letter from..." → Ilysiakos B.C.

**Why it fails**:
- Query generation uses the descriptive phrase as-is
- Retrieval system may not find documents if the description doesn't match
- No entity resolution/disambiguation step before query generation

**Architectural insight**: Need two-stage approach:
1. Entity resolution: Identify indirect references and resolve to canonical forms
2. Query augmentation: Include both description and resolved name in queries

---

### Pattern 3: Multi-Aspect Claims Requiring Diverse Documents

**Problem**: Claims that verify multiple independent facts need documents covering different aspects.

**Examples**:
- Comparing two people (requires docs about both)
- Verifying dates, locations, relationships (each needs different evidence)
- Negations ("Not the X") require docs about both X and the contrasted entity

**Why it fails**:
- Parallel queries may focus on the same main entity
- Query diversity strategy may not ensure aspect coverage
- Reranking may overweight documents about the primary entity

**Architectural insight**: Need structured claim decomposition:
- Parse claim into sub-claims or verification requirements
- Generate at least one query per sub-claim
- Ensure query diversity metric (queries shouldn't overlap too much)

---

### Pattern 4: Context-Dependent Entity Documents

**Problem**: Same entity appearing multiple times in gold docs, requiring different contexts.

**Examples**:
- "Jack Kevorkian" appears twice (different facts: birth year, career)
- "Splendor in the Grass" appears twice (different aspects: cast, director)

**Why it fails**:
- Standard deduplication removes multiple mentions
- Single document about entity may not contain all needed facts
- Retrieval system returns one "best" match per entity

**Architectural insight**: Need context-aware retrieval:
- Track which facts have been covered by retrieved documents
- Generate follow-up queries for missing facts about same entity
- Don't deduplicate too aggressively - allow multiple docs per entity if needed

---

### Pattern 5: Obscure or Secondary Entities

**Problem**: Some gold documents are about minor entities that support the main claim.

**Examples**:
- "Playboy Radio" (context for Christy Canyon)
- "Prelude to a Broken Arm" (artwork to identify Duchamp)
- "Where Are You My Brothers?" (work to identify conductor)

**Why it fails**:
- Query generation focuses on primary entities
- Secondary entities may not generate dedicated queries
- Reranking deprioritizes documents about minor entities

**Architectural insight**: Need completeness checking:
- Extract ALL entities from claim, including minor ones
- Ensure at least one query covers each entity
- Balance reranking to maintain diversity, not just relevance to main claim

---

## Architectural Improvement Recommendations

### 1. Iterative Retrieval with Gap Detection

**Current**: One-shot parallel retrieval
**Proposed**: Multi-round retrieval with feedback

```python
def iterative_retrieval(claim, max_rounds=3):
    gold_entities = extract_all_entities(claim)
    retrieved_docs = []

    for round in range(max_rounds):
        # Identify which entities are not yet covered
        covered_entities = extract_entities_from_docs(retrieved_docs)
        missing_entities = gold_entities - covered_entities

        if not missing_entities:
            break

        # Generate focused queries for missing entities
        queries = generate_queries_for_entities(missing_entities)
        new_docs = retrieve_and_rerank(queries)
        retrieved_docs.extend(new_docs)

    return retrieved_docs[:21]
```

**Benefits**:
- Addresses Pattern 1 (bridging entities) by detecting gaps
- Addresses Pattern 4 (context-dependent) by retrieving until all facts covered
- Addresses Pattern 5 (obscure entities) by dedicated follow-up queries

---

### 2. Entity Resolution Before Query Generation

**Current**: Use claims/descriptions as-is in queries
**Proposed**: Resolve indirect references first

```python
class EntityResolver(dspy.Signature):
    """Identify all entities mentioned directly or indirectly in the claim.
    For indirect references, provide the resolved canonical name."""

    claim: str = dspy.InputField()
    entities: list[dict] = dspy.OutputField(
        desc="List of {mention: str, canonical: str, type: str}"
    )

# Example output:
# [
#   {"mention": "north Belgian artist", "canonical": "Marcel Duchamp", "type": "person"},
#   {"mention": "Jiang Wen", "canonical": "Jiang Wen", "type": "person"}
# ]
```

**Benefits**:
- Addresses Pattern 2 (indirect references) directly
- Improves query precision by using canonical names
- Enables better document matching

---

### 3. Claim Decomposition for Query Diversity

**Current**: Generate 2-3 queries based on entity chains
**Proposed**: Decompose claim into verification requirements

```python
class ClaimDecomposer(dspy.Signature):
    """Break down a complex claim into independent sub-claims that need verification.
    Each sub-claim should require different evidence."""

    claim: str = dspy.InputField()
    sub_claims: list[str] = dspy.OutputField(
        desc="2-4 sub-claims that together verify the main claim"
    )
    required_entities: list[list[str]] = dspy.OutputField(
        desc="For each sub-claim, list the entities needed to verify it"
    )

# Example:
# Claim: "I would be more worried about playing chess against the north Belgian
#         artist that made Prelude to a Broken Arm, than Jiang Wen."
#
# sub_claims:
#   1. "The north Belgian artist made Prelude to a Broken Arm"
#   2. "This artist is known for chess"
#   3. "Jiang Wen is less associated with chess"
#
# required_entities:
#   1. ["north Belgian artist", "Marcel Duchamp", "Prelude to a Broken Arm"]
#   2. ["Marcel Duchamp", "chess"]
#   3. ["Jiang Wen", "chess"]
```

**Benefits**:
- Addresses Pattern 3 (multi-aspect claims)
- Ensures query diversity by targeting different verification requirements
- Makes explicit what evidence is needed

---

### 4. Hybrid Ranking (Retrieval Score + Relevance Score)

**Current**: Rerank purely by LLM-based relevance scoring
**Proposed**: Combine retrieval score and relevance score

```python
# Current:
scored_docs.sort(key=lambda x: x[1], reverse=True)  # Only LLM score

# Proposed:
def hybrid_score(doc, retrieval_score, relevance_score, alpha=0.4):
    """Combine retrieval and relevance scores.

    retrieval_score: ColBERT score (0-1), measures query-document similarity
    relevance_score: LLM score (0-100), measures claim verification utility
    alpha: weight for retrieval score (0=only LLM, 1=only retrieval)
    """
    normalized_relevance = relevance_score / 100.0
    return alpha * retrieval_score + (1 - alpha) * normalized_relevance

scored_docs.sort(key=lambda x: x.hybrid_score, reverse=True)
```

**Benefits**:
- Prevents LLM reranking from completely overriding retrieval
- Maintains documents that are query-relevant even if LLM rates them lower
- Addresses Pattern 5 by keeping diverse documents in the final set

---

### 5. Increase Initial Retrieval Budget with Adaptive Pruning

**Current**: k=23 per query, fixed
**Proposed**: Adaptive retrieval based on claim complexity

```python
def adaptive_k(claim):
    """Determine retrieval budget based on claim complexity."""
    num_entities = count_entities(claim)
    num_clauses = count_clauses(claim)
    has_negation = "not" in claim.lower()

    base_k = 25
    k = base_k + (num_entities * 3) + (num_clauses * 2)
    if has_negation:
        k += 10  # Negations require contrasting evidence

    return min(k, 50)  # Cap at 50 to avoid too many docs
```

**Benefits**:
- More coverage for complex claims
- Reduces risk of missing gold docs in initial retrieval
- Balances cost (more docs) with completeness

---

## Summary of Gap Types and Solutions

| Gap Pattern | Root Cause | Architectural Solution |
|-------------|------------|------------------------|
| **1. Missing Bridging Entities** | Queries focus on explicit entities | Iterative retrieval + gap detection |
| **2. Indirect References** | No entity resolution step | Entity resolution before query gen |
| **3. Multi-Aspect Claims** | Insufficient query diversity | Claim decomposition for sub-claims |
| **4. Context-Dependent Entities** | Over-aggressive deduplication | Context-aware retrieval tracking |
| **5. Obscure/Secondary Entities** | Reranking deprioritizes minor entities | Hybrid ranking + diversity preservation |

## Recommended Priority

1. **High Priority**: Iterative retrieval with gap detection (addresses multiple patterns)
2. **High Priority**: Entity resolution for indirect references (high-impact improvement)
3. **Medium Priority**: Hybrid ranking to preserve retrieval diversity
4. **Medium Priority**: Claim decomposition for better query coverage
5. **Low Priority**: Adaptive k (marginal improvement, increases cost)

## Next Steps

To validate these hypotheses and identify which gaps are most prevalent:

1. **Run evaluation on test set** - Capture failed examples with full retrieval traces
2. **Manual analysis** - Categorize each failure by gap pattern
3. **Frequency analysis** - Determine which gap patterns are most common
4. **Targeted experiments** - Implement solutions for top-2 gap patterns
5. **A/B testing** - Compare baseline vs. improved architecture on held-out set

This analysis provides a framework for systematic improvement of the multi-hop retrieval system based on architectural insights from the codebase.
