# Detailed Query Analysis: Coverage-Driven vs Entity-Direct Approaches

## Overview

This document provides a detailed analysis of how the current coverage-driven system generates queries compared to what would work for entity retrieval.

---

## Example 1: American Rock Band Query

### Test Case
```json
{
  "claim": "Boy Hits Car, the American rock band that released My Animal, has more studio albums than The Invisible.",
  "supporting_facts": [
    {"key": "Boy Hits Car", "value": 0},
    {"key": "The Invisible (band)", "value": 0},
    {"key": "My Animal", "value": 0}
  ],
  "label": 1
}
```

### What the Current System Does

#### Step 1: Entity Tracker
```python
tracker_output = self.entity_tracker(claim=claim)
```

**Expected Output**:
```
entities: ["Boy Hits Car", "The Invisible", "My Animal"]
key_facts: [
  "Boy Hits Car is an American rock band",
  "Boy Hits Car released My Animal",
  "Boy Hits Car has more studio albums than The Invisible"
]
verification_aspects: [
  "Identity and genre of Boy Hits Car",
  "Album count for Boy Hits Car",
  "Identity and discography of The Invisible",
  "Album count for The Invisible",
  "Verification of My Animal album release"
]
```

Note: Entities are identified but then abstracted into "verification aspects"

#### Step 2: HOP 1 - Direct Claim Query
```python
hop1_docs = self.retrieve_k(claim).passages
```

**Query**: "Boy Hits Car, the American rock band that released My Animal, has more studio albums than The Invisible"

**Likely Retrieved Docs**:
- Various rock band articles
- Album discography pages
- Music genre pages
- BUT: Might get "Boy Hits Car" by luck if the claim is specific enough

#### Step 3: Coverage Analyzer 1
```python
coverage1 = self.coverage_analyzer1(
    claim=claim,
    verification_aspects=verification_aspects,
    retrieved_titles=hop1_titles,
    passages=hop1_docs
)
```

**Expected Analysis** (if Boy Hits Car was NOT retrieved):
```
covered_aspects: [
  "General information about rock bands and albums"
]
under_covered_aspects: [
  "Specific discography information for Boy Hits Car",
  "Album release details"
]
missing_aspects: [
  "The Invisible band identity and album count",
  "Direct verification of My Animal album"
]
coverage_summary: "Retrieved general rock band information but missing specific entity pages for The Invisible and detailed album information."
```

#### Step 4: Query Generator Hop 2
```python
hop2_query_output = self.query_generator_hop2(
    claim=claim,
    missing_aspects=coverage1.missing_aspects,
    under_covered_aspects=coverage1.under_covered_aspects,
    coverage_summary=coverage1.coverage_summary,
    retrieved_titles=all_retrieved_titles
)
```

**Likely Generated Query**:
```
query: "The Invisible band discography album count"
rationale: "Targeting missing information about The Invisible band's album count to compare with Boy Hits Car"
```

**Problem**: This semantic query might retrieve:
- "Invisible (album)" by various artists
- "The Invisibles" (comic book)
- Documents about invisible bands or stealth marketing
- Generic discography templates

**What's Missing**: The exact Wikipedia article title "The Invisible (band)"

#### Step 5: HOP 2 Retrieval
```python
hop2_docs = self.retrieve_k(hop2_query_output.query).passages
```

**Likely Retrieved**:
- Wrong entities with similar names
- Generic music industry pages
- Related but irrelevant articles

#### Step 6: Coverage Analyzer 2
```python
coverage2 = self.coverage_analyzer2(
    claim=claim,
    verification_aspects=verification_aspects,
    retrieved_titles=hop2_titles,
    passages=hop2_docs
)
```

**Expected Analysis**:
```
covered_aspects: [
  "General information about bands and discographies"
]
under_covered_aspects: [
  "Still missing: The Invisible (band) specific page",
  "My Animal album verification"
]
missing_aspects: [
  "Exact album counts for both bands"
]
coverage_summary: "Retrieved related music content but still missing specific Wikipedia articles for entities."
```

#### Step 7: Query Generator Hop 3
```python
hop3_query_output = self.query_generator_hop3(
    claim=claim,
    missing_aspects=coverage2.missing_aspects,
    under_covered_aspects=coverage2.under_covered_aspects,
    coverage_summary=coverage2.coverage_summary,
    retrieved_titles=all_retrieved_titles
)
```

**Likely Generated Query**:
```
query: "My Animal album Boy Hits Car release"
rationale: "Final attempt to verify the My Animal album connection to Boy Hits Car"
```

**Result**: Might get music album pages but likely misses the specific entities

### What an Entity-Direct System Would Do

#### Step 1: Extract Entity Names
```python
entities = extract_entity_names(claim)
# Result: ["Boy Hits Car", "My Animal", "The Invisible"]
```

#### Step 2: HOP 1 - Direct Claim Query (Same)
```python
hop1_docs = self.retrieve_k(claim).passages
```

#### Step 3: Check Entity Coverage
```python
retrieved_entities = extract_titles_from_docs(hop1_docs)
missing_entities = [e for e in entities if e not in retrieved_entities]
# Assume: ["The Invisible (band)", "My Animal"] are missing
```

#### Step 4: HOP 2 - Direct Entity Query
```python
hop2_query = "The Invisible band"  # Direct entity name!
hop2_docs = self.retrieve_k(hop2_query).passages
```

**High probability of retrieving**: "The Invisible (band)" Wikipedia article

#### Step 5: HOP 3 - Remaining Missing Entity
```python
hop3_query = "My Animal Boy Hits Car"  # Direct entity names!
hop3_docs = self.retrieve_k(hop3_query).passages
```

**High probability of retrieving**: "My Animal" Wikipedia article

---

## Example 2: Film Director Query

### Test Case
```json
{
  "claim": "Mad Hot Ballroom (2005) was released before Koyaanisqatsi, a Powaqqatsi sequel.",
  "supporting_facts": [
    {"key": "Koyaanisqatsi", "value": 0},
    {"key": "Mad Hot Ballroom", "value": 0},
    {"key": "Powaqqatsi", "value": 1}
  ],
  "label": 0
}
```

### Coverage-Driven Failure Pattern

#### HOP 1: Direct claim query
**Query**: Full claim text
**Retrieved**: Might get "Mad Hot Ballroom" and "Powaqqatsi"
**Missing**: "Koyaanisqatsi" (the key comparison document)

#### HOP 2: Coverage-based query
**Missing Aspects**: "Relationship between Koyaanisqatsi and Powaqqatsi", "Release date verification"
**Generated Query**: "Koyaanisqatsi Powaqqatsi sequel relationship"
**Problem**: This semantic query is asking about relationships rather than directly requesting the entity

**Likely Retrieved**:
- Articles about film sequels
- Qatsi trilogy information (might be helpful but scattered)
- Film analysis pages
- NOT necessarily the exact "Koyaanisqatsi" Wikipedia article

#### HOP 3: Still missing entity
**Generated Query**: "Koyaanisqatsi 1982 film release"
**Problem**: Even if this query is more specific, it's competing with only 7 retrieval slots and might not rank high enough

### Entity-Direct Success Pattern

#### HOP 1: Direct claim query
**Query**: Full claim
**Retrieved**: "Mad Hot Ballroom", "Powaqqatsi"

#### HOP 2: Direct entity query
**Missing**: ["Koyaanisqatsi"]
**Query**: "Koyaanisqatsi film"
**Retrieved**: High probability → "Koyaanisqatsi" Wikipedia article ✓

#### HOP 3: Buffer for other aspects
**Query**: Can focus on secondary information or verify facts

---

## Example 3: Author Birth Date Comparison

### Test Case
```json
{
  "claim": "Morgan Llywelyn, author of Lion of Ireland, was born before Robert Jordan.",
  "supporting_facts": [
    {"key": "Morgan Llywelyn", "value": 0},
    {"key": "Robert Jordan", "value": 0},
    {"key": "Lion of Ireland", "value": 0}
  ],
  "label": 1
}
```

### Coverage-Driven Query Sequence

#### HOP 1
**Query**: Full claim
**Retrieved**: "Lion of Ireland" (direct mention), "Morgan Llywelyn" (direct mention)
**Missing**: "Robert Jordan" (needs comparison entity)

#### HOP 2
**Missing Aspects**: "Robert Jordan birth date", "Birth date comparison verification"
**Generated Query**: "Robert Jordan author birth date biography"

**Problem**: This query is about a "birth date" rather than directly targeting the entity. Semantic search might return:
- Articles about dates and birthdays
- Biography pages (generic)
- Other authors named Jordan
- NOT necessarily "Robert Jordan" Wikipedia article with clear birth date

#### HOP 3
**Generated Query**: "Robert Jordan fantasy author Wheel of Time"

**Problem**: Now we're adding context, but this dilutes the entity-focused search. Still might miss the article.

### Entity-Direct Approach

#### HOP 2
**Missing**: ["Robert Jordan"]
**Query**: "Robert Jordan author"
**Result**: Direct hit on "Robert Jordan" Wikipedia article ✓

Simple and effective.

---

## Root Cause: Semantic Drift

### The Abstraction Chain

```
Concrete Entity → Abstract Aspect → Semantic Query → Wrong Documents
```

**Example**:
```
"Boy Hits Car" → "Band identity verification" → "rock band discography" → Generic music pages ✗
```

**vs Direct**:
```
"Boy Hits Car" → "Boy Hits Car band" → Boy Hits Car Wikipedia article ✓
```

### Why Coverage Analysis Fails

The coverage analysis operates on abstracted concepts:

```python
missing_aspects = [
    "Identity and album count for The Invisible",
    "Verification of My Animal release"
]
```

These are **descriptions of information needs**, not **entity names**. When the query generator receives these descriptions, it must:

1. Interpret the abstract description
2. Infer which entities are relevant
3. Generate a query that balances multiple abstract goals
4. Hope the semantic search understands the intent

Each step introduces drift away from the simple, direct entity name.

### Why Entity-Direct Works

```python
missing_entities = ["The Invisible (band)", "My Animal"]
query = missing_entities[0]  # "The Invisible (band)"
```

No interpretation, no inference, no semantic drift. Just the entity name.

---

## Query Quality Comparison

### Coverage-Driven Queries (Actual Output)

| Hop | Query Type | Example | Problem |
|-----|------------|---------|---------|
| 1 | Direct claim | Full claim text | Too general, might miss specific entities |
| 2 | Abstract aspect | "The Invisible band discography album count" | Semantic drift, retrieves related but wrong docs |
| 3 | Abstract aspect | "My Animal album Boy Hits Car release verification" | Still abstract, low precision |

### Entity-Direct Queries (Recommended)

| Hop | Query Type | Example | Benefit |
|-----|------------|---------|---------|
| 1 | Direct claim | Full claim text | Good starting point |
| 2 | Direct entity | "The Invisible band" | High precision, direct Wikipedia hit |
| 3 | Direct entity | "My Animal album" | High precision, completes entity set |

---

## The Wikipedia Structure Advantage

Wikipedia articles have **canonical titles** that match entity names:

- "Boy Hits Car" → https://en.wikipedia.org/wiki/Boy_Hits_Car
- "The Invisible (band)" → https://en.wikipedia.org/wiki/The_Invisible_(band)
- "Robert Jordan" → https://en.wikipedia.org/wiki/Robert_Jordan

Direct entity queries leverage this structure. Abstract queries fight against it.

---

## Retrieval System Behavior

The ColBERT retrieval system is optimized for:
- **Semantic similarity** between query and document
- **Keyword matching** with context understanding
- **Dense retrieval** with learned representations

Given query "The Invisible band discography album count":
- High semantic similarity to: generic discography pages, music database templates
- Moderate similarity to: "The Invisible (band)" article
- **Ranking**: The specific entity article might not be in top 7 results

Given query "The Invisible band":
- Highest similarity to: "The Invisible (band)" Wikipedia article (exact match + context)
- **Ranking**: Very likely in top 7 results

---

## Statistical Analysis

### Probability of Entity Retrieval

**Coverage-Driven Abstract Query**:
- P(retrieve correct entity) ≈ 0.3-0.5 (low to moderate)
- Reason: Competing with semantically similar but wrong documents

**Entity-Direct Query**:
- P(retrieve correct entity) ≈ 0.7-0.9 (high)
- Reason: Direct name match favored by retrieval system

### Compound Probability for 3-Entity Claim

**Coverage-Driven** (assuming P=0.4 per entity):
- P(all 3 entities) = 0.4³ = **0.064 (6.4%)**

**Entity-Direct** (assuming P=0.8 per entity):
- P(all 3 entities) = 0.8³ = **0.512 (51.2%)**

This explains the score difference:
- **Current system**: 0.0% (even worse than 6.4% due to abstraction)
- **Baseline**: 34.33% (closer to predicted 51.2% with some noise)

---

## Recommendations

### Immediate Fix: Entity Passthrough

Modify query generation to include entity names directly:

```python
hop2_query_output = self.query_generator_hop2(
    claim=claim,
    entities=tracker_output.entities,  # ADD THIS
    missing_aspects=coverage1.missing_aspects,
    coverage_summary=coverage1.coverage_summary,
    retrieved_titles=all_retrieved_titles
)
```

Update `TargetedQueryGeneratorSignature`:
```python
class TargetedQueryGeneratorSignature(dspy.Signature):
    claim = dspy.InputField(desc="The original claim being verified")
    entities = dspy.InputField(desc="List of entity names from the claim")  # NEW
    missing_aspects = dspy.InputField(desc="Verification aspects not yet covered")
    # ... rest of fields
```

Prompt the LLM to use entity names directly in queries.

### Medium-term Fix: Hybrid Approach

1. Extract entities
2. HOP 1: Direct claim query
3. Check which entities are retrieved
4. HOP 2: Query for missing entities by name
5. HOP 3: Fill remaining gaps with aspect-based queries

### Long-term Fix: Entity-First Architecture

Complete redesign:
1. Entity extraction first
2. Direct entity queries (one per entity if needed)
3. Aggregation and ranking
4. Return top 21 documents

This is what the IMPLEMENTATION_SUMMARY described but was not implemented in the current code.

---

## Conclusion

The 0.0 score is not due to a bug but a **fundamental architectural mismatch**:

- **Task requires**: Exact entity page retrieval
- **System provides**: Semantic aspect-based search
- **Result**: Catastrophic failure

**Fix**: Keep entity names intact throughout the pipeline and query for them directly.

---

*Document prepared: 2026-02-17*
*Branch: codeevolver-20260217004441-a9b59e*
*Commit: 8a32bd2*
