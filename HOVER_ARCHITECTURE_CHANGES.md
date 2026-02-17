# HoverMultiHop Architecture Changes

## Overview
Replaced the summarization-based multi-hop architecture with a query decomposition and parallel retrieval strategy with score-based document reranking.

## Changes Made

### 1. New Module: ClaimQueryDecomposer
**File:** `langProBe/hover/hover_program.py`

**Signature:** `claim -> query1, query2, query3`

**Purpose:** Generates 3 diverse search queries that focus on different entities/aspects mentioned in the claim.

**Example:**
- **Claim:** "The director of Film X was born in Country Y"
- **Query 1:** "Film X director" (focuses on film and director relationship)
- **Query 2:** "Film X" (focuses on the film itself)
- **Query 3:** "Country Y birthplace" (focuses on birthplace information)

**Implementation:**
```python
class ClaimQueryDecomposerSignature(dspy.Signature):
    """Generate 3 diverse search queries that focus on different entities and aspects mentioned in the claim."""
    claim = dspy.InputField(desc="The claim to decompose into search queries")
    query1 = dspy.OutputField(desc="First search query focusing on a specific entity or aspect")
    query2 = dspy.OutputField(desc="Second search query focusing on a different entity or aspect")
    query3 = dspy.OutputField(desc="Third search query focusing on yet another entity or aspect")

class ClaimQueryDecomposer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.decompose = dspy.ChainOfThought(ClaimQueryDecomposerSignature)
```

### 2. New Module: DocumentRelevanceScorer
**File:** `langProBe/hover/hover_program.py`

**Signature:** `claim, document -> relevance_score, reasoning`

**Purpose:** Scores each document for relevance to the claim with a score between 0.0 and 1.0, along with reasoning.

**Implementation:**
```python
class DocumentRelevanceScorerSignature(dspy.Signature):
    """Score how relevant a document is to answering or verifying the given claim."""
    claim = dspy.InputField(desc="The claim to verify or answer")
    document = dspy.InputField(desc="The document to score for relevance")
    relevance_score = dspy.OutputField(desc="Relevance score between 0.0 and 1.0")
    reasoning = dspy.OutputField(desc="Explanation of why this score was assigned")

class DocumentRelevanceScorer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.score = dspy.ChainOfThought(DocumentRelevanceScorerSignature)
```

### 3. Refactored: HoverMultiHop.forward()
**File:** `langProBe/hover/hover_program.py`

**New Architecture:**

1. **Query Decomposition:** Generate 3 diverse queries from the claim
2. **Parallel Retrieval:** Retrieve k=15 documents per query (total 45 documents)
3. **Document Scoring:** Score all 45 documents for relevance
4. **Reranking:** Sort by relevance score (descending)
5. **Selection:** Return top 21 documents

**Key Parameters:**
- `k = 15` (documents per query)
- Total documents retrieved: 45 (3 queries × 15 docs)
- Final documents returned: 21 (top-ranked)

## Old vs New Architecture

### Old Architecture (Summarization-based Multi-hop)
```
Claim → [Retrieve 7 docs] → Summarize
         ↓
         Summary 1 → [Generate Query 2] → [Retrieve 7 docs] → Summarize
                                            ↓
                                            Summary 2 → [Generate Query 3] → [Retrieve 7 docs]
                                                                              ↓
                                                                              Return 21 docs
```

**Problems:**
- Sequential dependency: Later hops depend on earlier summarizations
- Information loss: Summarization can lose critical entities/details
- Limited coverage: Later queries miss entities lost in summarization
- No quality filtering: All retrieved documents are returned

### New Architecture (Query Decomposition + Parallel Retrieval)
```
Claim → [Decompose into 3 queries]
         ↓
         [Query 1] → [Retrieve 15 docs] ─┐
         [Query 2] → [Retrieve 15 docs] ─┼→ [45 docs total] → [Score all docs] → [Rerank by score] → [Return top 21]
         [Query 3] → [Retrieve 15 docs] ─┘
```

**Benefits:**
- ✅ **No sequential dependency:** All queries run in parallel
- ✅ **No information loss:** No summarization step
- ✅ **Comprehensive coverage:** All entities targeted simultaneously
- ✅ **Quality filtering:** Score-based reranking ensures best 21 documents
- ✅ **Entity targeting:** Diverse queries ensure all claim aspects covered

## Performance Characteristics

### Latency
- **Old:** Sequential (3 hops) - Higher latency due to dependency chain
- **New:** Parallel (3 queries) - Lower latency, queries can run concurrently

### Coverage
- **Old:** Progressive narrowing - May miss entities lost in summarization
- **New:** Comprehensive - All entities targeted from the start

### Quality
- **Old:** No explicit quality filtering - Returns all 21 retrieved docs
- **New:** Explicit quality filtering - Scores and reranks 45 docs, returns best 21

### LLM Calls
- **Old:** ~5 calls (2 summarizations + 2 query generations)
- **New:** ~46 calls (1 decomposition + 45 scorings)
  - Note: Scoring can be optimized with batching or caching in future iterations

## Testing

A test script (`test_hover_new_architecture.py`) verifies:
- ✅ ClaimQueryDecomposer generates 3 queries
- ✅ Parallel retrieval gets 15 docs per query (45 total)
- ✅ DocumentRelevanceScorer scores each document
- ✅ Reranking selects top 21 documents
- ✅ Final output contains exactly 21 documents

## Backward Compatibility

The new implementation maintains the same interface:
- **Input:** `claim` (string)
- **Output:** `dspy.Prediction(retrieved_docs=...)` (list of 21 documents)

This ensures compatibility with existing evaluation scripts (`hover_utils.py`, `hover_pipeline.py`).

## Future Optimizations

1. **Batch Scoring:** Score multiple documents in parallel to reduce latency
2. **Caching:** Cache query decompositions for similar claims
3. **Dynamic k:** Adjust k based on query diversity or claim complexity
4. **Deduplication:** Remove duplicate documents before scoring
5. **Early Stopping:** Stop scoring if top 21 documents are clearly identified

## Files Modified

1. `langProBe/hover/hover_program.py` - Complete rewrite of HoverMultiHop class

## Files Created

1. `test_hover_new_architecture.py` - Test script for new architecture
2. `HOVER_ARCHITECTURE_CHANGES.md` - This documentation file
