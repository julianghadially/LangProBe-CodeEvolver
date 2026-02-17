# HoverMultiHop Parallel Diversified Retrieval Strategy

## Overview
This document describes the changes made to replace the summarization-based multi-hop retrieval in `HoverMultiHop` with a parallel diversified retrieval strategy.

## Problem with Original Approach
The original implementation suffered from an **information bottleneck** caused by:
1. **Sequential dependencies**: Each hop depended on summaries from previous hops
2. **Summarization loss**: LLM-based summaries could lose critical information needed for subsequent hops
3. **Limited retrieval**: Only k=7 documents per hop (21 total)
4. **LLM overhead**: Required 2 additional LLM calls for summarization

## New Approach: Parallel Diversified Retrieval

### Key Changes

#### 1. Increased Retrieval per Hop
- **Before**: k=7 documents per hop
- **After**: k=21 documents per hop

#### 2. Removed Summarization Bottleneck
- **Removed modules**:
  - `self.create_query_hop2` (ChainOfThought for query generation)
  - `self.create_query_hop3` (ChainOfThought for query generation)
  - `self.summarize1` (ChainOfThought for summarization)
  - `self.summarize2` (ChainOfThought for summarization)

- **Benefits**:
  - No information loss from summarization
  - Faster execution (no extra LLM calls)
  - More reliable retrieval

#### 3. Three Complementary Search Strategies

**Hop 1: Direct Claim Retrieval (k=21)**
```python
hop1_docs = self.retrieve_k(claim).passages
```
Retrieves documents directly related to the claim.

**Hop 2: Related Entities Retrieval (k=21)**
```python
hop2_query = f"related entities, people, or works mentioned in: {claim}"
hop2_docs = self.retrieve_k(hop2_query).passages
```
Retrieves documents about entities, people, or works mentioned in the claim.

**Hop 3: Background Context Retrieval (k=21)**
```python
hop3_query = f"background information and context about: {claim}"
hop3_docs = self.retrieve_k(hop3_query).passages
```
Retrieves background information and contextual documents.

#### 4. Diversity-Based Reranking

After retrieving 63 documents (21 from each hop), applies **Maximal Marginal Relevance (MMR)** to select the top 21 most diverse documents.

**Algorithm**:
1. Remove exact duplicates (case-insensitive)
2. Compute TF-IDF vectors for all documents and the claim
3. Calculate relevance scores (cosine similarity to claim)
4. Calculate pairwise document similarity
5. Iteratively select documents using MMR:
   - Start with most relevant document
   - For each subsequent selection, choose the document that maximizes:
     ```
     MMR = λ × relevance - (1-λ) × max_similarity_to_selected
     ```
   - Uses λ=0.5 for balanced diversity and relevance

**Benefits**:
- Prioritizes documents covering different entities/topics
- Maintains relevance to the original claim
- Eliminates redundant information
- Better coverage of multi-hop reasoning paths

## Constraints Maintained

✓ **Maximum 3 searches**: Exactly 3 retrieval operations (one per hop)
✓ **Maximum 21 documents output**: Diversity reranking ensures exactly 21 documents
✓ **Compatible with evaluation**: Output format unchanged (`dspy.Prediction(retrieved_docs=...)`)

## Performance Characteristics

### Advantages
1. **No information loss**: Avoids summarization bottleneck
2. **Parallel reasoning**: All hops can conceptually run in parallel
3. **Better coverage**: 3x more documents retrieved (63 vs 21)
4. **Smarter selection**: MMR ensures diversity while maintaining relevance
5. **Faster**: No extra LLM calls for summarization/query generation
6. **More reliable**: Deterministic query construction (no LLM variability)

### Computational Cost
- **Retrieval**: Same (3 retrieval operations)
- **LLM calls**: Reduced (removed 2 summarization + 2 query generation calls)
- **Post-processing**: Added TF-IDF + MMR computation (O(n²) for 63 docs, ~milliseconds)

## Dependencies Added
- `scikit-learn>=1.3.0` (for TfidfVectorizer and cosine_similarity)

## Testing
Run `python test_hover_multihop.py` to verify:
- Diversity reranking works correctly
- Deduplication functions properly
- Edge cases are handled
- Structure matches expected format

## Example Usage
```python
import dspy
from langProBe.hover.hover_program import HoverMultiHop

# Configure retriever
dspy.configure(rm=dspy.ColBERTv2(url="..."))

# Initialize and run
program = HoverMultiHop()
result = program(claim="The Eiffel Tower is located in France")

# Returns 21 diverse, relevant documents
print(f"Retrieved {len(result.retrieved_docs)} documents")
```

## Code Changes Summary

### Before (39 lines)
- Used k=7
- Had 4 ChainOfThought modules (2 for queries, 2 for summaries)
- Sequential hop execution with summarization
- Direct concatenation of retrieved docs

### After (132 lines)
- Uses k=21
- No ChainOfThought modules
- Parallel hop execution with fixed query patterns
- MMR-based diversity reranking
- Comprehensive documentation and error handling

## Migration Notes
- No API changes to the `forward()` method signature
- Drop-in replacement for existing code
- Maintains compatibility with HoverMultiHopPipeline
- Output format unchanged (list of document strings)
