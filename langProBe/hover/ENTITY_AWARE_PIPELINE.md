# Entity-Aware Gap Analysis Retrieval Pipeline

## Overview

The `HoverEntityAwareMultiHop` class implements a sophisticated entity-aware retrieval pipeline that addresses information gaps through targeted multi-hop retrieval. This system goes beyond traditional retrieval by identifying and filling coverage gaps for important entities in the claim.

## Architecture

### Key Components

#### 1. DSPy Signatures

##### ExtractClaimEntities
```python
class ExtractClaimEntities(dspy.Signature):
    """Extract all named entities from the input claim."""
```
- **Input**: claim (string)
- **Output**: entities (list[str])
- **Purpose**: Identifies all named entities including people, organizations, places, and titles

##### VerifyEntityCoverage
```python
class VerifyEntityCoverage(dspy.Signature):
    """Analyze which entities have zero or minimal coverage."""
```
- **Input**: claim, entities, documents
- **Output**: uncovered_entities (list[str])
- **Purpose**: Performs gap analysis to identify entities lacking sufficient information

##### RankDocumentsByRelevance
```python
class RankDocumentsByRelevance(dspy.Signature):
    """Score documents based on entity coverage and claim alignment."""
```
- **Input**: claim, entities, documents
- **Output**: relevance_scores (list[float])
- **Purpose**: Reranks documents based on entity coverage and relevance

#### 2. Multi-Hop Retrieval Flow

The pipeline implements a strategic 3-hop retrieval process:

```
Hop 1 (k=15): Broad claim-based retrieval
    ↓
Entity Extraction & Gap Analysis
    ↓
Hop 2 (k=10): Targeted retrieval for 1st uncovered entity
    ↓
Hop 3 (k=10): Targeted retrieval for 2nd uncovered entity
    ↓
Document Combination & Deduplication
    ↓
Reranking by Entity Coverage
    ↓
Final 21 Documents
```

### Pipeline Steps

#### Step 1: Entity Extraction
```python
entity_extraction = self.extract_entities(claim=claim)
entities = entity_extraction.entities
```
Extracts all named entities from the claim for tracking.

#### Step 2: Initial Retrieval (Hop 1)
```python
hop1_docs = self.retrieve_15(claim).passages
```
Retrieves 15 documents using the full claim query for broad coverage.

#### Step 3: Gap Analysis
```python
coverage_analysis = self.verify_coverage(
    claim=claim,
    entities=entities,
    documents=hop1_docs
)
uncovered_entities = coverage_analysis.uncovered_entities
```
Identifies which entities lack sufficient coverage in the retrieved documents.

#### Step 4: Targeted Retrieval (Hop 2)
```python
if len(uncovered_entities) > 0:
    entity_1 = uncovered_entities[0]
    hop2_query = self.create_entity_query(claim=claim, entity=entity_1, ...)
    hop2_docs = self.retrieve_10(hop2_query).passages
```
Performs targeted retrieval for the most important uncovered entity.

#### Step 5: Targeted Retrieval (Hop 3)
```python
if len(uncovered_entities) > 1:
    entity_2 = uncovered_entities[1]
    hop3_query = self.create_entity_query(claim=claim, entity=entity_2, ...)
    hop3_docs = self.retrieve_10(hop3_query).passages
```
Performs targeted retrieval for the second most important uncovered entity.

#### Step 6-7: Document Combination & Deduplication
```python
all_docs = hop1_docs + hop2_docs + hop3_docs
# Deduplicate using document hash
```
Combines all retrieved documents (up to 35 total) and removes duplicates.

#### Step 8: Reranking
```python
ranking = self.rank_documents(claim=claim, entities=entities, documents=unique_docs)
relevance_scores = ranking.relevance_scores
# Sort by relevance and select top 21
```
Ranks documents based on:
- Entity coverage (how many claim entities are mentioned)
- Claim alignment (relevance to verifying the claim)
- Information density (quality and depth)

## Advantages

### 1. **Adaptive Retrieval**
The system adapts to information gaps rather than following a fixed retrieval pattern.

### 2. **Entity-Centric**
Ensures comprehensive coverage of all important entities in the claim.

### 3. **Gap Analysis**
Explicitly identifies and addresses information gaps through targeted queries.

### 4. **Quality Reranking**
Final document selection is based on actual relevance rather than retrieval order.

### 5. **Efficient Resource Usage**
- Hop 1: Broad search (k=15)
- Hops 2-3: Targeted searches (k=10 each)
- Total: Up to 35 documents retrieved, refined to 21

## Usage Example

```python
import dspy
from hover_program import HoverEntityAwareMultiHop

# Configure DSPy
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    rm=dspy.ColBERTv2(url="your-retrieval-endpoint")
)

# Initialize pipeline
pipeline = HoverEntityAwareMultiHop()

# Run retrieval
claim = "Marie Curie won Nobel Prizes in Physics and Chemistry."
result = pipeline(claim=claim)

# Access results
print(f"Documents: {len(result.retrieved_docs)}")
print(f"Entities: {result.entities}")
print(f"Gaps: {result.uncovered_entities}")
```

## Output Structure

The pipeline returns a `dspy.Prediction` object with:

```python
{
    'retrieved_docs': list[str],      # Final 21 ranked documents
    'entities': list[str],             # Extracted entities from claim
    'uncovered_entities': list[str]    # Entities with coverage gaps
}
```

## Comparison to Original HoverMultiHop

| Feature | Original | Entity-Aware |
|---------|----------|--------------|
| Initial retrieval | k=7 | k=15 |
| Retrieval strategy | Summary-based | Gap analysis |
| Entity tracking | ❌ | ✅ |
| Gap analysis | ❌ | ✅ |
| Targeted queries | Generic | Entity-specific |
| Final reranking | ❌ | ✅ (by entity coverage) |
| Total documents | 21 (3×7) | 21 (reranked from 35) |

## Performance Considerations

1. **LM Calls**: The entity-aware pipeline makes additional LM calls for:
   - Entity extraction
   - Gap analysis
   - Document reranking

2. **Retrieval Efficiency**: Retrieves more documents initially (35 vs 21) but provides better coverage.

3. **Latency**: Slightly higher due to additional analysis steps, but provides better quality results.

## Future Enhancements

Possible improvements to the pipeline:

1. **Dynamic k-values**: Adjust retrieval counts based on claim complexity
2. **Multi-entity queries**: Combine multiple uncovered entities in single queries
3. **Iterative gap analysis**: Re-analyze gaps after each hop
4. **Entity weighting**: Prioritize entities based on claim importance
5. **Confidence scoring**: Return confidence scores for entity coverage

## Implementation Notes

- The pipeline preserves the original `HoverMultiHop` class for backward compatibility
- Uses DSPy's `ChainOfThought` for all reasoning steps
- Implements simple deduplication using document hashing
- Handles edge cases (no uncovered entities, fewer than 21 unique documents)
