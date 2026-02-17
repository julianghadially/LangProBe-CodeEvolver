# HoverMultiHop Refactoring Summary

## Overview
Successfully refactored the HoverMultiHop query generation logic in `langProBe/hover/hover_program.py` with an entity-extraction-first approach that ensures comprehensive entity coverage within the 3-search and 21-document constraints.

## Key Changes

### 1. **New DSPy Module: ClaimEntityExtractor**
- **Purpose**: Explicitly identifies all named entities, relationships, and key facts from claims before retrieval
- **Outputs**:
  - `primary_entities`: Core entities central to the claim
  - `secondary_entities`: Supporting entities that provide context
  - `relationships`: Key relationships between entities that need verification
  - `key_facts`: Specific facts, dates, or events requiring verification
- **Implementation**: Uses DSPy Signature with ChainOfThought reasoning

### 2. **New DSPy Module: EntityBasedQueryGenerator**
- **Purpose**: Generates targeted search queries for specific entity clusters
- **Inputs**:
  - Original claim
  - Entity cluster to focus on
  - Relationships to explore
  - Previous findings (for context in later hops)
- **Output**: Optimized search query for the given entity cluster

### 3. **New DSPy Module: DocumentRelevanceScorer**
- **Purpose**: Scores documents based on entity coverage and relevance
- **Scoring Criteria**:
  - Number of target entities mentioned
  - Whether key relationships are discussed
  - Whether key facts are verified or contradicted
- **Outputs**:
  - `relevance_score`: 0-10 score indicating document relevance
  - `covered_entities`: Which entities are covered in the document

### 4. **Enhanced Retrieval Strategy**
The new approach uses a three-hop retrieval with specialized k values:

#### **Hop 1: Primary Entities (k=30)**
- Extracts and retrieves documents for core entities
- Focuses on people, places, organizations most directly mentioned in claim
- Largest retrieval to ensure comprehensive coverage of main subjects

#### **Hop 2: Secondary/Bridging Entities (k=20)**
- Targets supporting entities that provide context
- Focuses on entities that connect primary entities
- Builds on findings from Hop 1

#### **Hop 3: Relationship Verification (k=15)**
- Verifies relationships between entities
- Confirms key facts and dates
- Uses context from both previous hops
- Focuses on inter-entity connections

**Total Documents Retrieved**: 30 + 20 + 15 = 65 documents

### 5. **Relevance-Based Reranking System**
After retrieving 65 documents, the system:

1. **Scores Each Document**: Uses DocumentRelevanceScorer to evaluate entity coverage
2. **Sorts by Relevance**: Orders documents by relevance score (0-10)
3. **Selects Top 21**: Greedy selection to maximize entity coverage
4. **Tracks Coverage**: Maintains set of covered entities to ensure diversity

## Algorithm Flow

```
1. Entity Extraction Phase
   └─> Extract primary entities, secondary entities, relationships, key facts

2. Multi-Hop Retrieval Phase
   ├─> Hop 1: Generate query for primary entities → Retrieve 30 docs
   ├─> Hop 2: Generate query for secondary entities → Retrieve 20 docs
   └─> Hop 3: Generate query for relationships → Retrieve 15 docs

3. Reranking Phase
   ├─> Score all 65 documents for entity coverage
   ├─> Sort by relevance score
   └─> Select top 21 documents with maximum entity coverage

4. Return Final Results
   └─> 21 documents optimized for comprehensive entity coverage
```

## Benefits of This Approach

### 1. **Explicit Entity Awareness**
- System now explicitly identifies what needs to be verified before searching
- Ensures no critical entities are missed in retrieval

### 2. **Structured Entity Clustering**
- Groups related entities together (primary, secondary, bridging)
- Allows targeted queries for each entity type

### 3. **Progressive Information Gathering**
- Each hop builds on previous findings
- Avoids redundant retrieval while ensuring completeness

### 4. **Optimized Document Selection**
- Reranking ensures final 21 documents maximize entity coverage
- Balances between high-quality scoring and entity diversity

### 5. **Scalable Retrieval Strategy**
- Larger k values for broader searches (primary entities)
- Smaller k values for focused searches (relationships)
- Efficiently uses the 3-search constraint

## Technical Implementation Details

### Entity Extraction
```python
self.entity_extractor = dspy.ChainOfThought(ClaimEntityExtractor)
extraction = self.entity_extractor(claim=claim)
```

### Query Generation Per Hop
```python
self.query_generator = dspy.ChainOfThought(EntityBasedQueryGenerator)
query = self.query_generator(
    claim=claim,
    entity_cluster=entity_cluster,
    relationships=relationships,
    previous_findings=previous_summary
).query
```

### Document Scoring & Reranking
```python
self.doc_scorer = dspy.ChainOfThought(DocumentRelevanceScorer)
scoring = self.doc_scorer(
    claim=claim,
    document=doc[:500],
    target_entities=all_entities,
    target_relationships=relationships
)
```

## Compliance with Requirements

✅ **Entity-extraction-first approach**: ClaimEntityExtractor explicitly identifies entities before retrieval

✅ **Named entities, relationships, and key facts**: All three explicitly extracted as separate fields

✅ **Entity clustering**: Primary and secondary entity groups implemented

✅ **Targeted queries per cluster**: EntityBasedQueryGenerator creates focused queries

✅ **Three-search strategy**: k=30, k=20, k=15 for the three hops

✅ **Relevance-based reranker**: DocumentRelevanceScorer evaluates all documents

✅ **Final 21 documents**: Reranking selects top 21 with maximum entity coverage

✅ **Comprehensive entity coverage**: Greedy selection algorithm ensures diverse coverage

## Backward Compatibility

The refactored `HoverMultiHop` class:
- Maintains the same interface: `forward(claim)` → `dspy.Prediction(retrieved_docs=...)`
- Returns exactly the same output structure
- Integrates seamlessly with existing `HoverMultiHopPipeline`
- Works with existing evaluation metrics in `hover_utils.py`

## Testing Recommendations

1. **Unit Tests**: Test each new DSPy module independently
2. **Integration Tests**: Verify end-to-end retrieval pipeline
3. **Coverage Tests**: Ensure all entities in test claims are covered
4. **Performance Tests**: Measure retrieval quality vs. original implementation
5. **Edge Cases**: Test with claims having varying entity counts

## Future Enhancements

Potential improvements for future iterations:
- Add entity type classification (person, organization, location, etc.)
- Implement more sophisticated entity clustering algorithms
- Add adaptive k values based on entity count
- Implement parallel document scoring for faster reranking
- Add entity coverage metrics to track which entities are found
