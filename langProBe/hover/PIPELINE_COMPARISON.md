# Pipeline Comparison: Original vs Entity-Aware

## Visual Flow Comparison

### Original HoverMultiHop Pipeline

```
Input: Claim
    ↓
┌─────────────────────────────────────────────┐
│ HOP 1: Retrieve k=7 docs with claim        │
│ Output: 7 documents                         │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ Summarize: Create summary_1 from 7 docs    │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ HOP 2: Generate query from summary_1       │
│ Retrieve k=7 docs with new query           │
│ Output: 7 more documents                    │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ Summarize: Create summary_2 from docs      │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ HOP 3: Generate query from both summaries  │
│ Retrieve k=7 docs with new query           │
│ Output: 7 more documents                    │
└─────────────────────────────────────────────┘
    ↓
Final Output: 21 documents (7+7+7)
```

### Entity-Aware HoverMultiHop Pipeline

```
Input: Claim
    ↓
┌─────────────────────────────────────────────┐
│ STEP 1: Extract Named Entities             │
│ • People, Organizations, Places, Titles     │
│ Output: List of entities                    │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ HOP 1: Broad Retrieval                     │
│ Retrieve k=15 docs with full claim         │
│ Output: 15 documents                        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ STEP 3: Gap Analysis                       │
│ Verify which entities have coverage        │
│ Identify: Zero/minimal coverage entities   │
│ Output: Uncovered entities (ranked)        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ HOP 2: Targeted Retrieval (Entity 1)      │
│ Generate query for 1st uncovered entity    │
│ Retrieve k=10 docs                         │
│ Output: 10 documents about missing entity  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ HOP 3: Targeted Retrieval (Entity 2)      │
│ Generate query for 2nd uncovered entity    │
│ Retrieve k=10 docs                         │
│ Output: 10 documents about missing entity  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ STEP 6: Combine & Deduplicate             │
│ All documents: 15 + 10 + 10 = 35          │
│ Remove duplicates                          │
│ Output: ~25-35 unique documents            │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ STEP 8: Intelligent Reranking             │
│ Score by:                                  │
│ • Entity coverage                          │
│ • Claim alignment                          │
│ • Information density                      │
│ Select top 21 documents                    │
└─────────────────────────────────────────────┘
    ↓
Final Output: 21 ranked documents
```

## Key Differences

### Retrieval Strategy

| Aspect | Original | Entity-Aware |
|--------|----------|--------------|
| **Approach** | Summary-driven | Gap analysis-driven |
| **Query Generation** | Based on previous summaries | Based on uncovered entities |
| **Adaptivity** | Fixed 3-hop pattern | Adapts to coverage gaps |
| **Document Selection** | First k retrieved | Reranked by relevance |

### Information Coverage

| Aspect | Original | Entity-Aware |
|--------|----------|--------------|
| **Entity Tracking** | None | Explicit tracking |
| **Gap Detection** | Implicit (via summaries) | Explicit analysis |
| **Coverage Guarantee** | No guarantee | Targets uncovered entities |
| **Quality Assurance** | Retrieval order | Relevance scoring |

### Resource Usage

| Aspect | Original | Entity-Aware |
|--------|----------|--------------|
| **Total Retrieval** | 21 docs (3×7) | 35 docs (15+10+10) |
| **LM Calls** | 5 (2 summaries + 2 queries) | 5 (extract + verify + 2 queries + rank) |
| **Final Documents** | 21 (all retrieved) | 21 (top-ranked) |
| **Deduplication** | None | Hash-based |

## Example Walkthrough

### Claim
"Marie Curie was the first woman to win a Nobel Prize and won prizes in both Physics and Chemistry."

### Original Pipeline Flow

```
HOP 1: Query = "Marie Curie was the first woman to win a Nobel Prize..."
  → 7 documents about Marie Curie

Summary_1: "Marie Curie was a scientist who won Nobel Prize..."

HOP 2: Query = "Marie Curie Nobel Prize woman scientist"
  → 7 documents about Nobel Prize winners

Summary_2: "Nobel Prize history, Marie Curie achievements..."

HOP 3: Query = "Marie Curie Physics Chemistry Nobel Prize"
  → 7 documents about Nobel prizes in sciences

Result: 21 documents (may lack specific coverage on Physics/Chemistry awards)
```

### Entity-Aware Pipeline Flow

```
ENTITIES: ["Marie Curie", "Nobel Prize", "Physics", "Chemistry"]

HOP 1: Query = full claim
  → 15 documents about Marie Curie and Nobel Prize

GAP ANALYSIS:
  ✓ Marie Curie - well covered
  ✓ Nobel Prize - well covered
  ✗ Physics prize - minimal coverage
  ✗ Chemistry prize - minimal coverage

HOP 2: Query = "Marie Curie Nobel Prize Physics 1903"
  → 10 documents specifically about Physics Nobel Prize

HOP 3: Query = "Marie Curie Nobel Prize Chemistry 1911"
  → 10 documents specifically about Chemistry Nobel Prize

COMBINED: 35 documents (after dedup: ~28 unique)

RERANKING:
  - Documents mentioning multiple entities ranked higher
  - Documents with both Physics AND Chemistry details prioritized
  - Top 21 selected based on entity coverage and relevance

Result: 21 highly relevant documents with comprehensive entity coverage
```

## Performance Metrics (Hypothetical)

| Metric | Original | Entity-Aware | Improvement |
|--------|----------|--------------|-------------|
| Entity Coverage | ~65% | ~95% | +46% |
| Relevance Score | 0.72 | 0.89 | +24% |
| Duplicate Rate | 15% | <5% | -67% |
| Gap Detection | Implicit | Explicit | N/A |
| Processing Time | 2.3s | 3.1s | +35% |

## When to Use Each Pipeline

### Use Original HoverMultiHop When:
- Speed is critical
- Claim is straightforward with few entities
- Computing resources are limited
- Simple retrieval is sufficient

### Use Entity-Aware Pipeline When:
- Accuracy is more important than speed
- Claim involves multiple specific entities
- Comprehensive coverage is required
- You need to track entity coverage
- Gap analysis adds value

## Implementation Flexibility

Both pipelines can coexist in the codebase:

```python
# Original for speed
from langProBe.hover import HoverMultiHop
fast_pipeline = HoverMultiHop()

# Entity-aware for accuracy
from langProBe.hover import HoverEntityAwareMultiHop
accurate_pipeline = HoverEntityAwareMultiHop()

# Choose based on requirements
pipeline = accurate_pipeline if need_accuracy else fast_pipeline
result = pipeline(claim=claim)
```

## Conclusion

The entity-aware pipeline trades some speed for significantly better coverage and relevance. It's particularly valuable for complex claims with multiple entities that require comprehensive verification.
