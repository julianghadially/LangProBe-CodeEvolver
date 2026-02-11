# Adaptive K-Value Allocation Implementation

## Overview

This document describes the adaptive k-value allocation strategy implemented in `HoverMultiHopPredict` for dynamic document budget distribution across multi-hop retrieval.

## Implementation Details

### 1. Complexity Analysis Module

**Added: `ClaimComplexitySignature`**
- A DSPy signature that uses Chain-of-Thought reasoning to analyze claim complexity
- **Input**: The claim to be verified
- **Outputs**:
  - `num_entities` (int, 1-5): Number of distinct entities/topics requiring verification
  - `complexity_score` (int, 1-5): Overall verification difficulty score

**Module**: `self.analyze_complexity = dspy.ChainOfThought(ClaimComplexitySignature)`

The signature includes detailed instructions that guide the LLM to consider:
- Number of distinct entities (people, places, organizations, events)
- Relationships between multiple entities
- Temporal or causal relationships requiring evidence from different sources

### 2. K-Value Allocation Strategy

**Added: `_allocate_k_values()` method**

This method implements the dynamic budget allocation based on claim complexity:

```python
def _allocate_k_values(self, num_entities: int) -> tuple[int, int, int]:
    """Dynamically allocate k values across 3 hops based on claim complexity."""
    num_entities = max(1, min(5, num_entities))  # Clamp to 1-5 range

    if num_entities <= 2:
        return (10, 8, 3)  # Simple claims
    elif num_entities == 3:
        return (7, 7, 7)   # Moderate claims
    else:  # num_entities >= 4
        return (5, 8, 8)   # Complex claims
```

**Allocation Strategies**:

| Complexity | num_entities | k-values | Total | Rationale |
|------------|--------------|----------|-------|-----------|
| **Simple** | 1-2 | [10, 8, 3] | 21 | Focus deeply on the primary entity in early hops |
| **Moderate** | 3 | [7, 7, 7] | 21 | Balanced exploration across all hops |
| **Complex** | 4+ | [5, 8, 8] | 21 | Cast a wider initial net, then deepen in later hops |

### 3. Modified Forward Method

The `forward()` method now:

1. **Analyzes complexity** before retrieval:
   ```python
   complexity_analysis = self.analyze_complexity(claim=claim)
   num_entities = complexity_analysis.num_entities
   ```

2. **Allocates k-values dynamically**:
   ```python
   k1, k2, k3 = self._allocate_k_values(num_entities)
   ```

3. **Creates hop-specific retrievers**:
   ```python
   # HOP 1
   retrieve_hop1 = dspy.Retrieve(k=k1)
   hop1_docs = retrieve_hop1(claim).passages

   # HOP 2
   retrieve_hop2 = dspy.Retrieve(k=k2)
   hop2_docs = retrieve_hop2(hop2_query).passages

   # HOP 3
   retrieve_hop3 = dspy.Retrieve(k=k3)
   hop3_docs = retrieve_hop3(hop3_query).passages
   ```

## Key Changes from Original Implementation

### Before
- Static `k=7` for all three hops
- Single shared retriever instance: `self.retrieve_k = dspy.Retrieve(k=self.k)`
- No complexity analysis
- Total documents: 21 (7+7+7)

### After
- Dynamic k-values based on claim complexity
- Hop-specific retriever instances created in `forward()`
- Complexity analysis using Chain-of-Thought reasoning
- Total documents: Still 21 but distributed adaptively (10+8+3, 7+7+7, or 5+8+8)

## Benefits

1. **Simple Claims**: Deep focus on the main entity early prevents information dilution
2. **Moderate Claims**: Balanced approach ensures all entities get adequate coverage
3. **Complex Claims**: Broad initial search captures diverse entities, with deeper exploration in later hops
4. **Budget Consistency**: Always uses exactly 21 documents total
5. **Automatic Adaptation**: No manual tuning needed per claim

## Example Usage

```python
from langProBe.hover.hover_program import HoverMultiHopPredict
import dspy

# Configure DSPy with your LM and retriever
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    rm=your_retriever
)

# Create the program
program = HoverMultiHopPredict()

# Run on different complexity claims
simple_claim = "Paris is the capital of France."
# Expected: k=[10, 8, 3] - deep dive on Paris/France

moderate_claim = "The Eiffel Tower was built for the 1889 World's Fair in Paris."
# Expected: k=[7, 7, 7] - balanced coverage of tower, fair, and Paris

complex_claim = "Marie Curie, born in Warsaw, won Nobel Prizes in both Physics and Chemistry while working in Paris."
# Expected: k=[5, 8, 8] - broad search for Curie, Warsaw, Nobel, Physics, Chemistry, Paris

result = program(claim=simple_claim)
print(f"Retrieved {len(result.retrieved_docs)} documents")
```

## Testing

Run the test script to verify the allocation strategy:

```bash
python test_adaptive_k_values.py
```

This will show the k-value distributions for all complexity levels (1-5 entities) and verify that each sums to exactly 21 documents.

## Future Enhancements

Potential improvements to consider:

1. **Fine-grained allocation**: More complexity levels (e.g., 1-10 scale)
2. **Custom strategies**: Allow passing custom allocation functions
3. **Adaptive feedback**: Learn from retrieval results to adjust mid-flight
4. **Confidence-based allocation**: Factor in the model's confidence about entity count
5. **Performance metrics**: Track which strategies work best for different claim types
