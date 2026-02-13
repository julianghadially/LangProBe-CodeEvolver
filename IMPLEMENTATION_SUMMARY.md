# Chain-of-Thought Query Planning Implementation Summary

## Overview

This document summarizes the implementation of a chain-of-thought query planning module for the HoverMultiHopPredict system in `hover_program.py`.

## What Was Changed

### File Modified
- **`/workspace/langProBe/hover/hover_program.py`**

### New Components Added

#### 1. ChainOfThoughtQueryPlanner Signature
A structured DSPy signature that enforces explicit reasoning at each retrieval hop:

```python
class ChainOfThoughtQueryPlanner(dspy.Signature):
    """Analyze the claim and retrieved context to strategically plan the next retrieval query."""

    # Inputs
    claim = dspy.InputField(desc="The claim that needs to be verified")
    retrieved_context = dspy.InputField(desc="Context from previous hops")

    # Outputs
    reasoning = dspy.OutputField(desc="Multi-hop reasoning chain explanation")
    missing_information = dspy.OutputField(desc="Gap analysis")
    next_query = dspy.OutputField(desc="Focused search query")
```

#### 2. Refactored HoverMultiHopPredict Module
Replaced simple `dspy.Predict` modules with `dspy.ChainOfThought` modules:

**Before:**
```python
self.extract_key_terms = dspy.Predict("claim->key_terms")
self.create_query_hop2 = dspy.Predict("claim,key_terms,hop1_titles->query")
self.create_query_hop3 = dspy.Predict("claim,key_terms,hop1_titles,hop2_titles->query")
```

**After:**
```python
self.query_planner_hop1 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
self.query_planner_hop2 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
self.query_planner_hop3 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
```

#### 3. Enhanced Forward Pass
Each hop now:
1. Receives full retrieved context from previous hops
2. Performs explicit reasoning about multi-hop chain
3. Analyzes what information is found vs. missing
4. Generates targeted query for specific gaps
5. Preserves full document context (not just titles)

## Key Improvements

### 1. Explicit Multi-Hop Reasoning
- **reasoning** field: Forces decomposition of entities and relationships
- Analyzes how pieces connect across hops
- Plans overall verification strategy

### 2. Gap Analysis
- **missing_information** field: Identifies what's found vs. what's needed
- Prevents redundant retrievals
- Guides next query generation

### 3. Targeted Query Generation
- **next_query** field: Focuses on specific missing information
- More efficient retrievals
- Better coverage of required facts

### 4. Full Context Preservation
- Stores complete document text (not just titles)
- No information loss between hops
- Better reasoning about cumulative evidence

### 5. Chain-of-Thought Reasoning
- Uses `dspy.ChainOfThought` instead of `dspy.Predict`
- Encourages explicit reasoning steps
- Better explainability and debugging

## Files Created

### 1. Test Script
**`/workspace/test_cot_query_planner.py`**
- Demonstrates the new signature structure
- Shows example reasoning at each hop
- Compares old vs. new approach

### 2. Documentation Files
- **`HOVER_COT_QUERY_PLANNER_README.md`**: Comprehensive documentation
- **`HOVER_COMPARISON.md`**: Side-by-side comparison of old vs. new
- **`HOVER_FLOW_DIAGRAM.md`**: Visual flow diagrams and examples
- **`IMPLEMENTATION_SUMMARY.md`**: This file

## Backward Compatibility

✅ **The implementation is fully backward compatible**

External API remains unchanged:
```python
from langProBe.hover.hover_program import HoverMultiHopPredict

program = HoverMultiHopPredict()
result = program(claim="Your claim here")
retrieved_docs = result.retrieved_docs
```

The improvements are internal to the query planning process.

## Example: Multi-Hop Reasoning Flow

### Claim
"The director of the 2017 film The Shape of Water also directed a 2006 film about the Spanish Civil War."

### Hop 1
- **Reasoning**: "Need to identify director of The Shape of Water first"
- **Missing**: "Director's identity is unknown"
- **Query**: "The Shape of Water 2017 director"
- **Found**: Guillermo del Toro

### Hop 2
- **Reasoning**: "Now know del Toro is director, need his 2006 film"
- **Missing**: "2006 film title and Spanish Civil War connection"
- **Query**: "Guillermo del Toro 2006 film Spanish Civil War"
- **Found**: Pan's Labyrinth (2006)

### Hop 3
- **Reasoning**: "Have both films, need to verify SCW theme"
- **Missing**: "Explicit confirmation of Spanish Civil War theme"
- **Query**: "Pan's Labyrinth Spanish Civil War theme plot"
- **Found**: Set during Spanish Civil War aftermath

## Technical Details

### Context Building
Each hop builds cumulative context:
```python
# Hop 1
hop1_context = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(hop1_docs)])

# Hop 2 (cumulative)
hop2_context = hop1_context + "\n\n" + "\n\n".join([f"Doc {i+1}: {doc}"
                                                     for i, doc in enumerate(hop2_docs)])
```

### Query Planning
Each hop uses full context:
```python
hop2_plan = self.query_planner_hop2(
    claim=claim,
    retrieved_context=hop1_context  # Full context from previous hop
)
hop2_query = hop2_plan.next_query  # Targeted query
```

## Performance Implications

### Expected Improvements
1. **Better Retrieval Coverage**: Targeted queries should find more relevant docs
2. **Reduced Redundancy**: Gap analysis prevents duplicate retrievals
3. **Strategic Navigation**: Explicit reasoning guides multi-hop path
4. **Improved Success Rate**: More likely to find all required facts

### Potential Trade-offs
1. **Increased Latency**: Chain-of-thought reasoning adds overhead
2. **More LM Calls**: Each hop now generates reasoning + missing_info + query
3. **Context Length**: Preserving full docs increases context size

## Testing and Validation

### Verification Performed
1. ✅ Module imports successfully
2. ✅ Pipeline integration works
3. ✅ Backward compatibility maintained
4. ✅ Test script demonstrates functionality

### To Run Tests
```bash
# Test the signature
python test_cot_query_planner.py

# Verify imports
python -c "from langProBe.hover.hover_program import HoverMultiHopPredict, ChainOfThoughtQueryPlanner"

# Verify pipeline
python -c "from langProBe.hover.hover_pipeline import HoverMultiHopPredictPipeline"
```

## Future Enhancements

### Potential Improvements
1. **Dynamic Hop Count**: Adjust hops based on complexity/confidence
2. **Early Stopping**: Stop when all gaps are filled
3. **Confidence Scoring**: Track verification confidence
4. **Query Refinement**: Refine based on retrieval quality
5. **Entity Tracking**: Explicitly track entities across hops
6. **Relationship Mapping**: Map entity relationships

### Integration Opportunities
1. **Optimization**: Use DSPy's optimization to tune query generation
2. **Evaluation**: Add metrics for reasoning quality
3. **Caching**: Cache intermediate reasoning results
4. **Visualization**: Visualize multi-hop reasoning chain

## References

### Related Files
- `/workspace/langProBe/hover/hover_program.py` - Main implementation
- `/workspace/langProBe/hover/hover_pipeline.py` - Pipeline integration
- `/workspace/langProBe/dspy_program.py` - Base classes
- `/workspace/codeevolver/hover.md` - HoVer dataset documentation

### DSPy Components Used
- `dspy.ChainOfThought` - Chain-of-thought reasoning module
- `dspy.Signature` - Structured signature definition
- `dspy.InputField` / `dspy.OutputField` - Field definitions
- `dspy.Retrieve` - Retrieval module

## Summary

The chain-of-thought query planning module transforms the HoverMultiHopPredict system from using simple query generation to strategic, reasoning-driven multi-hop retrieval. By explicitly decomposing reasoning, analyzing gaps, and generating targeted queries, the system should achieve better coverage and more efficient retrieval for complex multi-hop fact verification tasks.

The implementation maintains full backward compatibility while adding powerful new capabilities for explainable, strategic multi-hop reasoning.

---

**Implementation Date**: 2026-02-13
**Modified Files**: 1
**New Files**: 4 (test + 3 documentation)
**Lines Changed**: ~60 lines in hover_program.py
**Backward Compatible**: ✅ Yes
