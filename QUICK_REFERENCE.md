# Chain-of-Thought Query Planner - Quick Reference

## TL;DR

Replaced simple `dspy.Predict` query generators with `dspy.ChainOfThought` modules that explicitly reason about multi-hop chains, identify information gaps, and generate targeted queries.

## The Signature

```python
ChainOfThoughtQueryPlanner:
  Inputs:  claim, retrieved_context
  Outputs: reasoning, missing_information, next_query
```

## What Each Field Does

| Field | Type | Purpose |
|-------|------|---------|
| **claim** | Input | The fact to verify |
| **retrieved_context** | Input | Cumulative docs from previous hops |
| **reasoning** | Output | Explains multi-hop chain needed |
| **missing_information** | Output | What's found vs. what's missing |
| **next_query** | Output | Focused query for missing info |

## Before vs After

### Before
```python
# Simple Predict
self.create_query_hop2 = dspy.Predict("claim,key_terms,hop1_titles->query")

# Usage
hop2_query = self.create_query_hop2(
    claim=claim,
    key_terms=key_terms,
    hop1_titles=hop1_titles
).query
```

### After
```python
# Chain-of-Thought
self.query_planner_hop2 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)

# Usage
hop2_plan = self.query_planner_hop2(
    claim=claim,
    retrieved_context=hop1_context  # Full context, not just titles
)
hop2_query = hop2_plan.next_query   # Also get .reasoning and .missing_information
```

## Key Changes

1. ✅ **Module Type**: `dspy.Predict` → `dspy.ChainOfThought`
2. ✅ **Input**: Titles → Full document context
3. ✅ **Output**: Query only → Reasoning + Gaps + Query
4. ✅ **Strategy**: Generic queries → Targeted queries for gaps

## Example Output

```python
hop2_plan = query_planner(claim="...", retrieved_context="...")

# Available fields:
print(hop2_plan.reasoning)
# "Now we know Guillermo del Toro directed The Shape of Water.
#  We need to find his 2006 film about the Spanish Civil War."

print(hop2_plan.missing_information)
# "Found: Director is Guillermo del Toro
#  Missing: 2006 film title and Spanish Civil War connection"

print(hop2_plan.next_query)
# "Guillermo del Toro 2006 film Spanish Civil War"
```

## Usage (Unchanged!)

```python
from langProBe.hover.hover_program import HoverMultiHopPredict

program = HoverMultiHopPredict()
result = program(claim="Your claim here")
docs = result.retrieved_docs  # Still works the same!
```

## Files Modified

- ✏️ `/workspace/langProBe/hover/hover_program.py` (~60 lines changed)

## Files Added

- 📄 `test_cot_query_planner.py` - Test/demo script
- 📄 `HOVER_COT_QUERY_PLANNER_README.md` - Full documentation
- 📄 `HOVER_COMPARISON.md` - Before/after comparison
- 📄 `HOVER_FLOW_DIAGRAM.md` - Visual diagrams
- 📄 `IMPLEMENTATION_SUMMARY.md` - Implementation details
- 📄 `QUICK_REFERENCE.md` - This file

## Run Tests

```bash
# Test the new signature
python test_cot_query_planner.py

# Verify imports
python -c "from langProBe.hover.hover_program import HoverMultiHopPredict, ChainOfThoughtQueryPlanner; print('✓ Success')"
```

## Benefits

| Benefit | How |
|---------|-----|
| **Explicit Reasoning** | `reasoning` field forces decomposition |
| **Gap Analysis** | `missing_information` identifies what's needed |
| **Targeted Queries** | `next_query` focuses on specific gaps |
| **Full Context** | Preserves all docs, not just titles |
| **Explainability** | Can see reasoning at each hop |
| **Strategic Navigation** | Guides multi-hop path intelligently |

## Common Questions

**Q: Do I need to change my code?**
A: No! External API is unchanged.

**Q: Will this slow things down?**
A: Slightly - chain-of-thought adds reasoning overhead, but should improve success rate.

**Q: Can I see the reasoning?**
A: Yes! Each hop plan has `.reasoning` and `.missing_information` fields (though they're not returned by default - you'd need to modify the code to expose them).

**Q: How do I access the reasoning during execution?**
A: Currently the reasoning is used internally. To expose it, modify the `forward()` method to include it in the returned `dspy.Prediction`.

**Q: Is this compatible with DSPy optimization?**
A: Yes! The signature can be optimized like any other DSPy module.

## Architecture Diagram

```
┌────────────┐
│   Claim    │
└─────┬──────┘
      │
      ▼
┌──────────────────────────┐
│  Hop 1: Initial Analysis │
│  ┌────────────────────┐  │
│  │ • Reasoning        │  │
│  │ • Missing Info     │  │
│  │ • Next Query       │  │
│  └────────────────────┘  │
│  Retrieve docs           │
└─────┬────────────────────┘
      │ Full context
      ▼
┌──────────────────────────┐
│  Hop 2: Gap Analysis     │
│  ┌────────────────────┐  │
│  │ • Reasoning        │  │
│  │ • Missing Info     │  │
│  │ • Next Query       │  │
│  └────────────────────┘  │
│  Retrieve docs           │
└─────┬────────────────────┘
      │ Cumulative context
      ▼
┌──────────────────────────┐
│  Hop 3: Final Targeted   │
│  ┌────────────────────┐  │
│  │ • Reasoning        │  │
│  │ • Missing Info     │  │
│  │ • Next Query       │  │
│  └────────────────────┘  │
│  Retrieve docs           │
└─────┬────────────────────┘
      │
      ▼
┌──────────────┐
│ All Retrieved│
│  Documents   │
└──────────────┘
```

## Next Steps

1. Run the test to see it in action
2. Read the full documentation for details
3. Try with your own claims
4. Consider adding evaluation metrics
5. Explore optimization with DSPy

## Links

- [Full Documentation](HOVER_COT_QUERY_PLANNER_README.md)
- [Before/After Comparison](HOVER_COMPARISON.md)
- [Flow Diagrams](HOVER_FLOW_DIAGRAM.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Test Script](test_cot_query_planner.py)

---

**Quick Start**: `python test_cot_query_planner.py`
