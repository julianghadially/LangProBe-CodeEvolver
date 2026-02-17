# Executive Summary: 0.0 Score Root Cause Analysis

## The Problem

The HoverMultiHop system is achieving **0.0% accuracy** on the Hover benchmark evaluation, compared to the baseline's **46.67%** accuracy.

## Root Cause

**Entity names are being abstracted away and never directly queried.**

### The Failing Pipeline

```
Claim: "Boy Hits Car released My Animal"
    ↓
Extract entities: ["Boy Hits Car", "My Animal"]  ← Entities identified
    ↓
Convert to aspects: ["Band identity", "Album release"]  ← Entities abstracted
    ↓
Generate queries: "band discography album release"  ← Entities lost
    ↓
Retrieve: Generic music articles  ✗ Wrong documents
```

### What Should Happen

```
Claim: "Boy Hits Car released My Animal"
    ↓
Extract entities: ["Boy Hits Car", "My Animal"]
    ↓
Query directly: "Boy Hits Car band"  ← Entity name preserved
    ↓
Retrieve: "Boy Hits Car" Wikipedia article  ✓ Correct document
```

## The Evaluation Metric

The metric is binary and strict:

```python
def discrete_retrieval_eval(example, pred, trace=None):
    gold_titles = set([doc["key"] for doc in example["supporting_facts"]])
    found_titles = set([doc.split(" | ")[0] for doc in pred.retrieved_docs])
    return gold_titles.issubset(found_titles)  # ALL gold titles must be found
```

**Success requires**: Retrieving ALL Wikipedia articles listed in `supporting_facts`
**Current result**: Missing key articles → 0.0 score on every example

## Example Failures

### Example 1: Music Bands
**Claim**: "Boy Hits Car released My Animal and has more albums than The Invisible"
**Required**: Boy Hits Car, The Invisible (band), My Animal
**Retrieved**: Generic band/album articles
**Missing**: All 3 specific Wikipedia articles
**Score**: 0.0

### Example 2: Films
**Claim**: "Mad Hot Ballroom (2005) was released before Koyaanisqatsi, a Powaqqatsi sequel"
**Required**: Koyaanisqatsi, Mad Hot Ballroom, Powaqqatsi
**Retrieved**: Mad Hot Ballroom, Powaqqatsi, various film articles
**Missing**: Koyaanisqatsi (the critical comparison document)
**Score**: 0.0

### Example 3: Authors
**Claim**: "Morgan Llywelyn, author of Lion of Ireland, was born before Robert Jordan"
**Required**: Morgan Llywelyn, Robert Jordan, Lion of Ireland
**Retrieved**: Morgan Llywelyn, Lion of Ireland, various author bios
**Missing**: Robert Jordan (the comparison entity)
**Score**: 0.0

## Pattern Identified

In **every case**, the system:
1. ✓ Extracts entity names correctly
2. ✗ Converts them to abstract "verification aspects"
3. ✗ Generates queries about concepts, not entities
4. ✗ Retrieves semantically similar but incorrect documents
5. ✗ Misses 1+ required Wikipedia articles
6. ✗ Scores 0.0 (fails strict subset test)

## The Fix (One Line of Code)

### Current Code
```python
hop2_query_output = self.query_generator_hop2(
    claim=claim,
    missing_aspects=coverage1.missing_aspects,  # Abstract concepts
    under_covered_aspects=coverage1.under_covered_aspects,
    coverage_summary=coverage1.coverage_summary,
    retrieved_titles=all_retrieved_titles
)
```

### Fixed Code
```python
hop2_query_output = self.query_generator_hop2(
    claim=claim,
    entities=entities,  # ← ADD THIS: Pass entity names directly
    missing_aspects=coverage1.missing_aspects,
    under_covered_aspects=coverage1.under_covered_aspects,
    coverage_summary=coverage1.coverage_summary,
    retrieved_titles=all_retrieved_titles
)
```

**Also update the signature** to accept `entities` as an InputField.

## Expected Impact

| Approach | Score | Change |
|----------|-------|--------|
| **Current** (Coverage-driven, no entities) | 0.0% | Baseline |
| **Fix 1** (Add entity passthrough) | 15-25% | +15-25% |
| **Fix 2** (Entity-aware coverage) | 25-40% | +25-40% |
| **Fix 3** (Direct entity queries) | 35-50% | +35-50% |
| **Fix 4** (Return to baseline + enhancements) | 40-55% | +40-55% |

## Why This Happened

The coverage-driven architecture was designed for **semantic search** but the task requires **entity lookup**.

**Semantic Search** (what the system does):
- Query: "band discography and album count comparison"
- Result: Relevant music industry documents
- Good for: Essay questions, open-ended research

**Entity Lookup** (what the task needs):
- Query: "Boy Hits Car band"
- Result: Exact Wikipedia article "Boy Hits Car"
- Good for: Fact verification, knowledge base queries

## Comparison with Baseline

### Baseline (46.67% accuracy)
```python
# Simple approach: claim → summarize → query → summarize → query
hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
```

**Why it works**:
- Summaries preserve entity names
- Queries naturally include entities from summaries
- Direct pipeline keeps entities visible

### Current (0.0% accuracy)
```python
# Complex approach: claim → aspects → coverage → missing_aspects → query
hop2_query = self.query_generator_hop2(
    missing_aspects=coverage1.missing_aspects,  # Entities lost here
    coverage_summary=coverage1.coverage_summary
).query
```

**Why it fails**:
- Entities converted to aspects ("Band A", "Band B")
- Coverage analysis on abstractions
- Queries target concepts not names
- Entity information is diffused away

## Recommendations

### Immediate (Today)
**Action**: Implement Fix 1 (Entity Passthrough)
**Time**: 15 minutes
**Impact**: 0.0% → 15-25%
**Risk**: Very low
**Files**: `/workspace/langProBe/hover/hover_program.py` (3 lines changed)

### Short-term (This Week)
**Action**: Implement Fix 4 (Enhanced Baseline)
**Time**: 30 minutes
**Impact**: 0.0% → 40-55%
**Risk**: Low (proven baseline + incremental improvement)
**Files**: Revert to baseline, add entity-aware signatures

### Long-term (Next Sprint)
**Action**: Redesign with entity-first architecture
**Time**: 1-2 hours
**Impact**: 50-60%+
**Risk**: Medium (new architecture)
**Files**: Complete refactor of hover_program.py

## Key Insight

**Simpler is better for entity retrieval.**

The baseline's simple approach (claim → summarize → query) outperforms the complex coverage-driven approach (claim → entities → aspects → coverage → missing_aspects → query) because it keeps entity names visible throughout the pipeline.

**Lesson**: For structured entity retrieval tasks, preserve entity names and query for them directly. Abstraction layers hurt more than they help.

## Files Created

1. **EVALUATION_ANALYSIS.md** - Detailed analysis of failures
2. **DETAILED_QUERY_ANALYSIS.md** - Query generation comparison
3. **ACTIONABLE_FIXES.md** - Code changes and implementation guide
4. **EXECUTIVE_SUMMARY.md** - This document

## Testing the Fix

```bash
# Quick test
cd /workspace
python -m langProBe.evaluation \
    --benchmark hover \
    --dataset_mode debug \
    --file_path evaluation_fix_test \
    --lm openai/gpt-4o-mini

# Check score
cat evaluation_fix_test/evaluation_results.csv
```

## Success Criteria

- **Minimum**: Score > 20% (any non-zero score)
- **Target**: Score > 40% (match baseline)
- **Stretch**: Score > 50% (exceed baseline)

---

**Bottom Line**: The system extracts entities correctly but then throws them away. Pass the extracted entities to query generation, and the score will improve dramatically.

---

*Analysis completed: 2026-02-17*
*Current branch: codeevolver-20260217004441-a9b59e*
*Current score: 0.0% (Hover), 0.0% (HotpotQA implied)*
*Baseline score: 46.67% (Hover), 34.33% (HotpotQA)*
*Root cause: Entity names abstracted away in coverage analysis*
*Solution: Pass entity names directly to query generation*
*Estimated fix time: 15 minutes (minimal fix) to 2 hours (complete redesign)*
