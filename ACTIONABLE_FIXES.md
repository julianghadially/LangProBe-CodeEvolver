# Actionable Fixes for 0.0 Score Problem

## Quick Reference

**Problem**: System gets 0.0% score because entity names are abstracted into "verification aspects" and never directly queried.

**Solution**: Pass entity names directly to query generation and use them in queries.

**Estimated Impact**: 0.0% → 30-50% score improvement (based on baseline achieving 34.33-46.67%)

---

## Fix 1: Minimal Entity Passthrough (Easiest)

### Modify TargetedQueryGeneratorSignature

**File**: `/workspace/langProBe/hover/hover_program.py`

**Current (Lines 30-42)**:
```python
class TargetedQueryGeneratorSignature(dspy.Signature):
    """Generate a targeted search query to fill coverage gaps.
    Focus on under-covered or missing aspects. Use negative signals from already-retrieved document titles to diversify results."""

    claim = dspy.InputField(desc="The original claim being verified")
    missing_aspects = dspy.InputField(desc="Verification aspects that have not been covered yet")
    under_covered_aspects = dspy.InputField(desc="Aspects needing more evidence")
    coverage_summary = dspy.InputField(desc="Summary of evidence found and gaps remaining")
    retrieved_titles = dspy.InputField(desc="Titles of documents already retrieved (avoid retrieving again)")

    query = dspy.OutputField(desc="A search query focused on missing/under-covered aspects, formulated to retrieve documents different from those already obtained")
    rationale = dspy.OutputField(desc="Brief explanation of which gap this query targets")
```

**Fixed (Add entities field)**:
```python
class TargetedQueryGeneratorSignature(dspy.Signature):
    """Generate a targeted search query to fill coverage gaps.
    Focus on under-covered or missing aspects. Use negative signals from already-retrieved document titles to diversify results.
    IMPORTANT: Use entity names directly in queries for precise Wikipedia article retrieval."""

    claim = dspy.InputField(desc="The original claim being verified")
    entities = dspy.InputField(desc="List of specific entity names from the claim (people, places, organizations, albums, films, etc.)")  # NEW
    missing_aspects = dspy.InputField(desc="Verification aspects that have not been covered yet")
    under_covered_aspects = dspy.InputField(desc="Aspects needing more evidence")
    coverage_summary = dspy.InputField(desc="Summary of evidence found and gaps remaining")
    retrieved_titles = dspy.InputField(desc="Titles of documents already retrieved (avoid retrieving again)")

    query = dspy.OutputField(desc="A search query focused on missing entities and aspects. Use entity names directly (e.g., 'Boy Hits Car band' not 'band A identity')")
    rationale = dspy.OutputField(desc="Brief explanation of which gap this query targets")
```

### Modify forward() to Pass Entities

**Current (Lines 70-96)**:
```python
def forward(self, claim):
    # INITIALIZATION: Extract verification aspects
    tracker_output = self.entity_tracker(claim=claim)
    verification_aspects = tracker_output.verification_aspects
    all_retrieved_titles = []

    # HOP 1: Direct claim-based retrieval
    hop1_docs = self.retrieve_k(claim).passages
    hop1_titles = self._extract_titles(hop1_docs)
    all_retrieved_titles.extend(hop1_titles)

    # ANALYZE HOP 1 COVERAGE
    coverage1 = self.coverage_analyzer1(
        claim=claim,
        verification_aspects=verification_aspects,
        retrieved_titles=hop1_titles,
        passages=hop1_docs
    )

    # HOP 2: Coverage-driven query
    hop2_query_output = self.query_generator_hop2(
        claim=claim,
        missing_aspects=coverage1.missing_aspects,
        under_covered_aspects=coverage1.under_covered_aspects,
        coverage_summary=coverage1.coverage_summary,
        retrieved_titles=all_retrieved_titles
    )
```

**Fixed (Extract and pass entities)**:
```python
def forward(self, claim):
    # INITIALIZATION: Extract verification aspects
    tracker_output = self.entity_tracker(claim=claim)
    verification_aspects = tracker_output.verification_aspects
    entities = tracker_output.entities  # EXTRACT ENTITIES
    all_retrieved_titles = []

    # HOP 1: Direct claim-based retrieval
    hop1_docs = self.retrieve_k(claim).passages
    hop1_titles = self._extract_titles(hop1_docs)
    all_retrieved_titles.extend(hop1_titles)

    # ANALYZE HOP 1 COVERAGE
    coverage1 = self.coverage_analyzer1(
        claim=claim,
        verification_aspects=verification_aspects,
        retrieved_titles=hop1_titles,
        passages=hop1_docs
    )

    # HOP 2: Coverage-driven query WITH ENTITIES
    hop2_query_output = self.query_generator_hop2(
        claim=claim,
        entities=entities,  # PASS ENTITIES
        missing_aspects=coverage1.missing_aspects,
        under_covered_aspects=coverage1.under_covered_aspects,
        coverage_summary=coverage1.coverage_summary,
        retrieved_titles=all_retrieved_titles
    )
```

Repeat the same change for `query_generator_hop3` (Line 111-117).

**Complete Fix**:
```python
# HOP 3: Final gap-filling query WITH ENTITIES
hop3_query_output = self.query_generator_hop3(
    claim=claim,
    entities=entities,  # PASS ENTITIES
    missing_aspects=coverage2.missing_aspects,
    under_covered_aspects=coverage2.under_covered_aspects,
    coverage_summary=coverage2.coverage_summary,
    retrieved_titles=all_retrieved_titles
)
```

### Expected Impact

**Before**:
- Queries like "band discography album count"
- Retrieves generic music pages

**After**:
- Queries like "The Invisible band discography" (includes entity name)
- Higher probability of retrieving correct Wikipedia article

**Estimated improvement**: 0.0% → 15-25%

---

## Fix 2: Entity-Aware Coverage Analysis (Better)

### Modify CoverageAnalyzerSignature

**Current (Lines 15-28)**:
```python
class CoverageAnalyzerSignature(dspy.Signature):
    """Analyze coverage of verification aspects based on retrieved documents.
    Identify which aspects are well-covered, which need more evidence, and which are missing."""

    claim = dspy.InputField(desc="The original claim being verified")
    verification_aspects = dspy.InputField(desc="List of aspects that need to be verified")
    retrieved_titles = dspy.InputField(desc="Titles of documents already retrieved (to avoid duplicates)")
    passages = dspy.InputField(desc="The document passages retrieved in this hop")

    covered_aspects = dspy.OutputField(desc="Aspects that are well-covered by the retrieved documents")
    under_covered_aspects = dspy.OutputField(desc="Aspects mentioned but needing more evidence")
    missing_aspects = dspy.OutputField(desc="Aspects not yet addressed by retrieved documents")
    coverage_summary = dspy.OutputField(desc="Brief summary of what evidence has been found and what gaps remain")
```

**Fixed (Add entity tracking)**:
```python
class CoverageAnalyzerSignature(dspy.Signature):
    """Analyze entity coverage based on retrieved documents.
    Identify which entities are covered, which need more evidence, and which are completely missing.
    PRIORITY: Track entity retrieval, not abstract aspects."""

    claim = dspy.InputField(desc="The original claim being verified")
    required_entities = dspy.InputField(desc="List of entity names that must be retrieved for claim verification")  # NEW
    verification_aspects = dspy.InputField(desc="List of aspects that need to be verified")
    retrieved_titles = dspy.InputField(desc="Titles of documents already retrieved (to avoid duplicates)")
    passages = dspy.InputField(desc="The document passages retrieved in this hop")

    covered_entities = dspy.OutputField(desc="Entity names that are covered by retrieved documents")  # NEW
    missing_entities = dspy.OutputField(desc="Entity names not yet found in retrieved documents")  # NEW
    covered_aspects = dspy.OutputField(desc="Aspects that are well-covered by the retrieved documents")
    under_covered_aspects = dspy.OutputField(desc="Aspects mentioned but needing more evidence")
    missing_aspects = dspy.OutputField(desc="Aspects not yet addressed by retrieved documents")
    coverage_summary = dspy.OutputField(desc="Brief summary of what evidence has been found and what gaps remain")
```

### Update forward() to Track Entity Coverage

**Updated HOP 2 Query Generation**:
```python
# ANALYZE HOP 1 COVERAGE
coverage1 = self.coverage_analyzer1(
    claim=claim,
    required_entities=entities,  # PASS ENTITIES
    verification_aspects=verification_aspects,
    retrieved_titles=hop1_titles,
    passages=hop1_docs
)

# HOP 2: Entity-focused query
hop2_query_output = self.query_generator_hop2(
    claim=claim,
    entities=entities,
    missing_entities=coverage1.missing_entities,  # PRIORITIZE MISSING ENTITIES
    missing_aspects=coverage1.missing_aspects,
    under_covered_aspects=coverage1.under_covered_aspects,
    coverage_summary=coverage1.coverage_summary,
    retrieved_titles=all_retrieved_titles
)
```

### Update Query Generator to Prioritize Entities

**Modified Signature**:
```python
class TargetedQueryGeneratorSignature(dspy.Signature):
    """Generate a targeted search query to fill coverage gaps.
    PRIORITY: Query for missing entities by name first, then address abstract aspects.
    Use entity names directly in queries for precise Wikipedia article retrieval."""

    claim = dspy.InputField(desc="The original claim being verified")
    entities = dspy.InputField(desc="List of specific entity names from the claim")
    missing_entities = dspy.InputField(desc="Entity names not yet retrieved (HIGHEST PRIORITY)")  # NEW
    missing_aspects = dspy.InputField(desc="Verification aspects that have not been covered yet")
    under_covered_aspects = dspy.InputField(desc="Aspects needing more evidence")
    coverage_summary = dspy.InputField(desc="Summary of evidence found and gaps remaining")
    retrieved_titles = dspy.InputField(desc="Titles of documents already retrieved (avoid retrieving again)")

    query = dspy.OutputField(desc="A search query. If missing_entities exist, query for them directly by name. Otherwise, query for missing aspects.")
    rationale = dspy.OutputField(desc="Brief explanation of which gap this query targets")
```

### Expected Impact

**Before**:
- Coverage analysis returns abstract aspects
- Queries target concepts not entities

**After**:
- Coverage analysis explicitly tracks entity names
- Queries prioritize retrieving missing entities by name

**Estimated improvement**: 0.0% → 25-40%

---

## Fix 3: Direct Entity Query Strategy (Best)

### Complete Redesign of Query Logic

Replace the coverage-driven approach with direct entity targeting:

**New Signature**:
```python
class SimpleEntityQuerySignature(dspy.Signature):
    """Generate a simple query for a specific entity.
    The query should be the entity name plus minimal context for disambiguation."""

    entity_name = dspy.InputField(desc="The specific entity name to query for")
    claim_context = dspy.InputField(desc="The original claim for context")

    query = dspy.OutputField(desc="A simple query containing the entity name and type (e.g., 'Boy Hits Car band', 'Robert Jordan author')")
```

**New Forward Method**:
```python
def forward(self, claim):
    # STEP 1: Extract entities
    tracker_output = self.entity_tracker(claim=claim)
    entities = tracker_output.entities

    # STEP 2: HOP 1 - Direct claim query
    hop1_docs = self.retrieve_k(claim).passages
    hop1_titles = self._extract_titles(hop1_docs)

    # STEP 3: Identify missing entities
    missing_entities = [
        e for e in entities
        if not any(self._entity_in_title(e, title) for title in hop1_titles)
    ]

    # STEP 4: HOP 2 - Query for first missing entity
    if len(missing_entities) > 0:
        hop2_query = self.simple_query_gen(
            entity_name=missing_entities[0],
            claim_context=claim
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        hop2_titles = self._extract_titles(hop2_docs)
    else:
        # If all entities found, diversify with summary-based query
        hop2_query = claim  # Or use old summarization approach
        hop2_docs = self.retrieve_k(hop2_query).passages
        hop2_titles = self._extract_titles(hop2_docs)

    # STEP 5: HOP 3 - Query for second missing entity
    all_titles = hop1_titles + hop2_titles
    still_missing = [
        e for e in entities
        if not any(self._entity_in_title(e, title) for title in all_titles)
    ]

    if len(still_missing) > 0:
        hop3_query = self.simple_query_gen(
            entity_name=still_missing[0],
            claim_context=claim
        ).query
    else:
        hop3_query = claim  # Default fallback

    hop3_docs = self.retrieve_k(hop3_query).passages

    return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)

def _entity_in_title(self, entity, title):
    """Check if entity name appears in document title (fuzzy match)"""
    entity_normalized = entity.lower().strip()
    title_normalized = title.lower().strip()
    return entity_normalized in title_normalized
```

### Expected Impact

**Strategy**:
- Explicit entity tracking
- Direct entity queries (name only + type)
- Simple, deterministic logic

**Estimated improvement**: 0.0% → 35-50% (matching or exceeding baseline)

---

## Fix 4: Return to Baseline with Enhancements (Safest)

### Revert to Baseline Architecture

Since the baseline achieved 34.33% (HotpotQA) and 46.67% (Hover), consider reverting and adding incremental improvements:

**Baseline Code** (from commit 4a85a65):
```python
def forward(self, claim):
    # HOP 1
    hop1_docs = self.retrieve_k(claim).passages
    summary_1 = self.summarize1(
        claim=claim, passages=hop1_docs
    ).summary

    # HOP 2
    hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
    hop2_docs = self.retrieve_k(hop2_query).passages
    summary_2 = self.summarize2(
        claim=claim, context=summary_1, passages=hop2_docs
    ).summary

    # HOP 3
    hop3_query = self.create_query_hop3(
        claim=claim, summary_1=summary_1, summary_2=summary_2
    ).query
    hop3_docs = self.retrieve_k(hop3_query).passages

    return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
```

### Enhancement: Add Entity Awareness

**Modified Baseline**:
```python
class EntityAwareSummarizerSignature(dspy.Signature):
    """Summarize passages while preserving all entity names (people, places, organizations, albums, films).
    CRITICAL: List all entity names mentioned in passages before summarizing."""

    claim = dspy.InputField(desc="The claim being verified")
    passages = dspy.InputField(desc="Retrieved document passages")

    mentioned_entities = dspy.OutputField(desc="List of all entity names mentioned in passages")
    summary = dspy.OutputField(desc="Summary of passages that includes all entity names")

class EntityAwareQuerySignature(dspy.Signature):
    """Generate next query that targets missing entities or explores relationships.
    If entities from the claim are not mentioned in summaries, query for them directly."""

    claim = dspy.InputField(desc="The original claim")
    summary_1 = dspy.InputField(desc="Summary from hop 1")
    mentioned_entities = dspy.InputField(desc="Entities found so far")

    query = dspy.OutputField(desc="Next search query, prioritizing missing entities from claim")

# In __init__:
self.summarize1 = dspy.ChainOfThought(EntityAwareSummarizerSignature)
self.create_query_hop2 = dspy.ChainOfThought(EntityAwareQuerySignature)
```

### Expected Impact

**Strategy**:
- Keep proven baseline structure
- Add entity awareness to summaries and queries
- Minimal risk, incremental improvement

**Estimated improvement**: 34-46% → 40-55%

---

## Comparison of Fixes

| Fix | Complexity | Risk | Expected Score | Implementation Time |
|-----|-----------|------|----------------|-------------------|
| **Fix 1**: Entity Passthrough | Low | Low | 15-25% | 15 minutes |
| **Fix 2**: Entity-Aware Coverage | Medium | Medium | 25-40% | 30 minutes |
| **Fix 3**: Direct Entity Strategy | High | Medium | 35-50% | 1-2 hours |
| **Fix 4**: Enhanced Baseline | Low | Very Low | 40-55% | 30 minutes |

---

## Recommended Action Plan

### Phase 1: Quick Win (Today)
1. **Implement Fix 1** (Entity Passthrough)
   - Add `entities` field to `TargetedQueryGeneratorSignature`
   - Pass `entities` to query generators in `forward()`
   - Test on dev set
   - **Goal**: Achieve 15-25% score

### Phase 2: Validation (This Week)
2. **Implement Fix 4** (Enhanced Baseline) on separate branch
   - Revert to baseline code
   - Add entity-aware summarization
   - Test on dev set
   - **Goal**: Achieve 40-55% score

3. **Compare** Fix 1 vs Fix 4
   - Choose best performing approach
   - Merge to main branch

### Phase 3: Optimization (Next Week)
4. **Refine** chosen approach
   - Tune prompts
   - Optimize entity extraction
   - Add entity disambiguation
   - **Goal**: Achieve 50-60% score

---

## Testing Commands

### Run Evaluation on Dev Set
```bash
cd /workspace
python -m langProBe.evaluation \
    --benchmark hover \
    --dataset_mode debug \
    --file_path evaluation_test \
    --suppress_dspy_output \
    --lm openai/gpt-4o-mini
```

### Check Results
```bash
cat evaluation_test/evaluation_results.csv
```

### Quick Entity Coverage Test
```python
import dspy
from langProBe.hover.hover_program import HoverMultiHop

# Configure
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    rm=dspy.ColBERTv2(url="https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search")
)

# Test
program = HoverMultiHop()
result = program(claim="Boy Hits Car released My Animal and has more albums than The Invisible")

# Check titles
titles = [doc.split(" | ")[0] for doc in result.retrieved_docs]
print("Retrieved titles:")
for title in titles:
    print(f"  - {title}")

# Check for required entities
required = ["Boy Hits Car", "The Invisible", "My Animal"]
for entity in required:
    found = any(entity.lower() in title.lower() for title in titles)
    print(f"{entity}: {'✓' if found else '✗'}")
```

---

## Code Diffs

### Fix 1: Minimal Entity Passthrough

```diff
--- a/langProBe/hover/hover_program.py
+++ b/langProBe/hover/hover_program.py
@@ -30,11 +30,12 @@ class CoverageAnalyzerSignature(dspy.Signature):
 class TargetedQueryGeneratorSignature(dspy.Signature):
     """Generate a targeted search query to fill coverage gaps.
-    Focus on under-covered or missing aspects. Use negative signals from already-retrieved document titles to diversify results."""
+    Focus on under-covered or missing aspects. Use negative signals from already-retrieved document titles to diversify results.
+    IMPORTANT: Use entity names directly in queries for precise Wikipedia article retrieval."""

     claim = dspy.InputField(desc="The original claim being verified")
+    entities = dspy.InputField(desc="List of specific entity names from the claim")
     missing_aspects = dspy.InputField(desc="Verification aspects that have not been covered yet")
     under_covered_aspects = dspy.InputField(desc="Aspects needing more evidence")
     coverage_summary = dspy.InputField(desc="Summary of evidence found and gaps remaining")
@@ -71,6 +72,7 @@ class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
         # INITIALIZATION: Extract verification aspects
         tracker_output = self.entity_tracker(claim=claim)
         verification_aspects = tracker_output.verification_aspects
+        entities = tracker_output.entities  # Extract entities
         all_retrieved_titles = []

         # HOP 1: Direct claim-based retrieval
@@ -90,6 +92,7 @@ class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
         # HOP 2: Coverage-driven query
         hop2_query_output = self.query_generator_hop2(
             claim=claim,
+            entities=entities,  # Pass entities
             missing_aspects=coverage1.missing_aspects,
             under_covered_aspects=coverage1.under_covered_aspects,
             coverage_summary=coverage1.coverage_summary,
@@ -111,6 +114,7 @@ class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
         # HOP 3: Final gap-filling query
         hop3_query_output = self.query_generator_hop3(
             claim=claim,
+            entities=entities,  # Pass entities
             missing_aspects=coverage2.missing_aspects,
             under_covered_aspects=coverage2.under_covered_aspects,
             coverage_summary=coverage2.coverage_summary,
```

---

## Monitoring and Validation

### Metrics to Track
1. **Entity Retrieval Rate**: % of gold entities retrieved
2. **Score Improvement**: Accuracy on dev set
3. **Query Quality**: Manual inspection of generated queries

### Success Criteria
- **Minimum**: Score > 20% (better than 0.0%)
- **Target**: Score > 35% (match baseline)
- **Stretch**: Score > 50% (exceed baseline)

---

## Conclusion

**Immediate Action**: Implement Fix 1 (15 minutes) to get from 0.0% to 15-25%

**Best Long-term**: Implement Fix 4 (enhanced baseline) for 40-55% score

**Why**: The current coverage-driven approach has a fundamental flaw (entity abstraction). Rather than trying to fix multiple layers of abstraction, either:
1. Add a simple entity passthrough (Fix 1), or
2. Return to the proven baseline and add entity awareness (Fix 4)

Both approaches keep entity names intact throughout the pipeline, which is the key to success.

---

*Document prepared: 2026-02-17*
*Branch: codeevolver-20260217004441-a9b59e*
*For: CodeEvolver optimization*
