# Iterative Retrieval with Gap Analysis Implementation

## Changes Made

### 1. New DSPy Signature: GapAnalysis
Created a new `GapAnalysis` signature in `/workspace/langProBe/hover/hover_program.py` that:
- Takes the claim and initial retrieved document titles as input
- Uses chain-of-thought reasoning to identify missing entities or concepts
- Outputs 1 targeted follow-up query to address the identified gap

### 2. Updated ClaimDecomposition Signature
Modified `ClaimDecomposition` to produce exactly 2 sub-queries (down from 2-3) for the initial retrieval phase.

### 3. Modified HoverMultiHop.forward() Method
Completely restructured the retrieval logic to implement a three-phase architecture:

#### Phase 1: Initial Retrieval (100 documents)
- Decompose claim into 2 focused sub-queries
- Retrieve k=50 documents per sub-query
- Total: 100 documents

#### Phase 2: Gap Analysis & Follow-up Retrieval (150 documents total)
- Extract titles from initial 100 documents
- Use GapAnalysis signature to identify missing entities/concepts
- Generate 1 targeted follow-up query
- Retrieve k=50 additional documents
- Total: 150 documents across all retrievals

#### Phase 3: Two-Stage Reranking (21 final documents)
- Score all 150 documents using RelevanceScorer (chain-of-thought, 1-10 scale)
- Deduplicate by normalized document title
- Select top 21 unique documents by relevance score

### 4. Updated codeevolver.md
Updated architecture documentation to reflect:
- New iterative retrieval strategy
- Addition of GapAnalysis module
- New data flow with three phases
- Purpose statement highlighting gap analysis approach

## Constraints Satisfied

✅ All changes within `langProBe.hover.hover_pipeline.HoverMultiHopPipeline` parent module
✅ No new pipeline/wrapper classes created
✅ Maximum 3 total searches per claim (2 initial + 1 follow-up)
✅ Maintains maximum 21 retrieved documents for evaluation
✅ Uses existing two-stage reranking (RelevanceScorer + deduplication)
✅ All code is within HoverMultiHopPipeline's reachable code

## Technical Details

- Initial retrieval: 2 queries × 50 docs = 100 docs
- Follow-up retrieval: 1 query × 50 docs = 50 docs
- Total retrievals: 150 docs
- Final output: Top 21 unique docs after scoring and deduplication
- Search limit: 3 total searches per claim

## Architecture Rationale

This iterative approach addresses the evaluation pattern where the system consistently misses 1-2 critical documents per claim. The gap analysis phase:
1. Reviews what was retrieved initially
2. Identifies missing entities/concepts using chain-of-thought reasoning
3. Generates a targeted query to fill the gap
4. Increases document pool for better coverage before reranking
