# Verification-Guided Retrieval Architecture

## Overview
This implementation adds a self-correction loop to the HoverMultiHop system that uses LLM reasoning to identify retrieval gaps and strategically replace weak documents with more relevant ones.

## Architecture Components

### 1. **VerifyClaimSupport** Signature (Lines 4-12)
- **Purpose**: Analyzes the claim and all 21 retrieved documents to identify missing or unclear supporting facts
- **Input**: The claim + all 21 documents from 3-hop retrieval
- **Output**: Specific missing facts needed to fully verify/refute the claim
- **Key Design**: Focuses on concrete factual gaps rather than general categories

### 2. **TargetedRetrievalQuery** Signature (Lines 14-22)
- **Purpose**: Generates highly specific queries focused on missing information
- **Input**: Original claim + identified missing facts
- **Output**: Laser-focused query using precise terms and entities
- **Key Design**: Avoids broad queries, targets specific gaps

### 3. **DocumentRelevanceScorer** Signature (Lines 24-34)
- **Purpose**: Scores documents (0-10) based on how well they fill identified gaps
- **Input**: Claim + missing facts + individual document
- **Output**: Numeric relevance score + reasoning
- **Key Design**: Evaluates gap-filling capability, not general relevance

## Workflow

### Phase 1: Initial 3-Hop Retrieval (Lines 59-79)
- **Hop 1**: Direct retrieval on claim → 7 docs
- **Hop 2**: Query based on Hop 1 summary → 7 docs
- **Hop 3**: Query based on Hops 1+2 summaries → 7 docs
- **Total**: 21 documents

### Phase 2: Verification Loop (Lines 81-140)

#### Step 1: Identify Gaps (Lines 82-87)
```python
verification_result = self.verify_claim_support(
    claim=claim,
    retrieved_documents="[All 21 docs]"
)
```
- LLM analyzes all documents
- Identifies specific missing supporting facts

#### Step 2: Generate Targeted Query (Lines 89-94)
```python
targeted_query = self.generate_targeted_query(
    claim=claim,
    missing_facts=missing_facts
)
```
- Creates query focused solely on filling gaps
- Uses specific terms from missing facts

#### Step 3: Execute Targeted Retrieval (Lines 96-97)
```python
targeted_docs = self.retrieve_targeted(targeted_query).passages
```
- Retrieves 21 candidate documents with targeted query
- These candidates are specifically aimed at filling gaps

#### Step 4: Score Targeted Documents (Lines 99-117)
```python
for doc in targeted_docs:
    score = self.score_document(
        claim=claim,
        missing_facts=missing_facts,
        document=doc
    )
```
- Scores all 21 targeted docs on gap-filling ability
- Selects top 7 highest-scoring documents

#### Step 5: Score Initial Documents (Lines 119-131)
```python
for doc in initial_docs:
    score = self.score_document(
        claim=claim,
        missing_facts=missing_facts,
        document=doc
    )
```
- Scores all 21 initial docs using same criteria
- Identifies the 7 weakest documents

#### Step 6: Strategic Replacement (Lines 133-138)
```python
refined_docs = [doc for doc, score in initial_scores[7:]] + top_targeted_docs
```
- Keeps top 14 documents from initial retrieval
- Replaces bottom 7 with top 7 from targeted retrieval
- Maintains 21-document limit

## Key Design Decisions

### 1. **Budget-Aware Design**
- Keeps k=7 for initial 3 hops (3 × 7 = 21 retrieval calls)
- Uses k=21 for single targeted retrieval (21 retrieval calls)
- Total: 42 retrievals within budget constraints

### 2. **LLM-Guided Gap Identification**
- Uses Chain-of-Thought reasoning to identify specific gaps
- Focuses on concrete factual needs, not abstract categories
- Leverages LLM's understanding of claim-evidence relationships

### 3. **Strategic Document Replacement**
- Doesn't discard all initial documents (preserves good ones)
- Uses comparative scoring to identify weakest links
- Replaces worst performers with best gap-fillers

### 4. **Scoring Robustness**
- Try-except blocks handle LLM output variations
- Default scores prevent crashes (0.0 for targeted, 5.0 for initial)
- Numeric extraction with fallback handling

### 5. **Single Verification Loop**
- One correction pass balances quality vs. compute cost
- Prevents exponential growth in retrieval calls
- Sufficient for most claim verification scenarios

## Benefits

1. **Self-Correction**: System identifies its own retrieval gaps
2. **Targeted Improvement**: Focuses on specific missing information
3. **Strategic Optimization**: Replaces weakest documents, not random ones
4. **Claim-Specific**: Adapts to the unique needs of each claim
5. **Budget-Compliant**: Stays within 21-document final output limit

## Example Flow

**Claim**: "The actor who played Iron Man was born in New York."

1. **Initial Retrieval**: Gets docs about Iron Man, actors, Marvel movies
2. **Gap Identification**: "Missing: Actor's name, birthplace verification"
3. **Targeted Query**: "Robert Downey Jr birthplace New York"
4. **Targeted Retrieval**: Gets 21 docs about RDJ's biography
5. **Scoring**: RDJ biography docs score 9-10, generic Marvel docs score 2-3
6. **Replacement**: Swaps 7 weak Marvel docs with 7 strong RDJ biography docs
7. **Result**: 14 kept docs + 7 targeted docs = 21 optimized documents

## Performance Characteristics

- **Retrieval Calls**: 42 total (3×7 initial + 21 targeted)
- **LLM Calls**: ~45 (verification + query gen + 42 scoring calls)
- **Output**: Exactly 21 documents
- **Quality**: Higher precision through gap-focused refinement
