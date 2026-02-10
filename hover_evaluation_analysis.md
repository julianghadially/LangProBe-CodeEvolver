# HoVer Multi-Hop Retrieval: Failure Analysis

## Executive Summary

The HoVer system scored **0.0 on all three analyzed examples**, meaning **none of the required supporting documents were found** in the retrieved set. This complete failure stems from fundamental architectural problems in the current multi-hop retrieval approach.

---

## Current System Architecture

The `HoverMultiHopPredict` program implements a 3-hop sequential retrieval approach:

```python
# HOP 1
hop1_docs = Retrieve(claim, k=7)
summary_1 = Summarize(claim, hop1_docs)

# HOP 2
hop2_query = GenerateQuery(claim, summary_1)
hop2_docs = Retrieve(hop2_query, k=7)
summary_2 = Summarize(claim, summary_1, hop2_docs)

# HOP 3
hop3_query = GenerateQuery(claim, summary_1, summary_2)
hop3_docs = Retrieve(hop3_query, k=7)

# Return all 21 documents
return hop1_docs + hop2_docs + hop3_docs
```

**Evaluation Metric**: All required supporting documents must be in the top-21 retrieved documents.

**Current Score**: 46.67% (but 0.0 on the examples analyzed)

---

## Example Failure Analysis

### Example 1: Trigg Hound and Löwchen
**Claim**: "The variety of animal that the Trigg Hound belongs to and Löwchen are not both types of Foxhounds."

| Required Documents | Mentioned in Claim? | Analysis |
|-------------------|-------------------|----------|
| Löwchen | ✓ Direct | Likely retrieved in Hop 1 |
| Trigg Hound | ✓ Direct | Likely retrieved in Hop 1 |
| American Foxhound | ~ Partial ("Foxhounds") | May be missed - specific name not in claim |

**Why it fails**:
- Claim says "Foxhounds" (generic) but needs "American Foxhound" (specific)
- If "American Foxhound" ranks 8th+ in Hop 1 retrieval, it's missed
- Hops 2-3 work from summary that may lose this specificity
- **Result**: Missing 1+ required documents → Score: 0.0

---

### Example 2: River Rat and Coal Miner's Daughter
**Claim**: "The director of the film The River Rat also created the film Coal Miner's Daughter, which won an Academy Award."

| Required Documents | Mentioned in Claim? | Analysis |
|-------------------|-------------------|----------|
| The River Rat | ✓ Direct | Retrieved in Hop 1 |
| Coal Miner's Daughter (film) | ✓ Direct | Retrieved in Hop 1 |
| Thomas Rickman (writer) | ✗ NOT mentioned | **BRIDGING ENTITY** - requires inference |

**Why it fails**:
- "Thomas Rickman (writer)" is the **bridging entity** connecting the two films
- NOT mentioned in the claim - requires multi-hop reasoning
- Hop 1 retrieves film documents, not the director
- Hop 2 must generate query like "Who directed The River Rat?"
- But query generation is implicit and may fail to generate exact entity name
- Even if generated, retrieval may return film doc again instead of director doc
- **Result**: Missing "Thomas Rickman (writer)" → Score: 0.0

---

### Example 3: Filbornaskolan and Celtic
**Claim**: "A notable alumni of Filbornaskolan participated in the Celtic 1977 Scottish League Cup Final. He was a Celtic football coach and former professional player."

| Required Documents | Mentioned in Claim? | Analysis |
|-------------------|-------------------|----------|
| Filbornaskolan | ✓ Direct | Retrieved in Hop 1 |
| 1997 Scottish League Cup Final | ~ Partial ("1977") | May be retrieved |
| Henrik Larsson | ✗ NOT mentioned | **BRIDGING ENTITY** - requires inference |

**Why it fails**:
- "Henrik Larsson" is the **bridging entity** connecting school and match
- Claim mentions "notable alumni" but not the specific person
- Hop 1 retrieves school doc and maybe match doc
- Hop 2 must infer: need to find "notable alumni of Filbornaskolan who played for Celtic"
- Without explicit entity extraction, this connection is missed
- **Result**: Missing "Henrik Larsson" → Score: 0.0

---

## Core Failure Patterns

### Pattern 1: The Bridging Entity Problem

Multi-hop questions have this structure:
```
Entity A ←→ Bridging Entity ←→ Entity B
(in claim)   (NOT in claim)    (in claim)
```

**Example**:
```
"Filbornaskolan" ←→ "Henrik Larsson" ←→ "Celtic Cup Final"
   (mentioned)      (NOT mentioned)         (mentioned)
```

**Current approach fails because**:
1. Hop 1 retrieves Entity A and B documents (directly mentioned)
2. Bridging entity is NOT in claim → not retrieved directly
3. Summarization of A/B docs loses specific entity references
4. LLM query generation must "guess" the bridging entity name
5. Without explicit entity extraction, this guess consistently fails

---

### Pattern 2: Cascading Failures (Sequential Dependency)

```
Hop 1 fails → Summary 1 incomplete → Hop 2 query wrong → Hop 2 fails → ...
```

**Problem**: Each hop depends on success of previous hop. Early mistakes compound.

**Evidence**:
- If Hop 1 misses a key document → Summary 1 is incomplete
- Hop 2 query generation works from incomplete information
- Hop 2 retrieval cannot recover from Hop 1 failure
- System has no error correction or iterative refinement

---

### Pattern 3: Lossy Summarization

**What happens**:
```python
hop1_docs = [7 documents with rich entity information]
summary_1 = Summarize(hop1_docs)  # Compresses to short text
# Specific entity names, dates, relationships are lost
hop2_query = GenerateQuery(claim, summary_1)  # Works from lossy summary
```

**Problem**: Summarization drops critical details:
- Specific entity names (e.g., "Henrik Larsson" becomes "a player")
- Dates and numbers (e.g., "1997" becomes "late 1990s")
- Relationships (e.g., "alumni of" becomes "associated with")

**Impact**: Hop 2 query generation doesn't have the specificity needed to retrieve the right documents.

---

### Pattern 4: Insufficient Retrieval Coverage

**Current**: k=7 per hop × 3 hops = 21 total documents

**Problems**:
- Retrieval space is massive (likely millions of documents)
- k=7 is too small - if target document ranks 8th or lower, it's missed
- No guarantee that required documents are in top-7 for any given query
- Evaluation shows this is systematically insufficient

**Evidence**: 0.0 scores indicate required documents aren't in the 21 retrieved.

---

### Pattern 5: Implicit Query Generation Without Entity Extraction

**Current approach**:
```python
hop2_query = LLM.Generate(claim, summary_1)
# LLM must "guess" what entity to search for
```

**Problems**:
- No explicit entity extraction from claim or documents
- No structured reasoning chain
- No verification that required entities are present
- LLM query generation is a black box - no guarantees

**What's needed**:
- Explicit entity extraction: `entities = Extract(claim)` → ["Filbornaskolan", "Celtic Cup Final"]
- Explicit reasoning: "Need to find: person connecting these entities"
- Targeted retrieval: `Retrieve("alumni of Filbornaskolan")`

---

## Root Cause Analysis

### The Core Problem: Missing Structured Multi-Hop Reasoning

The current system treats multi-hop retrieval as:
1. Retrieve some docs
2. Summarize
3. Generate a query (somehow)
4. Repeat

**What's actually needed**:
1. **Parse** the claim to identify entities and relationships
2. **Decompose** the claim into sub-questions
3. **Plan** retrieval strategy for each sub-question
4. **Retrieve** in parallel or with verification
5. **Verify** that required entities are found
6. **Iterate** if needed

---

## Why Sequential Hops Don't Work

### Problem: Information Loss Across Hops

```
Claim (full info)
  → Hop 1 Retrieval (partial info)
    → Summary 1 (lossy compression)
      → Hop 2 Query (guessed from incomplete info)
        → Hop 2 Retrieval (likely wrong target)
          → Summary 2 (more loss)
            → Hop 3 Query (even more guessing)
              → Hop 3 Retrieval (probably wrong)
```

Each arrow represents **information loss**. By Hop 3, the system has diverged significantly from what's actually needed.

---

## What Works in Other Systems

### 1. IRCoT (Interleaving Retrieval with Chain-of-Thought)
- Explicitly generates reasoning chain before retrieval
- Example: "First, find the director of The River Rat. Then, find films by that director."
- Each step has explicit sub-query
- Iterative verification

### 2. Decompose (Question Decomposition)
- Breaks multi-hop question into single-hop sub-questions
- Example: Q1: "Who directed The River Rat?" Q2: "What other films did that director make?"
- Answers sub-questions sequentially
- Each answer feeds into next question

### 3. ReAct (Reasoning + Acting)
- Iterative search-and-reasoning loop
- After each retrieval, evaluates if answer is found
- If not, generates new search query
- Continues until answer found or max iterations

### 4. Graph-Based Retrieval
- Builds entity graph from claim
- Traverses graph to find connections
- Example: Filbornaskolan → [alumni] → Henrik Larsson → [played for] → Celtic
- Explicit relationship modeling

---

## Recommended Solutions

### Solution 1: Entity Extraction + Parallel Retrieval

**Instead of sequential hops**:
```python
# Extract entities from claim
entities = ExtractEntities(claim)  # ["Filbornaskolan", "Celtic Cup Final", ...]

# Retrieve for each entity in parallel
docs = []
for entity in entities:
    docs += Retrieve(entity, k=10)

# Also retrieve with full claim
docs += Retrieve(claim, k=10)

return docs  # Much higher coverage, no cascading failures
```

**Benefits**:
- No sequential dependency → no cascading failures
- Higher effective k (10 × N entities vs. 7 × 3 hops)
- Direct retrieval for each known entity
- Parallel execution is faster

---

### Solution 2: Query Decomposition

**Decompose claim into sub-questions**:
```python
# For: "Director of River Rat also made Coal Miner's Daughter"
sub_questions = [
    "Who directed The River Rat?",
    "What films did this director make?",
    "Did Coal Miner's Daughter win an Academy Award?"
]

# Retrieve for each sub-question
for question in sub_questions:
    docs += Retrieve(question, k=7)
```

**Benefits**:
- Explicit reasoning structure
- Each sub-question targets specific information
- More focused retrieval queries
- Can verify each step

---

### Solution 3: Iterative Retrieval with Verification

**Add verification and iteration**:
```python
required_entities = ExtractRequiredEntities(claim)
retrieved_entities = set()
max_iterations = 3

for i in range(max_iterations):
    missing = required_entities - retrieved_entities
    if not missing:
        break  # Success!

    # Generate query for missing entities
    query = GenerateQuery(claim, missing)
    new_docs = Retrieve(query, k=10)
    retrieved_entities.update(ExtractEntities(new_docs))
    docs += new_docs
```

**Benefits**:
- Explicit verification of success
- Can recover from early failures
- Iterative refinement
- Stops when task is complete

---

### Solution 4: Increase k and Use Ensemble Retrieval

**Simple but effective**:
```python
# Instead of k=7, use k=15 or k=20
hop1_docs = Retrieve(claim, k=20)

# Or use ensemble
docs = []
docs += DenseRetrieval(claim, k=10)
docs += BM25Retrieval(claim, k=10)
docs += RerankTop(docs, k=20)
```

**Benefits**:
- Better coverage of retrieval space
- Combines strengths of different retrieval methods
- Simple to implement

---

### Solution 5: Remove Summarization Bottleneck

**Instead of summarizing for next hop**:
```python
# Option A: Pass raw documents
hop2_query = GenerateQuery(claim, hop1_docs)  # LLM sees all docs

# Option B: Pass structured entity list
entities = ExtractEntities(hop1_docs)
hop2_query = GenerateQuery(claim, entities)  # No information loss

# Option C: Don't chain at all - parallel retrieval
```

**Benefits**:
- No information loss
- LLM has full context
- Entities and relationships preserved

---

### Solution 6: Reasoning Chain Generation (Like IRCoT)

**Explicit multi-hop reasoning**:
```python
# Generate reasoning chain first
reasoning_chain = GenerateReasoning(claim)
# Output: [
#   "Find: Alumni of Filbornaskolan",
#   "Filter: Those who played for Celtic",
#   "Verify: Participated in 1997 Cup Final"
# ]

# Retrieve for each reasoning step
for step in reasoning_chain:
    query = ConvertToQuery(step)
    docs += Retrieve(query, k=10)
```

**Benefits**:
- Explicit reasoning structure
- Interpretable
- Each step is targeted
- Can debug and improve reasoning

---

## Specific Recommendations for HoVer

### Immediate Fixes (Low-hanging fruit)

1. **Increase k from 7 to 15-20**
   - Simple change, likely improves coverage significantly
   - `self.k = 15` in `hover_program.py`

2. **Remove or reduce summarization**
   - Pass full documents or entity lists to next hop
   - Less information loss

3. **Add direct entity retrieval**
   - Extract entities from claim
   - Retrieve document for each entity
   - Add to existing hops

### Medium-term Improvements

4. **Implement query decomposition**
   - Break claim into sub-questions
   - Generate explicit sub-queries
   - Retrieve for each

5. **Add parallel retrieval**
   - Remove sequential dependency
   - Retrieve for all entities in parallel

### Long-term Solutions

6. **Implement IRCoT-style reasoning**
   - Generate explicit reasoning chain
   - Interleave retrieval with reasoning steps
   - Verify at each step

7. **Build entity graph**
   - Extract entities and relationships
   - Traverse graph to find connections
   - Use structured reasoning

---

## Conclusion

The HoVer system's **0.0 scores** are caused by **systematic architectural problems**, not just parameter tuning:

1. **Sequential dependency** causes cascading failures
2. **Lossy summarization** drops critical entity information
3. **Implicit query generation** without entity extraction fails to target bridging entities
4. **Insufficient retrieval coverage** (k=7 is too small)
5. **No verification or iteration** means early mistakes are never corrected

**The core issue**: Multi-hop questions have **bridging entities** not mentioned in the claim. The current approach cannot systematically find these entities because it relies on implicit LLM query generation from lossy summaries.

**Solution**: Implement explicit entity extraction, query decomposition, and parallel/iterative retrieval with verification.

---

## Appendix: Code References

- **System**: `/workspace/langProBe/hover/hover_program.py`
- **Pipeline**: `/workspace/langProBe/hover/hover_pipeline.py`
- **Evaluation**: `/workspace/langProBe/hover/hover_utils.py` (discrete_retrieval_eval)
- **Benchmark**: `/workspace/langProBe/hover/hover_data.py`
- **Results**: `/workspace/evaluation_hover_baseline/evaluation_results.csv`

Current score: **46.67%** (suggesting the failures analyzed are representative of broader issues)
