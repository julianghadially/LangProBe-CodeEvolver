# HoVer System: Current Approach vs. Ideal Approach

## Side-by-Side Comparison: Example 2 (River Rat Director)

**Claim**: "The director of the film The River Rat also created the film Coal Miner's Daughter, which won an Academy Award."

**Required Documents**:
- The River Rat
- Coal Miner's Daughter (film)
- Thomas Rickman (writer)

---

### CURRENT APPROACH (Score: 0.0)

```
┌─────────────────────────────────────────────────────────────┐
│ HOP 1: Retrieve(claim, k=7)                                  │
└─────────────────────────────────────────────────────────────┘
Query: "The director of the film The River Rat also created..."

Retrieved (likely):
  1. The River Rat (film) ✓
  2. Coal Miner's Daughter (film) ✓
  3. Academy Awards
  4. Directors (general)
  5. Film production
  6. Other films named "Rat"
  7. Coal mining films

Missing: Thomas Rickman (writer) ❌ (not in claim, ranked too low)

┌─────────────────────────────────────────────────────────────┐
│ Summarize: claim + hop1_docs → summary_1                     │
└─────────────────────────────────────────────────────────────┘
summary_1: "The River Rat and Coal Miner's Daughter are both
films. Coal Miner's Daughter won Academy Awards..."

Problem: Director's name not in summary ❌

┌─────────────────────────────────────────────────────────────┐
│ HOP 2: Generate Query(claim, summary_1) → Retrieve(k=7)      │
└─────────────────────────────────────────────────────────────┘
Generated query (LLM guess): "River Rat director films"

Retrieved (likely):
  1. The River Rat (again)
  2. Film directors (general)
  3. Other "Rat" movies
  4. Director filmography pages
  5. Coal Miner's Daughter (again)
  6. Academy Award winning directors
  7. 1980s films

Missing: Thomas Rickman (writer) ❌ (query too generic)

Problem: Without knowing the specific name "Thomas Rickman",
the query is too vague to retrieve the right document

┌─────────────────────────────────────────────────────────────┐
│ HOP 3: Generate Query(claim, summary_1, summary_2) → Retrieve│
└─────────────────────────────────────────────────────────────┘
Summary is even more diluted, same problems compound

Result: 21 documents total, but "Thomas Rickman (writer)" not found

FINAL SCORE: 0.0 ❌
```

---

### IDEAL APPROACH (Predicted Score: ~0.8-1.0)

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Extract Entities and Decompose Claim                │
└─────────────────────────────────────────────────────────────┘
Entities extracted:
  - "The River Rat" (film)
  - "Coal Miner's Daughter" (film)
  - [UNKNOWN DIRECTOR] (to be found)
  - "Academy Award"

Reasoning chain:
  1. Who directed "The River Rat"?
  2. What other films did this director make?
  3. Did "Coal Miner's Daughter" win Academy Awards?

Sub-questions generated:
  Q1: "Who directed The River Rat?"
  Q2: "What is the filmography of [director found in Q1]?"
  Q3: "Did Coal Miner's Daughter win an Academy Award?"

┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Parallel Retrieval for Known Entities (k=10 each)   │
└─────────────────────────────────────────────────────────────┘
Retrieve("The River Rat", k=10):
  → Includes "The River Rat" document ✓
  → May include cast/crew info
  → Document mentions: "Directed by Thomas Rickman"

Retrieve("Coal Miner's Daughter film", k=10):
  → Includes "Coal Miner's Daughter (film)" document ✓
  → May include awards info

Retrieve("The River Rat director", k=10):
  → Targets director specifically
  → Likely includes "Thomas Rickman (writer)" document ✓

Total so far: 30 documents (much better coverage)

┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Extract Entities from Retrieved Docs → Verify       │
└─────────────────────────────────────────────────────────────┘
From "The River Rat" doc:
  → Extract: director = "Thomas Rickman"

Verification:
  ✓ The River Rat document: FOUND
  ✓ Coal Miner's Daughter document: FOUND
  ? Thomas Rickman (writer) document: Check if in retrieved set

If not found, targeted retrieval:
  → Retrieve("Thomas Rickman writer", k=10)
  → Ensures we get the specific document

┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Answer Sub-Questions with Verification              │
└─────────────────────────────────────────────────────────────┘
Q1: "Who directed The River Rat?"
  → Answer: Thomas Rickman ✓ (from retrieved docs)

Q2: "What other films did Thomas Rickman make?"
  → Retrieved "Thomas Rickman (writer)" doc
  → Answer: Coal Miner's Daughter (among others) ✓

Q3: "Did Coal Miner's Daughter win Academy Awards?"
  → Answer: Yes ✓ (from retrieved docs)

All required documents in final set: ✓ ✓ ✓

FINAL SCORE: 1.0 ✓
```

---

## Key Differences: Why Ideal Approach Succeeds

| Aspect | Current Approach | Ideal Approach |
|--------|------------------|----------------|
| **Entity Extraction** | None - relies on implicit retrieval | Explicit extraction of entities from claim |
| **Reasoning** | Implicit in LLM query generation | Explicit reasoning chain and sub-questions |
| **Retrieval Strategy** | Sequential with fixed k=7 | Parallel with k=10-15 per entity |
| **Coverage** | 21 docs total (7×3 hops) | 30-50 docs (10-15 per entity/sub-question) |
| **Verification** | None - hopes for the best | Checks if required entities are found |
| **Error Recovery** | Cannot recover from early mistakes | Iterative: if missing, retrieves again |
| **Bridging Entities** | ❌ Likely missed (not in claim) | ✓ Found via sub-question answering |
| **Information Loss** | High (summarization at each hop) | Low (structured entity lists) |

---

## The Bridging Entity Problem: Detailed Analysis

### What is a Bridging Entity?

In multi-hop questions, a **bridging entity** connects two entities mentioned in the claim, but is itself **not mentioned**.

```
┌─────────────┐       ┌──────────────────┐       ┌─────────────────┐
│  Entity A   │◄─────►│ BRIDGING ENTITY  │◄─────►│   Entity B      │
│ (in claim)  │       │ (NOT in claim)   │       │  (in claim)     │
└─────────────┘       └──────────────────┘       └─────────────────┘
```

### Example 2: Thomas Rickman as Bridging Entity

```
┌──────────────────┐       ┌──────────────────┐       ┌─────────────────────┐
│  The River Rat   │◄─────►│ Thomas Rickman   │◄─────►│  Coal Miner's       │
│   (mentioned)    │       │ (NOT mentioned)  │       │  Daughter           │
│                  │       │                  │       │  (mentioned)        │
│  "film"          │       │  "director"      │       │  "film"             │
└──────────────────┘       └──────────────────┘       └─────────────────────┘
        ↓                           ↓                           ↓
    Retrieved in              MISSED!                   Retrieved in
      Hop 1 ✓                  ❌                         Hop 1 ✓
```

### Why Current Approach Misses Bridging Entities

1. **Hop 1**: Retrieves entities directly mentioned in claim
   - Gets "The River Rat" ✓
   - Gets "Coal Miner's Daughter" ✓
   - Misses "Thomas Rickman" ❌ (not in claim)

2. **Summary 1**: Compresses Hop 1 results
   - "The River Rat and Coal Miner's Daughter are films..."
   - May or may not mention director (depends on retrieved docs)
   - If director name was in retrieved docs, might be lost in summary

3. **Hop 2 Query Generation**: Must infer the bridging entity
   - LLM must guess: "Need to search for the director"
   - But doesn't know the specific name "Thomas Rickman"
   - Generates generic query like "River Rat director"

4. **Hop 2 Retrieval**: Generic query → poor results
   - Might retrieve general "director" documents
   - Might retrieve "The River Rat" doc again
   - Unlikely to retrieve specific "Thomas Rickman (writer)" doc

5. **Result**: Bridging entity missing → Score 0.0

### How Ideal Approach Finds Bridging Entities

1. **Explicit Sub-Question**: "Who directed The River Rat?"
   - Targets the bridging entity directly
   - Not just "retrieve something about directors"
   - But "retrieve THE DIRECTOR OF THIS SPECIFIC FILM"

2. **Document Analysis**: Extract director from retrieved docs
   - "The River Rat" document contains: "Directed by Thomas Rickman"
   - Extract this structured information
   - Now we know: bridging entity = "Thomas Rickman"

3. **Targeted Retrieval**: Retrieve document for bridging entity
   - Retrieve("Thomas Rickman writer", k=10)
   - Gets the specific document needed

4. **Verification**: Check if bridging entity connects A and B
   - Does Thomas Rickman direct The River Rat? ✓
   - Did Thomas Rickman work on Coal Miner's Daughter? ✓
   - Connection verified!

---

## Statistics: Why Coverage Matters

### Current Approach
- Retrieval per hop: k=7
- Number of hops: 3
- Total documents: 21
- Coverage per entity: ~7 documents (assuming 3 entities)

### If target document is ranked:
- Rank 1-7: Retrieved ✓
- Rank 8-21: Might be retrieved in later hops (low probability)
- Rank 22+: Never retrieved ❌

### Problem
If "Thomas Rickman (writer)" is ranked 15th for query "The director of the film The River Rat...", it will be **missed entirely**.

---

### Ideal Approach
- Retrieval per entity: k=10-15
- Number of entities/sub-questions: 3-5
- Total documents: 30-75
- Coverage per entity: 10-15 documents

### If target document is ranked:
- Rank 1-15: Retrieved ✓ (in initial pass)
- Rank 16-30: Retrieved ✓ (in targeted retrieval)
- Rank 31+: Can be retrieved in iterative refinement

### Advantage
Much higher probability of finding all required documents, including bridging entities.

---

## Code Comparison

### Current Implementation
```python
class HoverMultiHopPredict(dspy.Module):
    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(claim=claim, passages=hop1_docs).summary

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(claim=claim, context=summary_1, passages=hop2_docs).summary

        # HOP 3
        hop3_query = self.create_query_hop3(claim=claim, summary_1=summary_1, summary_2=summary_2).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
```

**Issues:**
- Sequential dependency
- Lossy summarization
- Implicit query generation
- k=7 too small
- No verification

---

### Ideal Implementation (Pseudocode)
```python
class HoverIdealPredict(dspy.Module):
    def forward(self, claim):
        # STEP 1: Explicit entity extraction and reasoning
        entities = self.extract_entities(claim).entities  # ["The River Rat", "Coal Miner's Daughter"]
        sub_questions = self.decompose_claim(claim).questions
        # ["Who directed The River Rat?", "What films did this director make?", ...]

        # STEP 2: Parallel retrieval for known entities
        all_docs = []
        for entity in entities:
            all_docs.extend(self.retrieve_k(entity, k=15).passages)

        # Also retrieve for original claim
        all_docs.extend(self.retrieve_k(claim, k=15).passages)

        # STEP 3: Answer sub-questions and extract bridging entities
        for question in sub_questions:
            # Try to answer from existing docs
            answer = self.answer_question(question, all_docs)
            if answer.confidence < threshold:
                # Need more docs
                new_docs = self.retrieve_k(question, k=15).passages
                all_docs.extend(new_docs)

        # STEP 4: Verification - check if required entities are present
        retrieved_entities = self.extract_entities_from_docs(all_docs).entities
        required_entities = self.identify_required_entities(claim, all_docs).entities

        missing = set(required_entities) - set(retrieved_entities)
        if missing:
            # Targeted retrieval for missing entities
            for entity in missing:
                all_docs.extend(self.retrieve_k(entity, k=15).passages)

        # STEP 5: Deduplicate and return top-k
        unique_docs = deduplicate(all_docs)
        return dspy.Prediction(retrieved_docs=unique_docs[:50])
```

**Advantages:**
- Parallel retrieval (no cascading failures)
- Explicit entity extraction and reasoning
- Verification and iterative refinement
- Higher coverage (k=15 per query, multiple queries)
- Can find bridging entities through sub-question answering

---

## Conclusion

The difference between **0.0** and **1.0** comes down to:

1. **Explicit vs. Implicit**: Ideal approach explicitly extracts entities and generates sub-questions, rather than hoping the LLM will implicitly generate the right queries

2. **Parallel vs. Sequential**: Ideal approach retrieves for multiple entities in parallel, avoiding cascading failures

3. **Verification vs. Hope**: Ideal approach verifies required entities are found and retrieves again if needed, rather than hoping they're in the initial 21 documents

4. **Structure vs. Ambiguity**: Ideal approach uses structured entity lists and reasoning chains, rather than lossy text summaries

**The core insight**: Multi-hop questions have **bridging entities** not mentioned in the claim. Finding them requires explicit reasoning and verification, not just sequential retrieval and summarization.
