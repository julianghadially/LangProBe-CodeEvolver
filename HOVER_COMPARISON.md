# HoverMultiHopPredict: Before vs After Comparison

## Side-by-Side Code Comparison

### Module Initialization

#### BEFORE
```python
class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 9
        # Simple Predict modules - no explicit reasoning
        self.extract_key_terms = dspy.Predict("claim->key_terms")
        self.create_query_hop2 = dspy.Predict("claim,key_terms,hop1_titles->query")
        self.create_query_hop3 = dspy.Predict("claim,key_terms,hop1_titles,hop2_titles->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
```

#### AFTER
```python
class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 9
        # Chain-of-Thought modules with structured reasoning
        self.query_planner_hop1 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.query_planner_hop2 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.query_planner_hop3 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.retrieve_k = dspy.Retrieve(k=self.k)
```

### Forward Pass - Hop 1

#### BEFORE
```python
def forward(self, claim):
    # Extract key terms from claim (3-5 key named entities or concepts)
    key_terms = self.extract_key_terms(claim=claim).key_terms

    # HOP 1 - Just retrieve based on claim
    hop1_docs = self.retrieve_k(claim).passages
    hop1_titles = ", ".join([doc.split("\n")[0] if "\n" in doc else doc[:100]
                            for doc in hop1_docs])
```

#### AFTER
```python
def forward(self, claim):
    # HOP 1: Initial analysis and query generation with reasoning
    hop1_plan = self.query_planner_hop1(
        claim=claim,
        retrieved_context=""  # Empty for first hop
    )
    hop1_query = hop1_plan.next_query  # Focused query based on reasoning
    hop1_docs = self.retrieve_k(hop1_query).passages
    hop1_context = "\n\n".join([f"Doc {i+1}: {doc}"
                                for i, doc in enumerate(hop1_docs)])
```

**Key Differences:**
- **NEW**: Explicit reasoning about what entities/relationships are needed
- **NEW**: Gap analysis (what's missing from retrieved_context)
- **NEW**: Targeted query generation based on reasoning
- **NEW**: Full document context preservation (not just titles)

### Forward Pass - Hop 2

#### BEFORE
```python
    # HOP 2 - Query based on claim, key_terms, and hop1 titles
    hop2_query = self.create_query_hop2(
        claim=claim,
        key_terms=key_terms,
        hop1_titles=hop1_titles
    ).query
    hop2_docs = self.retrieve_k(hop2_query).passages
    hop2_titles = ", ".join([doc.split("\n")[0] if "\n" in doc else doc[:100]
                            for doc in hop2_docs])
```

#### AFTER
```python
    # HOP 2: Reason about what was found and what's missing
    hop2_plan = self.query_planner_hop2(
        claim=claim,
        retrieved_context=hop1_context  # Full context from hop 1
    )
    hop2_query = hop2_plan.next_query  # Query targets missing info
    hop2_docs = self.retrieve_k(hop2_query).passages
    hop2_context = hop1_context + "\n\n" + "\n\n".join([f"Doc {i+1}: {doc}"
                                                         for i, doc in enumerate(hop2_docs)])
```

**Key Differences:**
- **NEW**: Receives full retrieved context from previous hop
- **NEW**: Analyzes what was found vs. what's still missing
- **NEW**: Generates query specifically for missing information
- **NEW**: Accumulates full context (not just titles)

### Forward Pass - Hop 3

#### BEFORE
```python
    # HOP 3 - Query based on all previous information
    hop3_query = self.create_query_hop3(
        claim=claim,
        key_terms=key_terms,
        hop1_titles=hop1_titles,
        hop2_titles=hop2_titles
    ).query
    hop3_docs = self.retrieve_k(hop3_query).passages

    return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
```

#### AFTER
```python
    # HOP 3: Final targeted retrieval for remaining gaps
    hop3_plan = self.query_planner_hop3(
        claim=claim,
        retrieved_context=hop2_context  # Cumulative context from hops 1 & 2
    )
    hop3_query = hop3_plan.next_query  # Precise query for final gaps
    hop3_docs = self.retrieve_k(hop3_query).passages

    return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
```

**Key Differences:**
- **NEW**: Receives cumulative context from all previous hops
- **NEW**: Identifies final gaps in information
- **NEW**: Generates precise query to fill remaining gaps

## Query Generation Comparison

### BEFORE: Simple Signature
```python
# Hop 2 signature
"claim, key_terms, hop1_titles -> query"

# Hop 3 signature
"claim, key_terms, hop1_titles, hop2_titles -> query"
```

**Problems:**
- No explicit reasoning about what's needed
- No gap analysis
- Passes all available information without structure
- May generate generic queries
- No tracking of what's been found vs. what's missing

### AFTER: Structured Chain-of-Thought
```python
class ChainOfThoughtQueryPlanner(dspy.Signature):
    """Analyze the claim and retrieved context to strategically plan the next retrieval query.

    Decompose what entities, relationships, and facts are needed to verify the claim.
    Analyze what information has already been found versus what's still missing.
    Generate a targeted query to find the specific missing information needed for the next hop.
    """

    claim = dspy.InputField(
        desc="The claim that needs to be verified through multi-hop reasoning"
    )
    retrieved_context = dspy.InputField(
        desc="The context retrieved so far from previous hops (may be empty for first hop)"
    )

    reasoning = dspy.OutputField(
        desc="Explain the multi-hop reasoning chain needed: what entities/relationships "
             "are mentioned in the claim and how they connect"
    )
    missing_information = dspy.OutputField(
        desc="Identify specific gaps: what key information was found in retrieved_context "
             "vs. what's still needed to verify the claim"
    )
    next_query = dspy.OutputField(
        desc="A focused search query to find the specific missing information identified above"
    )
```

**Benefits:**
- ✅ Explicit multi-hop reasoning decomposition
- ✅ Gap analysis at each hop
- ✅ Targeted query generation
- ✅ Better explainability
- ✅ Strategic navigation of multi-hop space

## Example Output Comparison

### Claim
*"The director of the 2017 film The Shape of Water also directed a 2006 film about the Spanish Civil War."*

### BEFORE (Hypothetical - no reasoning visible)

**Hop 1:**
- Input: claim
- Query: "The Shape of Water 2017"

**Hop 2:**
- Input: claim, key_terms="Shape of Water, director, 2006, Spanish Civil War", hop1_titles="..."
- Query: "The Shape of Water director 2006 film Spanish Civil War"

**Hop 3:**
- Input: claim, key_terms, hop1_titles, hop2_titles
- Query: "Shape of Water director Spanish Civil War 2006 film"

**Issues:**
- ❌ No visible reasoning
- ❌ May not understand what's missing
- ❌ Queries may be redundant or generic

### AFTER (With Chain-of-Thought)

**Hop 1:**
- **Reasoning**: "The claim mentions two films: 'The Shape of Water' (2017) and a 2006 film about the Spanish Civil War. First, we need to identify the director of The Shape of Water, then find their 2006 film."
- **Missing**: "Director's identity and filmography"
- **Query**: "Guillermo del Toro The Shape of Water director"

**Hop 2:**
- **Reasoning**: "Now we know Guillermo del Toro directed The Shape of Water. We need to find his 2006 film and verify it's about the Spanish Civil War."
- **Missing**: "2006 film title and confirmation of Spanish Civil War theme"
- **Query**: "Guillermo del Toro 2006 film Spanish Civil War"

**Hop 3:**
- **Reasoning**: "We've identified del Toro as the director and need to confirm the specific 2006 film (Pan's Labyrinth) is about the Spanish Civil War."
- **Missing**: "Verification that Pan's Labyrinth is about the Spanish Civil War"
- **Query**: "Pan's Labyrinth 2006 Guillermo del Toro Spanish Civil War theme"

**Benefits:**
- ✅ Clear reasoning at each step
- ✅ Explicit gap identification
- ✅ Focused, non-redundant queries
- ✅ Strategic progression toward verification

## Performance Implications

### Context Quality
- **BEFORE**: Only titles passed between hops → Loss of detailed information
- **AFTER**: Full document context preserved → Better reasoning about gaps

### Query Targeting
- **BEFORE**: Generic queries with all available info → May retrieve redundant docs
- **AFTER**: Focused queries for missing info → More efficient retrievals

### Explainability
- **BEFORE**: Black box query generation → Hard to debug
- **AFTER**: Explicit reasoning → Easy to understand and debug

### Multi-Hop Reasoning
- **BEFORE**: Implicit reasoning → May miss connections
- **AFTER**: Explicit decomposition → Better multi-hop navigation

## Migration Guide

The new implementation is **backward compatible** - no changes needed to calling code:

```python
# Both old and new versions work the same way externally
from langProBe.hover.hover_program import HoverMultiHopPredict

program = HoverMultiHopPredict()
result = program(claim="Your claim here")
retrieved_docs = result.retrieved_docs
```

The improvements are internal to the query planning process and don't affect the API.

## Summary of Changes

| Aspect | Before | After |
|--------|--------|-------|
| Query Module | `dspy.Predict` | `dspy.ChainOfThought` |
| Signature | Simple string | Structured signature class |
| Reasoning | Implicit | Explicit with reasoning field |
| Gap Analysis | None | Explicit missing_information field |
| Context Passing | Titles only | Full document context |
| Query Focus | Generic | Targeted for missing info |
| Explainability | Low | High |
| Multi-Hop Strategy | Implicit | Explicit decomposition |
