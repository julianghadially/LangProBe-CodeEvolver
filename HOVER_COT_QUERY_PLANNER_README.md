# Chain-of-Thought Query Planning Module for HoverMultiHopPredict

## Overview

This document describes the chain-of-thought query planning module added to `hover_program.py` that explicitly decomposes multi-hop reasoning before each retrieval step.

## What Changed

### Before: Simple Query Generation

```python
# OLD APPROACH
self.extract_key_terms = dspy.Predict("claim->key_terms")
self.create_query_hop2 = dspy.Predict("claim,key_terms,hop1_titles->query")
self.create_query_hop3 = dspy.Predict("claim,key_terms,hop1_titles,hop2_titles->query")
```

The old approach used simple `dspy.Predict` modules that:
- Generated queries without explicit reasoning
- Passed all available context (claim, key_terms, titles) to the query generator
- Lacked gap analysis between hops
- Could generate generic or redundant queries

### After: Chain-of-Thought Query Planning

```python
# NEW APPROACH
self.query_planner_hop1 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
self.query_planner_hop2 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
self.query_planner_hop3 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
```

The new approach uses a structured `dspy.ChainOfThought` module with the signature:
```python
claim, retrieved_context -> reasoning, missing_information, next_query
```

## The ChainOfThoughtQueryPlanner Signature

### Input Fields

1. **claim** (str): The claim that needs to be verified through multi-hop reasoning
2. **retrieved_context** (str): The context retrieved so far from previous hops (empty for first hop)

### Output Fields

1. **reasoning** (str): Explains the multi-hop reasoning chain needed
   - Analyzes what entities and relationships are mentioned in the claim
   - Identifies how they connect across hops
   - Plans the overall verification strategy

2. **missing_information** (str): Identifies specific gaps
   - What information was found in retrieved_context
   - What key information is still missing
   - What's needed to complete the verification

3. **next_query** (str): A focused search query
   - Targets the specific missing information identified
   - Avoids redundant retrievals
   - Strategically progresses toward claim verification

## How It Works

### Hop 1: Initial Analysis
```python
hop1_plan = self.query_planner_hop1(
    claim=claim,
    retrieved_context=""  # Empty for first hop
)
hop1_query = hop1_plan.next_query
hop1_docs = self.retrieve_k(hop1_query).passages
```

The system:
1. Analyzes the claim to understand required entities/relationships
2. Plans the multi-hop reasoning chain
3. Generates an initial focused query

### Hop 2: Gap Analysis
```python
hop2_plan = self.query_planner_hop2(
    claim=claim,
    retrieved_context=hop1_context  # Full context from hop 1
)
hop2_query = hop2_plan.next_query
```

The system:
1. Reviews what was found in hop 1
2. Identifies what's still missing
3. Generates a targeted query for the missing information

### Hop 3: Final Targeted Retrieval
```python
hop3_plan = self.query_planner_hop3(
    claim=claim,
    retrieved_context=hop2_context  # Cumulative context from hops 1 & 2
)
hop3_query = hop3_plan.next_query
```

The system:
1. Reviews cumulative context from previous hops
2. Identifies final gaps in information
3. Generates a precise query to fill remaining gaps

## Key Improvements

### 1. Explicit Multi-Hop Reasoning
The `reasoning` field forces the system to:
- Decompose the claim into entities and relationships
- Plan the multi-hop chain before retrieval
- Understand how pieces connect

**Example:**
```
Reasoning: The claim mentions two films: "The Shape of Water" (2017) and a
2006 film about the Spanish Civil War. We need to: (1) identify the director
of The Shape of Water, (2) find their 2006 film, (3) verify it's about the
Spanish Civil War.
```

### 2. Gap Analysis
The `missing_information` field explicitly tracks:
- What information has been found
- What information is still needed
- Prevents redundant retrievals

**Example:**
```
Missing Information: We confirmed Guillermo del Toro directed "The Shape of
Water," but we lack information about his 2006 film and whether it relates
to the Spanish Civil War.
```

### 3. Targeted Queries
The `next_query` field generates:
- Focused searches for specific missing pieces
- Strategic queries based on gap analysis
- Queries that progress toward verification

**Example:**
```
Next Query: What is the title of Guillermo del Toro's 2006 film, and does
it relate to the Spanish Civil War?
```

### 4. Progressive Context Building
Each hop receives full context from previous hops:
- Cumulative understanding across retrievals
- Better reasoning about what's missing
- Strategic navigation of the multi-hop space

### 5. Chain-of-Thought Reasoning
Using `dspy.ChainOfThought` instead of `dspy.Predict`:
- Encourages explicit reasoning steps
- Better explainability of query decisions
- More strategic query generation

## Example Execution

For the claim: *"The director of the 2017 film The Shape of Water also directed a 2006 film about the Spanish Civil War."*

### Hop 1 Output
- **Reasoning**: Identifies need to find director of "The Shape of Water"
- **Missing**: Director's identity and filmography
- **Query**: "The Shape of Water 2017 director"

### Hop 2 Output
- **Reasoning**: Now knows Guillermo del Toro is director, needs 2006 film info
- **Missing**: 2006 film title and connection to Spanish Civil War
- **Query**: "Guillermo del Toro 2006 film Spanish Civil War"

### Hop 3 Output
- **Reasoning**: Needs to confirm specific 2006 film details
- **Missing**: Verification that the 2006 film is about Spanish Civil War
- **Query**: "Pan's Labyrinth 2006 Guillermo del Toro Spanish Civil War"

## Benefits

1. **Strategic Navigation**: The system strategically navigates multi-hop reasoning rather than generating generic queries
2. **Reduced Redundancy**: Gap analysis prevents redundant retrievals
3. **Better Explainability**: Explicit reasoning shows why each query was generated
4. **Focused Retrievals**: Queries target specific missing information
5. **Improved Success Rate**: More likely to find all needed information across hops

## Usage

The module is a drop-in replacement for the original `HoverMultiHopPredict`:

```python
from langProBe.hover.hover_program import HoverMultiHopPredict

# Initialize the program
program = HoverMultiHopPredict()

# Use with a claim
claim = "The director of the 2017 film The Shape of Water also directed a 2006 film about the Spanish Civil War."
result = program(claim=claim)

# Access retrieved documents
retrieved_docs = result.retrieved_docs
```

## Testing

Run the test script to see the chain-of-thought reasoning in action:

```bash
python test_cot_query_planner.py
```

This will demonstrate:
- The signature structure
- Example reasoning at each hop
- Gap analysis across hops
- Targeted query generation

## Future Enhancements

Potential improvements to consider:

1. **Dynamic Hop Count**: Adjust number of hops based on complexity
2. **Confidence Scoring**: Track confidence that required information has been found
3. **Early Stopping**: Stop retrievals when all gaps are filled
4. **Query Refinement**: Refine queries based on retrieval quality
5. **Entity Tracking**: Explicitly track entities across hops
6. **Relationship Mapping**: Map relationships between entities
