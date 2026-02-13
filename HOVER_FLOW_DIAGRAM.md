# HoverMultiHopPredict Flow Diagrams

## OLD APPROACH: Simple Query Generation

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT: Claim                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │ Extract Key Terms   │
                    │  (dspy.Predict)     │
                    └─────────────────────┘
                              ↓
            ┌─────────────────────────────────────────┐
            │           HOP 1                         │
            ├─────────────────────────────────────────┤
            │  Query = claim                          │
            │  ↓                                       │
            │  Retrieve(k=9)                          │
            │  ↓                                       │
            │  Extract titles from docs               │
            └─────────────────────────────────────────┘
                              ↓
            ┌─────────────────────────────────────────┐
            │           HOP 2                         │
            ├─────────────────────────────────────────┤
            │  Inputs: claim, key_terms, hop1_titles  │
            │  ↓                                       │
            │  Generate Query (dspy.Predict)          │
            │  ↓                                       │
            │  Retrieve(k=9)                          │
            │  ↓                                       │
            │  Extract titles from docs               │
            └─────────────────────────────────────────┘
                              ↓
            ┌─────────────────────────────────────────┐
            │           HOP 3                         │
            ├─────────────────────────────────────────┤
            │  Inputs: claim, key_terms,              │
            │          hop1_titles, hop2_titles       │
            │  ↓                                       │
            │  Generate Query (dspy.Predict)          │
            │  ↓                                       │
            │  Retrieve(k=9)                          │
            └─────────────────────────────────────────┘
                              ↓
                ┌───────────────────────────┐
                │   Return All Retrieved    │
                │         Documents         │
                └───────────────────────────┘

Problems:
❌ No explicit reasoning about what's needed
❌ Only titles preserved → information loss
❌ No gap analysis between hops
❌ Queries may be generic/redundant
```

## NEW APPROACH: Chain-of-Thought Query Planning

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT: Claim                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
            ┌─────────────────────────────────────────┐
            │           HOP 1: Initial Analysis       │
            ├─────────────────────────────────────────┤
            │  Inputs:                                │
            │    • claim                              │
            │    • retrieved_context = ""             │
            │                                         │
            │  ┌───────────────────────────────────┐ │
            │  │ ChainOfThought Query Planner     │ │
            │  ├───────────────────────────────────┤ │
            │  │ 1. Reasoning:                    │ │
            │  │    "Claim mentions X and Y,      │ │
            │  │     need to find Z first..."     │ │
            │  │                                  │ │
            │  │ 2. Missing Information:          │ │
            │  │    "Need to identify entity X"   │ │
            │  │                                  │ │
            │  │ 3. Next Query:                   │ │
            │  │    "Focused query for X"         │ │
            │  └───────────────────────────────────┘ │
            │                ↓                        │
            │  Retrieve(next_query, k=9)              │
            │                ↓                        │
            │  Build full context (not just titles)   │
            └─────────────────────────────────────────┘
                              ↓
            ┌─────────────────────────────────────────┐
            │           HOP 2: Gap Analysis           │
            ├─────────────────────────────────────────┤
            │  Inputs:                                │
            │    • claim                              │
            │    • retrieved_context = hop1_context   │
            │                                         │
            │  ┌───────────────────────────────────┐ │
            │  │ ChainOfThought Query Planner     │ │
            │  ├───────────────────────────────────┤ │
            │  │ 1. Reasoning:                    │ │
            │  │    "Found X, now need Y to       │ │
            │  │     connect X to Z..."           │ │
            │  │                                  │ │
            │  │ 2. Missing Information:          │ │
            │  │    "Found: X                     │ │
            │  │     Missing: Y's relationship"   │ │
            │  │                                  │ │
            │  │ 3. Next Query:                   │ │
            │  │    "Query targeting Y"           │ │
            │  └───────────────────────────────────┘ │
            │                ↓                        │
            │  Retrieve(next_query, k=9)              │
            │                ↓                        │
            │  Accumulate context (hop1 + hop2)       │
            └─────────────────────────────────────────┘
                              ↓
            ┌─────────────────────────────────────────┐
            │       HOP 3: Final Target Retrieval     │
            ├─────────────────────────────────────────┤
            │  Inputs:                                │
            │    • claim                              │
            │    • retrieved_context = hop2_context   │
            │      (cumulative from hops 1 & 2)       │
            │                                         │
            │  ┌───────────────────────────────────┐ │
            │  │ ChainOfThought Query Planner     │ │
            │  ├───────────────────────────────────┤ │
            │  │ 1. Reasoning:                    │ │
            │  │    "Have X and Y, need final     │ │
            │  │     confirmation of Z..."        │ │
            │  │                                  │ │
            │  │ 2. Missing Information:          │ │
            │  │    "Found: X, Y                  │ │
            │  │     Missing: Z verification"     │ │
            │  │                                  │ │
            │  │ 3. Next Query:                   │ │
            │  │    "Precise query for Z"         │ │
            │  └───────────────────────────────────┘ │
            │                ↓                        │
            │  Retrieve(next_query, k=9)              │
            └─────────────────────────────────────────┘
                              ↓
                ┌───────────────────────────┐
                │   Return All Retrieved    │
                │         Documents         │
                └───────────────────────────┘

Benefits:
✅ Explicit reasoning at each hop
✅ Full context preserved across hops
✅ Gap analysis guides next query
✅ Targeted, focused queries
✅ Strategic multi-hop navigation
```

## Detailed Flow: Example Claim

**Claim:** "The director of the 2017 film The Shape of Water also directed a 2006 film about the Spanish Civil War."

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLAIM ANALYSIS                           │
├─────────────────────────────────────────────────────────────────┤
│  Entities:                                                      │
│    • Film 1: "The Shape of Water" (2017)                        │
│    • Film 2: Unknown (2006)                                     │
│    • Director: Unknown                                          │
│    • Theme: Spanish Civil War                                   │
│                                                                 │
│  Relationships Needed:                                          │
│    1. Director → Film 1                                         │
│    2. Director → Film 2                                         │
│    3. Film 2 → Theme                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                           HOP 1                                 │
├─────────────────────────────────────────────────────────────────┤
│  Context In: [Empty]                                            │
│                                                                 │
│  Reasoning:                                                     │
│    "Need to identify the director of The Shape of Water        │
│     (2017) first, as that's the starting point."               │
│                                                                 │
│  Missing Info:                                                  │
│    "Director's identity is unknown"                             │
│                                                                 │
│  Query:                                                         │
│    "The Shape of Water 2017 director Guillermo del Toro"       │
│                                                                 │
│  Retrieved Docs:                                                │
│    ✓ The Shape of Water directed by Guillermo del Toro         │
│    ✓ Won Best Picture, Best Director                            │
│    ✓ Released December 2017                                     │
│                                                                 │
│  Context Out:                                                   │
│    [Director = Guillermo del Toro identified] ✅                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                           HOP 2                                 │
├─────────────────────────────────────────────────────────────────┤
│  Context In:                                                    │
│    "Guillermo del Toro directed The Shape of Water (2017)"     │
│                                                                 │
│  Reasoning:                                                     │
│    "Now we know del Toro is the director. Need to find his     │
│     2006 film and confirm it's about the Spanish Civil War."   │
│                                                                 │
│  Missing Info:                                                  │
│    "2006 film title and its connection to Spanish Civil War"   │
│                                                                 │
│  Query:                                                         │
│    "Guillermo del Toro 2006 film Spanish Civil War"            │
│                                                                 │
│  Retrieved Docs:                                                │
│    ✓ Pan's Labyrinth (2006) directed by Guillermo del Toro     │
│    ✓ Set in 1944 post-Spanish Civil War Spain                  │
│    ✓ Dark fantasy film                                          │
│                                                                 │
│  Context Out:                                                   │
│    [2006 film = Pan's Labyrinth] ✅                             │
│    [Theme = Spanish Civil War era] ✅                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                           HOP 3                                 │
├─────────────────────────────────────────────────────────────────┤
│  Context In:                                                    │
│    "Guillermo del Toro directed The Shape of Water (2017)      │
│     and Pan's Labyrinth (2006), which is set in Spanish        │
│     Civil War era Spain"                                        │
│                                                                 │
│  Reasoning:                                                     │
│    "Have director (del Toro), 2017 film (Shape of Water),      │
│     and 2006 film (Pan's Labyrinth). Need to verify Pan's      │
│     Labyrinth specifically deals with Spanish Civil War."      │
│                                                                 │
│  Missing Info:                                                  │
│    "Explicit confirmation of Spanish Civil War theme"          │
│                                                                 │
│  Query:                                                         │
│    "Pan's Labyrinth 2006 Spanish Civil War theme plot"         │
│                                                                 │
│  Retrieved Docs:                                                │
│    ✓ Set during Spanish Civil War aftermath                    │
│    ✓ Protagonist lives with fascist stepfather                 │
│    ✓ Historical context is Spanish Civil War                   │
│                                                                 │
│  Context Out:                                                   │
│    [All facts verified] ✅✅✅                                   │
│    CLAIM CAN NOW BE EVALUATED                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Information Flow Comparison

### OLD: Information Loss

```
HOP 1: Full Docs → Extract Titles → [Title1, Title2, ...]
                      ↓
                [Information Lost!]
                      ↓
HOP 2: Uses only titles from Hop 1
       Full Docs → Extract Titles → [Title3, Title4, ...]
                      ↓
                [Information Lost!]
                      ↓
HOP 3: Uses only titles from Hops 1 & 2
```

### NEW: Full Context Preservation

```
HOP 1: Full Docs → Full Context Preserved → [Doc1, Doc2, Doc3, ...]
                                  ↓
                    [All Information Retained!]
                                  ↓
HOP 2: Has full context from Hop 1
       Full Docs → Add to Context → [Doc1...Doc9]
                                  ↓
                    [Cumulative Information!]
                                  ↓
HOP 3: Has full context from Hops 1 & 2
       Full Docs → Final Context → [Doc1...Doc18]
```

## Strategic Query Evolution

### OLD: Generic Queries

```
HOP 1: "The Shape of Water"
       ↓
HOP 2: "The Shape of Water director 2006 Spanish Civil War film"
       ↓
HOP 3: "Shape of Water director 2006 film Spanish Civil War"
       ↓
[Queries are similar and may retrieve redundant docs]
```

### NEW: Targeted Queries

```
HOP 1: "The Shape of Water 2017 director"
       ↓ [Found: Director is Guillermo del Toro]

HOP 2: "Guillermo del Toro 2006 film Spanish Civil War"
       ↓ [Found: Pan's Labyrinth in 2006]

HOP 3: "Pan's Labyrinth Spanish Civil War theme"
       ↓ [Verifying: Connection to Spanish Civil War]

[Each query builds on previous findings and targets specific gaps]
```

## Summary: Key Architectural Improvements

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURAL CHANGES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. MODULE TYPE                                                 │
│     OLD: dspy.Predict                                           │
│     NEW: dspy.ChainOfThought                                    │
│     → Encourages explicit reasoning steps                       │
│                                                                 │
│  2. SIGNATURE STRUCTURE                                         │
│     OLD: "claim, key_terms, titles -> query"                    │
│     NEW: "claim, context -> reasoning, missing, query"          │
│     → Structured decomposition of reasoning                     │
│                                                                 │
│  3. CONTEXT HANDLING                                            │
│     OLD: Only titles preserved                                  │
│     NEW: Full document context preserved                        │
│     → No information loss between hops                          │
│                                                                 │
│  4. GAP ANALYSIS                                                │
│     OLD: None                                                   │
│     NEW: Explicit missing_information field                     │
│     → Identifies what's found vs. what's needed                 │
│                                                                 │
│  5. QUERY TARGETING                                             │
│     OLD: Generic queries with all info                          │
│     NEW: Focused queries for specific gaps                      │
│     → More efficient and effective retrieval                    │
│                                                                 │
│  6. REASONING VISIBILITY                                        │
│     OLD: Black box                                              │
│     NEW: Explicit reasoning field                               │
│     → Better debugging and explainability                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
