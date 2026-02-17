# HoverMultiHop Architecture Comparison

## Original Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Original HoverMultiHop                   │
└─────────────────────────────────────────────────────────────┘

Input: claim
    │
    ▼
┌───────────────┐
│   HOP 1       │  Query: claim (as-is)
│   k=7         │  ─────────────────────► [7 documents]
└───────────────┘
    │
    ▼
┌───────────────┐
│  Summarize    │  Summary of 7 docs
│   Hop 1       │
└───────────────┘
    │
    ▼
┌───────────────┐
│   HOP 2       │  Query: generated from claim + summary_1
│   k=7         │  ─────────────────────► [7 documents]
└───────────────┘
    │
    ▼
┌───────────────┐
│  Summarize    │  Summary of hop 2 docs
│   Hop 2       │
└───────────────┘
    │
    ▼
┌───────────────┐
│   HOP 3       │  Query: from claim + summary_1 + summary_2
│   k=7         │  ─────────────────────► [7 documents]
└───────────────┘
    │
    ▼
Output: 21 documents (7+7+7)

ISSUES:
❌ No explicit entity awareness
❌ Queries not entity-focused
❌ Fixed k=7 for all hops
❌ No reranking or quality filtering
❌ May miss important entities
```

## New Entity-Extraction-First Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              New Entity-Focused HoverMultiHop               │
└─────────────────────────────────────────────────────────────┘

Input: claim
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│          PHASE 1: Entity Extraction                       │
│  ┌─────────────────────────────────────────────────┐     │
│  │     ClaimEntityExtractor                        │     │
│  │  • Primary Entities (core subjects)             │     │
│  │  • Secondary Entities (context/bridging)        │     │
│  │  • Relationships (entity connections)           │     │
│  │  • Key Facts (dates, events to verify)          │     │
│  └─────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────┘
    │
    ├──────────────┬──────────────┬──────────────┐
    │              │              │              │
    ▼              ▼              ▼              ▼
  Primary      Secondary    Relationships    Key Facts
  Entities     Entities
    │              │              │              │
    └──────────────┴──────────────┴──────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────────────────┐
│          PHASE 2: Multi-Hop Targeted Retrieval            │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│  HOP 1: Primary Entity Retrieval                          │
│  ┌─────────────────────────────────────────┐             │
│  │  EntityBasedQueryGenerator              │             │
│  │  Input: Primary Entities + Relationships│             │
│  │  Output: Focused query                  │             │
│  └─────────────────────────────────────────┘             │
│                      │                                    │
│                      ▼                                    │
│  ┌─────────────────────────────────────────┐             │
│  │  Retrieve (k=30)                        │             │
│  │  Focus: Core subjects of the claim      │             │
│  └─────────────────────────────────────────┘             │
│                      │                                    │
│                      ▼                                    │
│              [30 documents]                               │
└───────────────────────────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────┐
│  HOP 2: Secondary/Bridging Entity Retrieval               │
│  ┌─────────────────────────────────────────┐             │
│  │  EntityBasedQueryGenerator              │             │
│  │  Input: Secondary Entities + Context    │             │
│  │  Output: Focused query                  │             │
│  └─────────────────────────────────────────┘             │
│                      │                                    │
│                      ▼                                    │
│  ┌─────────────────────────────────────────┐             │
│  │  Retrieve (k=20)                        │             │
│  │  Focus: Supporting & bridging entities  │             │
│  └─────────────────────────────────────────┘             │
│                      │                                    │
│                      ▼                                    │
│              [20 documents]                               │
└───────────────────────────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────┐
│  HOP 3: Relationship Verification Retrieval               │
│  ┌─────────────────────────────────────────┐             │
│  │  EntityBasedQueryGenerator              │             │
│  │  Input: Relationships + Key Facts       │             │
│  │  Output: Verification query             │             │
│  └─────────────────────────────────────────┘             │
│                      │                                    │
│                      ▼                                    │
│  ┌─────────────────────────────────────────┐             │
│  │  Retrieve (k=15)                        │             │
│  │  Focus: Entity relationships & facts    │             │
│  └─────────────────────────────────────────┘             │
│                      │                                    │
│                      ▼                                    │
│              [15 documents]                               │
└───────────────────────────────────────────────────────────┘
                       │
                       ▼
              Total: 65 documents
                       │
                       ▼
┌───────────────────────────────────────────────────────────┐
│          PHASE 3: Relevance-Based Reranking               │
│  ┌─────────────────────────────────────────┐             │
│  │  For each of 65 documents:              │             │
│  │                                         │             │
│  │  DocumentRelevanceScorer                │             │
│  │  • Score: 0-10 relevance                │             │
│  │  • Track: Covered entities              │             │
│  │                                         │             │
│  │  Scoring factors:                       │             │
│  │  ✓ Entity mentions                      │             │
│  │  ✓ Relationship discussion              │             │
│  │  ✓ Fact verification                    │             │
│  └─────────────────────────────────────────┘             │
│                      │                                    │
│                      ▼                                    │
│  ┌─────────────────────────────────────────┐             │
│  │  Sort by relevance score (descending)   │             │
│  └─────────────────────────────────────────┘             │
│                      │                                    │
│                      ▼                                    │
│  ┌─────────────────────────────────────────┐             │
│  │  Greedy Selection Algorithm             │             │
│  │  • Select top 21 documents              │             │
│  │  • Maximize entity coverage             │             │
│  │  • Ensure diversity                     │             │
│  └─────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────┘
                       │
                       ▼
Output: 21 optimized documents with maximum entity coverage

IMPROVEMENTS:
✅ Explicit entity extraction before retrieval
✅ Entity-focused query generation
✅ Adaptive k values (30/20/15) per hop type
✅ Comprehensive coverage (65 docs → 21 best)
✅ Relevance-based quality filtering
✅ Entity coverage maximization
```

## Key Architectural Differences

| Aspect | Original | New Entity-First |
|--------|----------|------------------|
| **Entity Awareness** | Implicit in queries | Explicit extraction phase |
| **Query Strategy** | Generic summaries | Targeted entity clusters |
| **Retrieval Scale** | Fixed k=7 per hop | Adaptive k=30/20/15 |
| **Document Pool** | 21 (7×3) | 65 → ranked to 21 |
| **Quality Control** | None | Relevance scoring |
| **Entity Coverage** | Not tracked | Explicitly maximized |
| **Query Focus** | Broad | Narrow (per entity type) |
| **Reranking** | None | Score-based selection |

## Entity Flow Example

Consider the claim: "The director of The Matrix also directed Cloud Atlas with Tom Hanks."

### Entity Extraction Output:
```
Primary Entities:
  - The Matrix (movie)
  - Cloud Atlas (movie)
  - Tom Hanks (actor)

Secondary Entities:
  - Wachowskis (directors - implicit)
  - Movie directors (category)
  - Film production

Relationships:
  - Director of The Matrix = Director of Cloud Atlas
  - Tom Hanks acted in Cloud Atlas
  - Same person directed both films

Key Facts:
  - The Matrix release date
  - Cloud Atlas release date
  - Tom Hanks filmography
  - Director credits
```

### Query Generation:

**Hop 1 (k=30):** "The Matrix director Cloud Atlas Tom Hanks filmography"
→ Retrieves comprehensive docs about these core entities

**Hop 2 (k=20):** "Wachowski brothers filmography Matrix Cloud Atlas directors"
→ Retrieves docs about bridging entities (the actual directors)

**Hop 3 (k=15):** "Wachowski directed The Matrix and Cloud Atlas with Tom Hanks"
→ Retrieves docs that verify the specific relationships

### Reranking:
Scores each of 65 documents based on:
- How many entities mentioned (The Matrix, Cloud Atlas, Tom Hanks, Wachowskis)
- Whether directorship relationship is confirmed
- Whether Tom Hanks' role is mentioned
- Factual accuracy of claims

Selects top 21 that collectively cover all entities and relationships.

## Performance Implications

### Original Approach:
- **Pros**: Simple, fast (3 retrievals × k=7)
- **Cons**: May miss entities, no quality filtering

### New Entity-First Approach:
- **Pros**: Comprehensive entity coverage, quality filtering, targeted retrieval
- **Cons**: More LLM calls (extraction + scoring), higher latency

### Trade-offs:
- Accuracy ↑ (better entity coverage)
- Latency ↑ (more processing steps)
- Cost ↑ (more LLM calls for scoring)
- Precision ↑ (relevance-based selection)
