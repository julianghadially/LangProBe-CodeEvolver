# Architecture Diagram: HotPot Serper + Firecrawl

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HotpotMultiHopPredict                            │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                         HOP 1                                 │   │
│  │                                                               │   │
│  │  1. create_query_hop1                                        │   │
│  │     Input:  question                                         │   │
│  │     Output: optimized_query_1                                │   │
│  │     ├─────────────────────────────────────────────────┐     │   │
│  │     │  DSPy LM: "question -> query"                    │     │   │
│  │     │  Reformulates question for better search         │     │   │
│  │     └─────────────────────────────────────────────────┘     │   │
│  │                                                               │   │
│  │  2. _search_and_scrape(optimized_query_1)                   │   │
│  │     ┌──────────────────────────────────────┐                │   │
│  │     │  SerperService                       │                │   │
│  │     │  • Query: "query site:wikipedia.org" │                │   │
│  │     │  • Returns: Top 5 Wikipedia URLs     │                │   │
│  │     └──────────────────────────────────────┘                │   │
│  │               ↓                                               │   │
│  │     ┌──────────────────────────────────────┐                │   │
│  │     │  FirecrawlService                    │                │   │
│  │     │  • Scrapes: Top URL                  │                │   │
│  │     │  • Returns: Markdown (max 15k chars) │                │   │
│  │     └──────────────────────────────────────┘                │   │
│  │     Output: hop1_context (full Wikipedia page)               │   │
│  │                                                               │   │
│  │  3. summarize1                                               │   │
│  │     Input:  question + hop1_context                          │   │
│  │     Output: summary_1                                        │   │
│  │     ├─────────────────────────────────────────────────┐     │   │
│  │     │  DSPy LM: "question, context -> summary"        │     │   │
│  │     │  Extracts key facts from Wikipedia page         │     │   │
│  │     └─────────────────────────────────────────────────┘     │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                         HOP 2                                 │   │
│  │                                                               │   │
│  │  4. create_query_hop2                                        │   │
│  │     Input:  question + summary_1                             │   │
│  │     Output: optimized_query_2                                │   │
│  │     ├─────────────────────────────────────────────────┐     │   │
│  │     │  DSPy LM: "question, summary_1 -> query"        │     │   │
│  │     │  Generates second hop query based on findings   │     │   │
│  │     └─────────────────────────────────────────────────┘     │   │
│  │                                                               │   │
│  │  5. _search_and_scrape(optimized_query_2)                   │   │
│  │     ┌──────────────────────────────────────┐                │   │
│  │     │  SerperService                       │                │   │
│  │     │  • Query: "query site:wikipedia.org" │                │   │
│  │     │  • Returns: Top 5 Wikipedia URLs     │                │   │
│  │     └──────────────────────────────────────┘                │   │
│  │               ↓                                               │   │
│  │     ┌──────────────────────────────────────┐                │   │
│  │     │  FirecrawlService                    │                │   │
│  │     │  • Scrapes: Top URL                  │                │   │
│  │     │  • Returns: Markdown (max 15k chars) │                │   │
│  │     └──────────────────────────────────────┘                │   │
│  │     Output: hop2_context (full Wikipedia page)               │   │
│  │                                                               │   │
│  │  6. summarize2                                               │   │
│  │     Input:  question + summary_1 + hop2_context              │   │
│  │     Output: summary_2                                        │   │
│  │     ├─────────────────────────────────────────────────┐     │   │
│  │     │  DSPy LM: "question, prev_summary, context -> summary"│  │
│  │     │  Combines information from both hops            │     │   │
│  │     └─────────────────────────────────────────────────┘     │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    ANSWER GENERATION                          │   │
│  │                                                               │   │
│  │  7. generate_answer                                          │   │
│  │     Input:  question + summary_1 + summary_2                 │   │
│  │     Output: answer (minimal factoid)                         │   │
│  │     ├─────────────────────────────────────────────────┐     │   │
│  │     │  DSPy LM: GenerateAnswer signature              │     │   │
│  │     │  Produces minimal factoid with no elaboration   │     │   │
│  │     └─────────────────────────────────────────────────┘     │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Example

### Example Question: "What year was the director of Titanic born?"

```
┌──────────────────────────────────────────────────────────────────┐
│ INPUT: "What year was the director of Titanic born?"            │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ HOP 1 QUERY: "Titanic movie director"                           │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ SERPER SEARCH: "Titanic movie director site:wikipedia.org"      │
│ RESULTS:                                                         │
│   1. https://en.wikipedia.org/wiki/Titanic_(1997_film)          │
│   2. https://en.wikipedia.org/wiki/James_Cameron                │
│   3. ...                                                         │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ FIRECRAWL SCRAPE: https://en.wikipedia.org/wiki/Titanic_(...)   │
│ CONTENT (15,000 chars):                                          │
│   "Titanic is a 1997 American epic romantic disaster film       │
│    directed, written, co-produced and co-edited by James        │
│    Cameron. The film stars Leonardo DiCaprio and Kate           │
│    Winslet... [full Wikipedia article content]"                 │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ SUMMARY 1: "Titanic (1997) was directed by James Cameron"       │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ HOP 2 QUERY: "James Cameron birth year"                         │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ SERPER SEARCH: "James Cameron birth year site:wikipedia.org"    │
│ RESULTS:                                                         │
│   1. https://en.wikipedia.org/wiki/James_Cameron                │
│   2. ...                                                         │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ FIRECRAWL SCRAPE: https://en.wikipedia.org/wiki/James_Cameron   │
│ CONTENT (15,000 chars):                                          │
│   "James Francis Cameron (born August 16, 1954) is a Canadian   │
│    filmmaker and environmentalist. He is known for making       │
│    science fiction and epic films... [full article]"            │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ SUMMARY 2: "James Cameron was born on August 16, 1954"          │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ FINAL ANSWER: "1954"                                             │
└──────────────────────────────────────────────────────────────────┘
```

## Component Interaction

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  SerperService  │◄─────┤ HotpotMultiHop  ├─────►│ FirecrawlService│
│                 │      │    Predict      │      │                 │
│  • Google API   │      │                 │      │  • Web Scraping │
│  • Wikipedia    │      │  • DSPy Module  │      │  • Markdown     │
│    filtering    │      │  • Orchestrates │      │  • 15k limit    │
│  • 5 results    │      │    workflow     │      │                 │
│                 │      │                 │      │                 │
└─────────────────┘      └────────┬────────┘      └─────────────────┘
                                  │
                                  │
                         ┌────────▼────────┐
                         │                 │
                         │   DSPy LM       │
                         │                 │
                         │  • Query gen    │
                         │  • Summarize    │
                         │  • Answer gen   │
                         │                 │
                         └─────────────────┘
```

## Comparison with Previous Architecture

### OLD: ColBERTv2
```
Question → [ColBERT] → 7 passages (abstracts, ~1400 chars)
                              ↓
                         [Summarize]
                              ↓
Query 2 → [ColBERT] → 7 passages (abstracts, ~1400 chars)
                              ↓
                         [Summarize]
                              ↓
                          [Answer]
```

**Issues:**
- Limited context (only abstracts)
- Fixed retrieval (no query optimization)
- Total context: ~2,800 characters

### NEW: Serper + Firecrawl
```
Question → [Query Gen] → [Serper] → Top URL → [Firecrawl] → 15k chars
                                                    ↓
                                              [Summarize]
                                                    ↓
Summary 1 → [Query Gen] → [Serper] → Top URL → [Firecrawl] → 15k chars
                                                    ↓
                                              [Summarize]
                                                    ↓
                                               [Answer]
```

**Benefits:**
- Rich context (full pages)
- Optimized queries (DSPy query generation)
- Total context: ~30,000 characters
- Better retrieval (Google's ranking)

## API Call Flow

```
Time →

T0:  create_query_hop1(question)
     └─► DSPy LM call #1

T1:  serper.search(query_1)
     └─► Serper API call #1

T2:  firecrawl.scrape(url_1)
     └─► Firecrawl API call #1

T3:  summarize1(question, context_1)
     └─► DSPy LM call #2

T4:  create_query_hop2(question, summary_1)
     └─► DSPy LM call #3

T5:  serper.search(query_2)
     └─► Serper API call #2

T6:  firecrawl.scrape(url_2)
     └─► Firecrawl API call #2

T7:  summarize2(question, summary_1, context_2)
     └─► DSPy LM call #4

T8:  generate_answer(question, summary_1, summary_2)
     └─► DSPy LM call #5

Total: 5 LM calls + 2 Serper calls + 2 Firecrawl calls
```

## Error Handling Flow

```
_search_and_scrape(query)
    │
    ├─► serper.search(query)
    │   ├─► Success → list of URLs
    │   └─► Failure → Exception caught
    │                 └─► Return error message
    │
    ├─► No results? → Return "No search results" message
    │
    ├─► firecrawl.scrape(top_url)
    │   ├─► Success → Return markdown
    │   └─► Failure → Fallback to search snippet
    │
    └─► Any exception → Return error message with details
```

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Latency (total) | 10-20s | Depends on API response times |
| Latency (Serper) | 0.5-2s | Per search |
| Latency (Firecrawl) | 2-5s | Per scrape |
| Latency (LM) | 1-3s | Per call (GPT-4) |
| Context size | ~30k chars | 15k per hop |
| API calls | 9 total | 5 LM + 2 Serper + 2 Firecrawl |
| Cost per query | $0.01-0.05 | Varies by LM model |
