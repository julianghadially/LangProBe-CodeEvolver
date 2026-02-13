# HotPot Multi-Hop QA: Serper + Firecrawl Migration

## Overview

This document explains the migration from ColBERTv2 retrieval to a two-stage **Serper + Firecrawl** architecture for the HotPot Multi-Hop QA system.

## Problem Statement

The original implementation using ColBERTv2 retrieval was scoring **0.0** on HotPot QA evaluations. The issue was that Wikipedia abstracts (returned by ColBERTv2) didn't contain enough detailed information to answer complex multi-hop questions that require reasoning across multiple facts.

## Solution: Two-Stage Search Architecture

### Stage 1: Serper Web Search
- **Purpose**: Find the most relevant Wikipedia pages for each reasoning hop
- **Implementation**: Uses `SerperService` to search Google with `site:wikipedia.org` constraint
- **Benefit**: Leverages Google's ranking to find the most relevant articles

### Stage 2: Firecrawl Page Scraping
- **Purpose**: Extract full page content from the top search result
- **Implementation**: Uses `FirecrawlService` to scrape and convert Wikipedia pages to markdown
- **Benefit**: Provides rich, complete context instead of just abstracts
- **Constraint**: Maximum 15,000 characters per page (truncated if longer)

## Key Changes to `hotpot_program.py`

### 1. New Imports
```python
from services import SerperService, FirecrawlService
```

### 2. Modified `GenerateAnswer` Signature
**Before:**
```python
answer = dspy.OutputField(desc="The answer itself and nothing else")
```

**After:**
```python
answer = dspy.OutputField(desc="Only the minimal factoid answer with NO elaboration, explanation, or additional text. Just the answer itself.")
```

**Reason**: Emphasizes exact-match answers without elaboration to improve scoring.

### 3. Replaced `dspy.Retrieve` with `_search_and_scrape()`

**Before:**
- Used `dspy.Retrieve(k=7)` to get 7 passages from ColBERTv2
- Returned short abstracts from Wikipedia

**After:**
- Uses `_search_and_scrape()` method for each hop
- Searches Wikipedia via Serper
- Scrapes the top result with Firecrawl
- Returns full page markdown (up to 15,000 chars)

### 4. Modified Signature Field Names

**Before:**
```python
self.summarize1 = dspy.Predict("question,passages->summary")
self.summarize2 = dspy.Predict("question,context,passages->summary")
```

**After:**
```python
self.summarize1 = dspy.Predict("question,context->summary")
self.summarize2 = dspy.Predict("question,previous_summary,context->summary")
```

**Reason**: Changed from `passages` (list of short texts) to `context` (single markdown document).

### 5. Added Query Generation for Hop 1

**Before:**
- Used the original question directly for first hop retrieval

**After:**
```python
self.create_query_hop1 = dspy.Predict("question->query")
```

**Reason**: Allows the LM to reformulate the question into a better search query.

## Architecture Flow

```
User Question
    |
    v
[HOP 1: Query Generation]
    |
    v
[Serper Search] --> Wikipedia URLs
    |
    v
[Firecrawl Scrape] --> Full page markdown
    |
    v
[Summarize Hop 1] --> Summary 1
    |
    v
[HOP 2: Query Generation] (uses Summary 1)
    |
    v
[Serper Search] --> Wikipedia URLs
    |
    v
[Firecrawl Scrape] --> Full page markdown
    |
    v
[Summarize Hop 2] --> Summary 2
    |
    v
[Generate Answer] --> Minimal factoid answer
```

## Resource Constraints

- **Max 2 Serper searches**: One per hop
- **Max 2 page scrapes**: One per hop (top result only)
- **Max content per page**: 15,000 characters (truncated)

## Benefits

1. **Richer Context**: Full Wikipedia pages vs. short abstracts
2. **Targeted Retrieval**: Google's ranking finds the most relevant pages
3. **Better Coverage**: 15,000 chars per page vs. ~200 chars per passage
4. **Fewer API Calls**: 2 searches + 2 scrapes vs. 14 ColBERT retrievals
5. **Improved Accuracy**: Full pages enable better multi-hop reasoning

## Expected Impact

The richer context from full Wikipedia pages should enable the LLM to:
- Extract more specific facts needed for multi-hop reasoning
- Make better connections between information across hops
- Generate exact-match answers instead of approximate responses
- Improve from 0.0 to competitive scores on HotPot QA metrics

## Usage

```python
import dspy
from langProPlus.hotpotGEPA.hotpot_program import HotpotMultiHopPredict

# Initialize with Serper + Firecrawl services
program = HotpotMultiHopPredict()

# Run multi-hop reasoning
result = program(question="Your multi-hop question here")
print(result.answer)
```

## Requirements

- `SERPER_KEY` environment variable (for Google Search API)
- `FIRECRAWL_KEY` environment variable (for page scraping)
- DSPy configured with an LM (e.g., OpenAI, Anthropic)

## Error Handling

The implementation includes graceful fallbacks:
- If no search results: Returns error message
- If scraping fails: Falls back to search snippet
- If any exception: Returns descriptive error message

## Testing

See `test_hotpot_serper_firecrawl.py` for a simple test script.

## Pipeline Compatibility

The `HotpotMultiHopPredictPipeline` class no longer needs to set up ColBERTv2:

**Before:**
```python
self.rm = dspy.ColBERTv2(url=COLBERT_URL)
with dspy.context(rm=self.rm):
    return self.program(question=question)
```

**After:**
```python
# No retrieval model needed - services are initialized in HotpotMultiHopPredict
return self.program(question=question)
```

The pipeline can be simplified since retrieval is now handled internally by the program.
