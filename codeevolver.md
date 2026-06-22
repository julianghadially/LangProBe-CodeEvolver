## PARENT_MODULE_PATH: "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"

## ARCHITECTURE TITLE: "6-hop retrieval with parallel 3-query claim extraction (GenerateClaimQueries) + 3 gap-fill hops (ExtractGapQuery), k=25, RRF merge, top-3 passage gap context"

## ARCHITECTURE SUMMARY:
HoverMultiHopPipeline wraps HoverMultiHop, a 6-hop multi-hop retrieval system. The architecture begins with a single LM call (`GenerateClaimQueries`) that reads the raw claim and generates 3 distinct targeted queries simultaneously — one per expected Wikipedia article, with explicit guidance for sports seasons ("YEAR-YY TeamName season"), films ("(film)" disambiguation), songs ("(song)" disambiguation), and persons (full name). Hops 1–3 execute these targeted queries (k=25 each). Hops 4–6 use `ExtractGapQuery` to identify and fill retrieval gaps using top-3 passages per hop (350 chars each) for richer gap analysis context. A dedup guard prevents repeated queries in hops 4–6. After all six hops, candidates are merged via Reciprocal Rank Fusion (RRF, k=60) capped at 21 unique documents.

## ARCHITECTURE DESCRIPTION:
HoverMultiHopPipeline is the top-level entry point. It configures the language model (gpt-5.4-nano with low reasoning effort) and a CountingRM-wrapped ColBERTv2 retriever, then delegates all retrieval logic to HoverMultiHop via a dspy.context call.

HoverMultiHop performs six retrieval hops with k=25 candidates each, using 4 LM calls total:

- **GenerateClaimQueries (1 LM call)**: A ChainOfThought module over the `GenerateClaimQueries` signature reads the raw claim and outputs 3 distinct search queries simultaneously. Explicit guidance covers: named persons (full name), sports seasons ("YEAR-YY TeamName season" format), films/songs (title + "(film)"/"(song)" disambiguation), and described/implied entities. All 3 queries must target different Wikipedia articles and are kept short (1–6 words).

- **Hops 1–3**: The 3 queries from `GenerateClaimQueries` are each sent to the ColBERTv2 retriever (k=25). These targeted hops maximize recall for the ~3 Wikipedia articles the claim typically requires.

- **Hop 4 (ExtractGapQuery, 1 LM call)**: A ChainOfThought module over the `ExtractGapQuery` signature receives the claim, top-7 unique retrieved titles, top-3 passages per hop (truncated to 350 chars each) from hops 1–3, and a semicolon-separated `already_searched` string. Outputs a `missing_entity` field (forcing explicit intermediate reasoning) then a `query` field.

- **Hop 5 (ExtractGapQuery, 1 LM call)**: Same as hop 4 but with top-3 passages from hops 1–4 and the updated `already_searched` string.

- **Hop 6 (ExtractGapQuery, 1 LM call)**: Same pattern with top-3 passages from hops 1–5. A `_is_duplicate_query` guard (normalised lowercase comparison) prevents executing the retrieval if hop 6's query matches any prior query.

**Merge — Reciprocal Rank Fusion (RRF, k=60)**: Each document across all hop lists (up to 6) is scored by the sum of 1/(rank + 60) across all hops where it appears. Documents are ranked by descending RRF score and the top 21 unique documents are returned. RRF naturally up-weights documents that rank highly across multiple hops and rescues high-value docs ranked 4+ that round-robin would have dropped.

**Gap analysis context**: `_get_key_passages` now returns top-3 docs per hop (350 chars each) instead of only the top-1 doc at 500 chars, giving ExtractGapQuery 3× more material to identify missing entity names. `_get_retrieved_titles` checks top-7 docs per hop (up from 5) for a broader dedup and coverage signal.
