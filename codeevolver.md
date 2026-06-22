## PARENT_MODULE_PATH: "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"

## ARCHITECTURE TITLE: "6-hop retrieval with parallel 3-query claim extraction (GenerateClaimQueries) + 3 gap-fill hops (ExtractGapQuery), k=25, RRF merge, missing_entity intermediate reasoning"

## ARCHITECTURE SUMMARY:
HoverMultiHopPipeline wraps HoverMultiHop, a 6-hop multi-hop retrieval system. The architecture begins with a single LM call (`GenerateClaimQueries`) that reads the raw claim and generates 3 distinct targeted queries simultaneously — one per expected Wikipedia article, with explicit guidance for sports seasons ("YEAR-YY TeamName season"), films ("(film)" disambiguation), songs ("(song)" disambiguation), and persons (full name). Hops 1–3 execute these targeted queries (k=25 each). Hops 4–6 use `ExtractGapQuery` to identify and fill retrieval gaps; `ExtractGapQuery` now includes a `missing_entity` output field that forces explicit intermediate reasoning before generating the final query. A dedup guard prevents repeated queries in hops 4–6. After all six hops, candidates are merged via Reciprocal Rank Fusion (RRF) capped at 21 unique documents.

## ARCHITECTURE DESCRIPTION:
HoverMultiHopPipeline is the top-level entry point. It configures the language model (gpt-5.4-nano with low reasoning effort) and a CountingRM-wrapped ColBERTv2 retriever, then delegates all retrieval logic to HoverMultiHop via a dspy.context call.

HoverMultiHop performs six retrieval hops with k=25 candidates each, using 4 LM calls total:

- **GenerateClaimQueries (1 LM call)**: A ChainOfThought module over the `GenerateClaimQueries` signature reads the raw claim and outputs 3 distinct search queries simultaneously. Improved docstring now explicitly guides: named persons use their full name directly; sports season articles use exact "YEAR-YY TeamName season" format; films/songs use title + "(film)"/"(song)" disambiguation when needed; described/implied entities are inferred. A critical constraint forbids queries for general concepts or locations. All 3 queries must target different Wikipedia articles and are kept short (1–6 words).

- **Hops 1–3**: The 3 queries from `GenerateClaimQueries` are each sent to the ColBERTv2 retriever (k=25). These targeted hops maximize recall for the ~3 Wikipedia articles the claim typically requires.

- **Hop 4 (ExtractGapQuery, 1 LM call)**: A ChainOfThought module over the improved `ExtractGapQuery` signature receives the claim, top-3 passages from each of hops 1–3, and a semicolon-separated `already_searched` string. Now outputs a `missing_entity` field first (forcing the model to name the specific missing Wikipedia article title), then a `query` field. Key rules guide sports seasons, films, songs, and persons. Forbids queries matching `already_searched` or targeting general concepts.

- **Hop 5 (ExtractGapQuery, 1 LM call)**: Same as hop 4 but receives top-2 passages from hops 1–4 and the updated `already_searched` string.

- **Hop 6 (ExtractGapQuery, 1 LM call)**: Same pattern with top-2 passages from hops 1–5. A `_is_duplicate_query` guard (normalised lowercase comparison) prevents executing the retrieval if hop 6's query matches any prior query.

**Merge — Reciprocal Rank Fusion (RRF)**: Replaces the previous interleaved round-robin. Each document across all hop lists (up to 6) is scored by the sum of 1/(60+rank+1) across all hops where it appears. Documents are ranked by descending RRF score and the top 21 unique documents are returned. RRF naturally up-weights documents that rank highly across multiple hops, improving precision over round-robin.
