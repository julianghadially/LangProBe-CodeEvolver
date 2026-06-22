## PARENT_MODULE_PATH: "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"

## ARCHITECTURE TITLE: "7-hop retrieval with 4-query claim extraction (3 explicit + 1 inferred, GenerateClaimQueries) + 3 gap-fill hops (ExtractGapQuery), fuzzy dedup on all gap hops (5–7), 800-char passages, top_n=2, k=25, round-robin merge"

## ARCHITECTURE SUMMARY:
HoverMultiHopPipeline wraps HoverMultiHop, a 7-hop multi-hop retrieval system. The architecture begins with a single LM call (`GenerateClaimQueries`) that reads the raw claim and generates 4 distinct targeted queries simultaneously — 3 for entities explicitly named or described in the claim (including broader category/overview articles when the claim uses phrases like "in this religion" or "[place] has several [things]"), and a 4th for an implicit or inferred entity needed for multi-hop reasoning. Hops 1–4 execute these targeted queries (k=25 each). Hops 5–7 use `ExtractGapQuery` to identify and fill remaining retrieval gaps using top-2 passages per hop (800 chars each). A fuzzy dedup guard (difflib SequenceMatcher, threshold 0.85) prevents near-duplicate queries across all three gap-fill hops (5, 6, and 7). After all hops, non-empty candidates are merged via round-robin interleaving capped at 21 unique documents.

## ARCHITECTURE DESCRIPTION:
HoverMultiHopPipeline is the top-level entry point. It configures the language model (gpt-5.4-nano with low reasoning effort) and a CountingRM-wrapped ColBERTv2 retriever, then delegates all retrieval logic to HoverMultiHop via a dspy.context call.

HoverMultiHop performs up to seven retrieval hops with k=25 candidates each, using 5 LM calls total:

- **GenerateClaimQueries (1 LM call)**: A ChainOfThought module over the `GenerateClaimQueries` signature reads the raw claim and outputs 4 distinct search queries simultaneously. Queries 1–3 target explicitly named or described entities (named persons use full name; films/songs get "(film)"/"(song)" disambiguation; sports seasons use "YEAR-YY TeamName season" format). Query 4 targets an IMPLICIT or INFERRED entity — not directly named in the claim but needed for multi-hop reasoning. Additional guidance for q3 or q4 covers: broader category/overview articles when claim uses phrases like "in this religion", "this culture", or "[place] has several [things]"; famous adaptations of compositions (e.g., "Stranger in Paradise (song)" from "Polovtsian Dances"); and overview articles for places with multiple sub-topics. All 4 queries must target different Wikipedia articles and are kept short (1–6 words).

- **Hops 1–4**: The 4 queries from `GenerateClaimQueries` are each sent to the ColBERTv2 retriever (k=25).

- **Hop 5 (ExtractGapQuery, 1 LM call)**: A ChainOfThought module over the `ExtractGapQuery` signature receives the claim, top-7 unique retrieved titles from hops 1–4, top-2 passages per hop (truncated to 800 chars each) as key context, and a semicolon-separated `already_searched` string. Outputs a `missing_entity` field then a `query` field. A fuzzy dedup guard prevents executing the retrieval if hop 5's query duplicates any of q1–q4.

- **Hop 6 (ExtractGapQuery, 1 LM call)**: Same pattern but with context from hops 1–5. Same fuzzy dedup guard applied against all prior queries (q1–q4 and hop5_query).

- **Hop 7 (ExtractGapQuery, 1 LM call)**: Same pattern but with context from hops 1–6. Same fuzzy dedup guard applied against all queries used up to and including hop 6.

**ExtractGapQuery patterns**: In addition to entity relationship patterns (founded-by, stars, owned-by, written-by, directed-by), the module now recognises broader-category gaps: when all retrieved articles are sub-topics of a broader concept X, it searches for the overview "X" article; when a composition has a well-known adaptation, it searches for the adapted work; when multiple sub-topic articles of a place are retrieved but no overview exists, it searches for the "[Things] of [Place]" overview article.

**Merge — Round-Robin Interleaving**: Candidates from all non-empty hop lists (up to 7) are interleaved round-robin with title-based deduplication. The top 21 unique documents are returned.

**Gap analysis context**: `_get_key_passages` returns the top-2 docs per hop at 800 chars each (increased from 500 to capture more article context such as cast lists that appear later in text). `_get_retrieved_titles` checks top-7 docs per hop. `_is_duplicate_query` uses fuzzy matching (SequenceMatcher ratio > 0.85) across all gap-fill hops 5–7.
