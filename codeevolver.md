## PARENT_MODULE_PATH: "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"

## ARCHITECTURE TITLE: "9-hop retrieval with 5-query claim extraction (3 explicit + 1 inferred + 1 description-based, GenerateClaimQueries) + 4 gap-fill hops (ExtractGapQuery), fuzzy dedup on all gap hops (6–9), top_n=3 passages, k=25, inverse-rank score-based merge"

## ARCHITECTURE SUMMARY:
HoverMultiHopPipeline wraps HoverMultiHop, a 9-hop multi-hop retrieval system. The architecture begins with a single LM call (`GenerateClaimQueries`) that reads the raw claim and generates 5 distinct targeted queries simultaneously — 3 for entities explicitly named or described in the claim, a 4th for an implicit or inferred entity needed for multi-hop reasoning, and a 5th description-based fallback that uses exact descriptive phrases from the claim to find entities not directly named (e.g., "James Mitchum 1975 film" when claim says "the 1975 film starring James Mitchum"). Hops 1–5 execute these targeted queries (k=25 each). Hops 6–9 use `ExtractGapQuery` to identify and fill remaining retrieval gaps using top-3 passages per hop (800 chars each). A fuzzy dedup guard (difflib SequenceMatcher, threshold 0.85) prevents near-duplicate queries across all four gap-fill hops. After all hops, non-empty candidates are merged via inverse-rank score-based merging (each doc scores sum of 1/(rank+1) across all hops) capped at 21 unique documents.

## ARCHITECTURE DESCRIPTION:
HoverMultiHopPipeline is the top-level entry point. It configures the language model (gpt-5.4-nano with low reasoning effort) and a CountingRM-wrapped ColBERTv2 retriever, then delegates all retrieval logic to HoverMultiHop via a dspy.context call.

HoverMultiHop performs up to nine retrieval hops with k=25 candidates each, using 6 LM calls total:

- **GenerateClaimQueries (1 LM call)**: A ChainOfThought module over the `GenerateClaimQueries` signature reads the raw claim and outputs 5 distinct search queries simultaneously. Queries 1–3 target explicitly named or described entities (named persons use full name; films/songs get "(film)"/"(song)" disambiguation; sports seasons use "YEAR-YY TeamName season" format). Query 4 targets an IMPLICIT or INFERRED entity — not directly named in the claim but needed for multi-hop reasoning. Query 5 is a DESCRIPTION-BASED FALLBACK that uses exact words or phrases from the claim to search for an entity described but not directly named (e.g., "James Mitchum 1975 film" rather than guessing "Moonrunners"). Additional guidance for q3 or q4 covers: broader category/overview articles when claim uses phrases like "in this religion", "this culture", or "[place] has several [things]"; famous adaptations of compositions; and overview articles for places with multiple sub-topics. All 5 queries must target different Wikipedia articles and are kept short (1–6 words).

- **Hops 1–5**: The 5 queries from `GenerateClaimQueries` are each sent to the ColBERTv2 retriever (k=25).

- **Hops 6–9 (ExtractGapQuery, 4 LM calls)**: Each gap-fill hop follows the same pattern — a ChainOfThought module over `ExtractGapQuery` receives the claim, top-7 unique retrieved titles from all prior hops, top-3 passages per hop (truncated to 800 chars each) as key context, and a semicolon-separated `already_searched` string. Outputs a `missing_entity` field then a `query` field. A fuzzy dedup guard prevents executing the retrieval if the gap query duplicates any prior query. Each successive hop expands context to include all previous hop results.

**ExtractGapQuery patterns**: In addition to entity relationship patterns (founded-by, stars, owned-by, written-by, directed-by), the module now recognises broader-category gaps, composition adaptations, place overview articles, and TV show/film inspiration patterns (e.g., "TV show [X] was inspired by film [Y]" → search for "[X]" or "[Y]" directly).

**Merge — Inverse-Rank Score-Based**: Each unique document is scored by summing 1/(rank+1) across all hops that returned it. Documents appearing in multiple hops and ranked higher accumulate higher scores. The top-21 by score are returned, ensuring docs appearing in multiple hops are prioritized over single-hop retrievals, and higher-ranked docs within each hop are weighted more heavily.

**Gap analysis context**: `_get_key_passages` returns the top-3 docs per hop at 800 chars each (increased from 2 to capture more article context). `_get_retrieved_titles` checks top-7 docs per hop. `_is_duplicate_query` uses fuzzy matching (SequenceMatcher ratio > 0.85) across all gap-fill hops 6–9.
