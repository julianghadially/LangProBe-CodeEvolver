## PARENT_MODULE_PATH: "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"

## ARCHITECTURE TITLE: "7-hop retrieval with 4-query claim extraction (3 explicit + 1 inferred, GenerateClaimQueries) + 3 gap-fill hops (ExtractGapQuery), fuzzy dedup, top_n=2 passages, k=25, round-robin merge"

## ARCHITECTURE SUMMARY:
HoverMultiHopPipeline wraps HoverMultiHop, a 7-hop multi-hop retrieval system. The architecture begins with a single LM call (`GenerateClaimQueries`) that reads the raw claim and generates 4 distinct targeted queries simultaneously — 3 for entities explicitly named or described in the claim, and a 4th for an implicit or inferred entity needed for multi-hop reasoning. Hops 1–4 execute these targeted queries (k=25 each). Hops 5–7 use `ExtractGapQuery` to identify and fill remaining retrieval gaps using top-2 passages per hop (500 chars each). A fuzzy dedup guard (difflib SequenceMatcher, threshold 0.85) prevents near-duplicate queries in hops 6 and 7. After all hops, candidates are merged via round-robin interleaving capped at 21 unique documents.

## ARCHITECTURE DESCRIPTION:
HoverMultiHopPipeline is the top-level entry point. It configures the language model (gpt-5.4-nano with low reasoning effort) and a CountingRM-wrapped ColBERTv2 retriever, then delegates all retrieval logic to HoverMultiHop via a dspy.context call.

HoverMultiHop performs up to seven retrieval hops with k=25 candidates each, using 5 LM calls total:

- **GenerateClaimQueries (1 LM call)**: A ChainOfThought module over the `GenerateClaimQueries` signature reads the raw claim and outputs 4 distinct search queries simultaneously. Queries 1–3 target explicitly named or described entities (named persons use full name; films/songs get "(film)"/"(song)" disambiguation; sports seasons use "YEAR-YY TeamName season" format). Query 4 targets an IMPLICIT or INFERRED entity — not directly named in the claim but needed for multi-hop reasoning (e.g., film studio if "X directed Y", a co-star, a notable person associated with a university). All 4 queries must target different Wikipedia articles and are kept short (1–6 words).

- **Hops 1–4**: The 4 queries from `GenerateClaimQueries` are each sent to the ColBERTv2 retriever (k=25). The 4th targeted hop on the inferred entity improves coverage for implicit multi-hop connections.

- **Hop 5 (ExtractGapQuery, 1 LM call)**: A ChainOfThought module over the `ExtractGapQuery` signature receives the claim, top-7 unique retrieved titles from hops 1–4, top-2 passages per hop (truncated to 500 chars each) as key context, and a semicolon-separated `already_searched` string. Outputs a `missing_entity` field then a `query` field.

- **Hop 6 (ExtractGapQuery, 1 LM call)**: Same pattern but with context from hops 1–5. A `_is_duplicate_query` guard (fuzzy matching via difflib SequenceMatcher with threshold 0.85, plus exact normalised lowercase comparison) prevents executing the retrieval if hop 6's query matches or closely resembles any prior query. This catches near-duplicates from formatting variations (em-dash vs hyphen) and minor typos.

- **Hop 7 (ExtractGapQuery, 1 LM call)**: Same pattern but with context from hops 1–6. Same fuzzy dedup guard applied against all queries used up to and including hop 6. This extra hop resolves "chained multi-hop" failures where an intermediate entity is found in hop 6 but the entity connected to it still needs a final retrieval.

**Merge — Round-Robin Interleaving**: Candidates from all hop lists (up to 7) are interleaved round-robin with title-based deduplication. The top 21 unique documents are returned. This prioritizes breadth across all targeted and gap-fill hops equally.

**Gap analysis context**: `_get_key_passages` returns the top-2 docs per hop at 500 chars each, giving the gap analysis LM calls richer context to identify missing entities. `_get_retrieved_titles` checks top-7 docs per hop for broad dedup and coverage signaling. `_is_duplicate_query` uses fuzzy matching (SequenceMatcher ratio > 0.85) to catch near-duplicates like formatting variations and minor typos in addition to exact normalised matches.
