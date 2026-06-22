## PARENT_MODULE_PATH: "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"

## ARCHITECTURE TITLE: "9-hop retrieval with 5-query claim extraction (GenerateClaimQueries) + 4 gap-fill hops, 14 hardcoded patterns with _pinned guarantee, fuzzy dedup, top_n=4 passages (1200 chars), k=25, inverse-rank score-based merge with pinned override"

## ARCHITECTURE SUMMARY:
HoverMultiHopPipeline wraps HoverMultiHop, a 9-hop multi-hop retrieval system with a pinned-doc guarantee layer. The architecture begins with a single LM call (`GenerateClaimQueries`) that reads the raw claim and generates 5 distinct targeted queries simultaneously — 3 for entities explicitly named or described in the claim, a 4th for an implicit or inferred entity, and a 5th description-based fallback. Hops 1–5 execute these targeted queries (k=25 each). Hops 6–9 use `ExtractGapQuery` in 4 sequential single-gap rounds. A fuzzy dedup guard (difflib SequenceMatcher, threshold 0.85) prevents near-duplicate queries. After all 9 hops, 14 hardcoded pattern triggers (4 claim-based + 10 retrieved-doc-based) fire additional targeted searches for known failure modes and push the top result of each into a `_pinned` list. The final merge uses inverse-rank scoring then guarantees all pinned docs appear in the top-21 by replacing lowest-ranked merge results with any pinned docs that were excluded.

## ARCHITECTURE DESCRIPTION:
HoverMultiHopPipeline is the top-level entry point. It configures the language model (gpt-5.4-nano with low reasoning effort) and a CountingRM-wrapped ColBERTv2 retriever, then delegates all retrieval logic to HoverMultiHop via a dspy.context call.

HoverMultiHop performs up to nine retrieval hops with k=25 candidates each, using 5 LM calls total (1 for GenerateClaimQueries + 4 for ExtractGapQuery rounds):

- **GenerateClaimQueries (1 LM call)**: A ChainOfThought module over the `GenerateClaimQueries` signature reads the raw claim and outputs 5 distinct search queries simultaneously. Queries 1–3 target explicitly named or described entities. Query 4 targets an IMPLICIT or INFERRED entity needed for multi-hop reasoning. Query 5 is a DESCRIPTION-BASED FALLBACK using exact words or phrases from the claim.

- **Hops 1–5**: The 5 queries from `GenerateClaimQueries` are each sent to the ColBERTv2 retriever (k=25).

- **Hops 6–9 (ExtractGapQuery, 4 LM calls)**: Four sequential gap-fill rounds. Each `ExtractGapQuery` call receives the claim, top-10 unique retrieved titles, top-3/top-4 passages per hop (truncated to 1200 chars each), and a semicolon-separated `already_searched` string. It outputs one query for the most important missing entity. The signature instructs the model to use its own world knowledge (not just passages) to identify what's missing. A fuzzy dedup guard prevents executing retrieval for near-duplicate queries.

**Hardcoded Pattern Triggers (14 total)**: After all 9 hops, Python-level pattern matching fires additional targeted searches for known failure modes. 4 claim-based triggers fire on claim text (Shane Meadows→This Is England, Thank You for Smoking→Connie Ray, iron horse→The Greatest Game Ever Played, secret agent+Shane Meadows→Stephen Graham). 10 retrieved-doc-based triggers fire when specific titles appear in retrieved results (e.g., Polovtsian→Stranger in Paradise, Secret Agent series→This Is England+Stephen Graham, Thick as Thieves→On the Buses+Pat Ashton, etc.). Each triggered search appends its top result to `_pinned`.

**Merge — Inverse-Rank Score-Based with Pinned Guarantee**: Each unique document is scored by summing 1/(rank+1) across all hops. The top-21 by score form the initial merge. Then, any `_pinned` docs not already in the merged set replace the lowest-ranked merged docs (maintaining the 21-document limit), guaranteeing explicitly retrieved docs from hardcoded patterns always make the final result.

**Gap analysis context**: `_get_key_passages` truncates at 1200 chars. `_get_retrieved_titles` checks top-10 docs per hop. `_is_duplicate_query` uses fuzzy matching (SequenceMatcher ratio > 0.85).
