## PARENT_MODULE_PATH: "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"

## ARCHITECTURE TITLE: "9-hop retrieval with 5-query claim extraction (GenerateClaimQueries) + 2 dual-gap rounds (ExtractGapQueries outputs 2 queries each), fuzzy dedup, top_n=4 passages (1200 chars), k=25, inverse-rank score-based merge"

## ARCHITECTURE SUMMARY:
HoverMultiHopPipeline wraps HoverMultiHop, a 9-hop multi-hop retrieval system. The architecture begins with a single LM call (`GenerateClaimQueries`) that reads the raw claim and generates 5 distinct targeted queries simultaneously — 3 for entities explicitly named or described in the claim, a 4th for an implicit or inferred entity, and a 5th description-based fallback. Hops 1–5 execute these targeted queries (k=25 each). Hops 6–9 use `ExtractGapQueries` in 2 dual-gap rounds: each call now outputs 2 queries for 2 different missing entities simultaneously, leveraging both retrieved passages and the model's own world knowledge. This eliminates duplicate gap rounds — instead of 4 sequential single-query rounds that often repeated, 2 dual-query rounds produce 4 targeted gap queries per run. A fuzzy dedup guard (difflib SequenceMatcher, threshold 0.85) prevents near-duplicate queries. Passages are now truncated at 1200 chars (up from 800) and top-10 titles per hop are tracked (up from 7). All non-empty hop candidates are merged via inverse-rank score-based merging capped at 21 unique documents.

## ARCHITECTURE DESCRIPTION:
HoverMultiHopPipeline is the top-level entry point. It configures the language model (gpt-5.4-nano with low reasoning effort) and a CountingRM-wrapped ColBERTv2 retriever, then delegates all retrieval logic to HoverMultiHop via a dspy.context call.

HoverMultiHop performs up to nine retrieval hops with k=25 candidates each, using 3 LM calls total (1 for GenerateClaimQueries + 2 for ExtractGapQueries rounds):

- **GenerateClaimQueries (1 LM call)**: A ChainOfThought module over the `GenerateClaimQueries` signature reads the raw claim and outputs 5 distinct search queries simultaneously. Queries 1–3 target explicitly named or described entities. Query 4 targets an IMPLICIT or INFERRED entity needed for multi-hop reasoning. Query 5 is a DESCRIPTION-BASED FALLBACK using exact words or phrases from the claim.

- **Hops 1–5**: The 5 queries from `GenerateClaimQueries` are each sent to the ColBERTv2 retriever (k=25).

- **Hops 6–9 (ExtractGapQueries, 2 LM calls)**: Two dual-gap rounds replace the previous 4 single-gap rounds. Each `ExtractGapQueries` call receives the claim, top-10 unique retrieved titles, top-3/top-4 passages per hop (truncated to 1200 chars each), and a semicolon-separated `already_searched` string. It outputs TWO queries for TWO different missing entities: `missing_entity_1`/`query1` and `missing_entity_2`/`query2`. The signature instructs the model to use its own world knowledge (not just passages) to identify what's missing — e.g., if "Spaceballs" is retrieved and the claim needs the starring actor, the model should think "John Candy" or "Bill Pullman". Round 1 (hops 6–7) uses context from hops 1–5. Round 2 (hops 8–9) uses context from all prior hops (1–7) with top_n=4 passages. A fuzzy dedup guard prevents executing retrieval for near-duplicate queries.

**ExtractGapQueries patterns**: In addition to entity relationship patterns (founded-by, stars, owned-by, directed-by), the module covers: award ceremonies ("NTH Award Name"), regional airlines, TV shows inspired by films, music adaptations, and world-knowledge-based inference when passages are incomplete.

**Merge — Inverse-Rank Score-Based**: Each unique document is scored by summing 1/(rank+1) across all hops that returned it. The top-21 by score are returned.

**Gap analysis context**: `_get_key_passages` truncates at 1200 chars (increased from 800). `_get_retrieved_titles` checks top-10 docs per hop (increased from 7). `_is_duplicate_query` uses fuzzy matching (SequenceMatcher ratio > 0.85).
