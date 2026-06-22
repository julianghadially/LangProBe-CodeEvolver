## PARENT_MODULE_PATH: "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"

## ARCHITECTURE TITLE: "5-hop retrieval with parallel 3-query claim extraction (GenerateClaimQueries) + gap-fill hops (ExtractGapQuery), k=25, interleaved round-robin deduplication"

## ARCHITECTURE SUMMARY:
HoverMultiHopPipeline wraps HoverMultiHop, a 5-hop multi-hop retrieval system. Instead of sequentially deriving queries from retrieved passages, the new architecture begins with a single LM call (`GenerateClaimQueries`) that reads the raw claim and generates 3 distinct targeted queries simultaneously — one per expected Wikipedia article, covering both explicitly named entities and implied/described ones. Hops 1–3 execute these 3 targeted queries (k=25 each). Hops 4–5 use `ExtractGapQuery` to identify and fill retrieval gaps by inspecting already-retrieved passages and an explicit `already_searched` field that prevents repeated queries. After all five hops, candidates are merged via interleaved round-robin deduplication capped at 21 unique documents.

## ARCHITECTURE DESCRIPTION:
HoverMultiHopPipeline is the top-level entry point. It configures the language model (gpt-5.4-nano with low reasoning effort) and a CountingRM-wrapped ColBERTv2 retriever, then delegates all retrieval logic to HoverMultiHop via a dspy.context call.

HoverMultiHop performs five retrieval hops with k=25 candidates each, using 3 LM calls total:

- **GenerateClaimQueries (1 LM call)**: A ChainOfThought module over the `GenerateClaimQueries` signature reads the raw claim and outputs 3 distinct search queries simultaneously. For explicitly named entities (persons, films, shows, places), it uses the name directly. For described or implied entities, it infers the most likely Wikipedia article title. All 3 queries must target different Wikipedia articles and are kept short (1–6 words), similar to Wikipedia article titles. This single upfront call replaces the previous pattern of deriving queries one-by-one from retrieved passages, which frequently missed entities explicitly named in the claim.

- **Hops 1–3**: The 3 queries from `GenerateClaimQueries` are each sent to the ColBERTv2 retriever (k=25). These targeted hops maximize recall for the ~3 Wikipedia articles the claim typically requires.

- **Hop 4 (ExtractGapQuery, 1 LM call)**: A ChainOfThought module over `ExtractGapQuery` receives the claim, the top-3 passages from each of hops 1–3, and a semicolon-separated `already_searched` string listing all prior queries. It prioritizes claim entities not yet surfaced in retrieved passages, then falls back to passage hints. It outputs a single short query for the most important missing article without repeating prior queries.

- **Hop 5 (ExtractGapQuery, 1 LM call)**: Same as hop 4 but receives top-2 passages from each of hops 1–4 and the updated `already_searched` string including hop 4's query.

**Merge — interleaved round-robin deduplication**: Round-robin across all 5 hop lists (prioritizing hops 1–3), adding each document if its normalised title (before " | ") has not been seen. Stops at 21 unique documents. The final `dspy.Prediction(retrieved_docs=final_docs[:21])` is returned upstream to the evaluator.
