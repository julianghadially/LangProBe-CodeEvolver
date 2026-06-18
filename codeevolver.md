PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval_with_resource_penalty_and_feedback

## ARCHITECTURE TITLE: 4-Hop Entity-Focused Retrieval with Round-Robin Interleaving, Gap Analysis, and Entity-Aware Summarization (k=15)

## ARCHITECTURE SUMMARY:
`HoverMultiHopPipeline` (`langProBe/hover/hover_pipeline.py`) is the evaluator-facing entry point. It wires together an LM (`openai/gpt-5.4-nano`, reasoning_effort="low"), a `CountingRM`-wrapped ColBERTv2 retriever, and the core `HoverMultiHop` program, then tags each prediction with the search count for downstream metric use.

`HoverMultiHop` (`langProBe/hover/hover_program.py`) implements a 4-hop entity-focused retrieval loop. Each hop uses a dedicated `dspy.Signature` with explicit instructions to identify and query for specific named entities from the claim that have not yet been retrieved. Retrieval uses k=15 per hop for broader coverage. Summarization uses `Summarize1Signature` and `Summarize2Signature` — both explicitly instruct the LM to identify which claim entities have been found and which are still missing. A fourth "gap analysis" hop (`QueryHop4GapSignature`) examines the titles of already-retrieved documents and generates a targeted query for whichever claim entity is still missing. All results are merged via round-robin interleaving (`_interleave_and_deduplicate`) so each hop contributes roughly equally (~5 docs) to the final 21 passages, preventing any single hop from dominating the budget.

The metric (`langProBe/hover/hover_utils.py`) scores predictions as retrieval success (all gold supporting-document titles found within the hard cap of 21 returned docs) minus a soft penalty for search queries beyond a 2-query free budget (0.002 per extra search), returning a `ScoreWithFeedback` object for reflective optimization.

## ARCHITECTURE DESCRIPTION:
**What the program does:** Given a factual claim (from the HoVer multi-hop fact-verification dataset), the pipeline must retrieve all documents that constitute the supporting evidence. The dataset is pre-filtered to examples requiring exactly 3 unique supporting documents (`hover_data.py` filters for `count_unique_docs == 3` in train, `<= 3` in test).

**Data flow:**
1. The evaluator calls `HoverMultiHopPipeline.forward(claim=...)`.
2. `CountingRM` is reset; the pipeline runs `HoverMultiHop` under a `dspy.context` binding the configured LM and counting RM.
3. **Hop 1** — `dspy.Retrieve(k=15)` is called with the raw claim, yielding 15 passages. `ChainOfThought(Summarize1Signature)` produces `summary_1`, explicitly identifying which named entities from the claim were found and which are still missing.
4. **Hop 2** — `ChainOfThought(QueryHop2Signature)` generates an entity-focused query targeting a named entity from the claim not yet found; another `dspy.Retrieve(k=15)` fetches 15 more passages. `ChainOfThought(Summarize2Signature)` produces `summary_2`, building on `summary_1` to track remaining missing entities.
5. **Hop 3** — `ChainOfThought(QueryHop3Signature)` generates a third entity-focused query identifying the remaining uncovered entity from the claim; a third `dspy.Retrieve(k=15)` adds 15 more passages.
6. **Hop 4 (Gap Analysis)** — `_interleave_and_deduplicate` is called on hops 1-3 (max_docs=30) to produce a balanced title list; the top 20 titles are fed to `ChainOfThought(QueryHop4GapSignature)`, which identifies the specific claim entity whose Wikipedia article is still missing and generates a direct query for it. A fourth `dspy.Retrieve(k=15)` fetches 15 more passages.
7. All passages from all four hops are merged via `_interleave_and_deduplicate` (round-robin interleaving across hops 1-4), deduplicated by normalized title (text before ' | '), and the top 21 are returned as `retrieved_docs` on a `dspy.Prediction`. Round-robin ensures each hop contributes ~5 docs rather than hop 1 consuming 15/21 slots.
8. `search_count` (4, one per hop) is attached from `CountingRM.get_count()`.

**Key modules and responsibilities:**
- `hover_pipeline.py` — `HoverMultiHopPipeline`: top-level pipeline; owns LM/RM instantiation, DSPy context scoping, and `search_count` attachment. Uses `MODEL = "openai/gpt-5.4-nano"`.
- `hover_program.py` — `HoverMultiHop`: 4-hop entity-focused ChainOfThought + Retrieve logic with round-robin interleaving deduplication. Contains `Summarize1Signature` and `Summarize2Signature` (entity-gap-aware summarization), plus `QueryHop2Signature`, `QueryHop3Signature`, and `QueryHop4GapSignature` — each with explicit instructions to identify and directly query for specific named entities missing from prior hops.
- `counting_rm.py` — `CountingRM`: thread-safe (threading.local) wrapper around `dspy.ColBERTv2` targeting a remote Modal ColBERT endpoint. Monkey-patches ColBERT's HTTP layer to use a shared connection-pooled `requests.Session` with 60s timeout and 2 retries.
- `hover_utils.py` — metric functions: `discrete_retrieval_eval` (binary subset check), `discrete_retrieval_eval_with_resource_penalty` (float), and the primary `discrete_retrieval_eval_with_resource_penalty_and_feedback` (returns `ScoreWithFeedback`).
- `hover_data.py` — `hoverBench`: loads `hover-nlp/hover` from HuggingFace, filters 3-hop examples, wraps in `dspy.Example` with `claim` as the sole input field.
- `tracing_setup.py` — imports `DSPyInstrumentor` on module load to emit OTel spans for all DSPy calls.
- `langProBe/dspy_program.py` — `LangProBeDSPyMetaProgram`: shared base class providing `setup_lm` and `program_type`.

**Metric detail:** `discrete_retrieval_eval_with_resource_penalty_and_feedback(output, supporting_facts, ...)` checks whether the normalized titles of all gold `supporting_facts` are a subset of the normalized titles extracted from `pred.retrieved_docs[:21]`. Success is 1.0 or 0.0. A penalty of `0.002 * max(0, search_count - 2)` is subtracted (2 free searches; with 4 hops the penalty is 0.004). The result is a `ScoreWithFeedback` with `score` (composite float in [0,1]) and a textual `feedback` string listing missing documents and search statistics, enabling reflective/GEPA-style optimization.
