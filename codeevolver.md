PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval_with_resource_penalty_and_feedback

## ARCHITECTURE TITLE: HoVer Multi-Hop Retrieval Pipeline with DSPy ChainOfThought, ColBERTv2, and Resource-Penalty Metric

## ARCHITECTURE SUMMARY:
The system is a three-hop iterative retrieval pipeline built on DSPy for the HoVer fact-checking benchmark. `HoverMultiHopPipeline` (`hover_pipeline.py`) is the top-level entry point evaluated per example; it wraps `HoverMultiHop` (`hover_program.py`), the core DSPy module containing all LM and retrieval steps. A `CountingRM` wrapper (`counting_rm.py`) intercepts every call to the ColBERTv2 retrieval service, counting queries per thread for use in scoring.

The metric (`hover_utils.py`) evaluates whether all gold supporting documents were retrieved within the hard cap of 21 documents, then applies a soft search-count penalty. The dataset (`hover_data.py`) loads 3-hop examples from the HoVer HuggingFace dataset, filters for claims requiring exactly 3 supporting documents, and wraps them as DSPy `Example` objects with `claim` as the sole input field.

## ARCHITECTURE DESCRIPTION:
**What the program does:** Given a factual `claim`, the pipeline retrieves the set of Wikipedia documents needed to verify it. The task is multi-hop: the evidence spans 2–3 distinct Wikipedia articles that must all be found.

**Data flow (per example):**
1. `HoverMultiHopPipeline.forward(claim)` resets the per-thread `CountingRM` counter, then delegates to `HoverMultiHop.forward(claim)` inside a `dspy.context(lm=..., rm=...)` block.
2. **Hop 1** – The raw `claim` is used directly as a retrieval query via `dspy.Retrieve(k=7)`. The 7 passages are summarized by `summarize1` (`ChainOfThought("claim,passages->summary")`).
3. **Hop 2** – `create_query_hop2` (`ChainOfThought("claim,summary_1->query")`) generates a refined query from the claim and hop-1 summary. A second `Retrieve(k=7)` call fetches 7 more passages, summarized by `summarize2` (`ChainOfThought("claim,context,passages->summary")`).
4. **Hop 3** – `create_query_hop3` (`ChainOfThought("claim,summary_1,summary_2->query")`) generates a third query using both summaries. A final `Retrieve(k=7)` retrieves 7 more passages.
5. All 21 passages (7×3) are concatenated into `retrieved_docs` and returned as a `dspy.Prediction`. The search count (always 3 for the base pipeline) is attached as `pred.search_count`.

**Key modules:**
- `hover_pipeline.py` – `HoverMultiHopPipeline`: pipeline wrapper, instantiates LM (`openai/gpt-5.4-nano` with `reasoning_effort="low"`), `CountingRM`, and `HoverMultiHop`.
- `hover_program.py` – `HoverMultiHop`: the DSPy module with 5 learnable sub-modules (3 ChainOfThought predictors + 1 summarizer pair + 1 Retrieve).
- `counting_rm.py` – `CountingRM`: thread-safe proxy around ColBERTv2; patches the HTTP timeout to 60s and retries transient failures; tracks per-thread query count.
- `hover_utils.py` – metric functions; `discrete_retrieval_eval_with_resource_penalty_and_feedback` returns a `ScoreWithFeedback` with a composite score = retrieval success (binary) minus a soft penalty of 0.002 per search beyond a free budget of 2.
- `hover_data.py` – `hoverBench`: loads `hover-nlp/hover` from HuggingFace, filters to 3-hop examples only (train requires exactly 3, validation allows ≤3), shuffles, and wraps as DSPy `Example(claim=...).with_inputs("claim")`.
- `tracing_setup.py` – auto-instruments DSPy with OpenTelemetry/OpenInference on import, propagating spans across parallel evaluation threads.

**Metric being optimized:** `discrete_retrieval_eval_with_resource_penalty_and_feedback` — composite score in [0, 1] equal to `max(0, success - 0.002 * max(0, search_count - 2))`, where `success` is 1 iff all gold document titles appear (normalized) in the first 21 entries of `pred.retrieved_docs`. The hard output constraint is ≤21 documents; the search penalty is a soft cost. The function also returns natural-language `feedback` identifying missing documents and penalty details, enabling reflective (GEPA-style) optimization.
