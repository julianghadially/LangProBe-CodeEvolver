PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval_with_resource_penalty_and_feedback

## ARCHITECTURE TITLE: 3-Hop Retrieval with k=15 per hop + Title Deduplication for 21-doc Coverage

## ARCHITECTURE SUMMARY:
The system is a DSPy-based multi-hop retrieval pipeline for the HoVer fact-checking benchmark. The top-level entry point `HoverMultiHopPipeline` (`langProBe/hover/hover_pipeline.py`) wraps `HoverMultiHop` (`langProBe/hover/hover_program.py`), binding a specific LM (gpt-5.4-nano at low reasoning effort) and a thread-safe `CountingRM` (`langProBe/hover/counting_rm.py`) that wraps a remote ColBERTv2 retriever. The pipeline retrieves supporting documents for a factual claim across three iterative hops, now with k=15 passages per hop (45 total) and title-based deduplication to maximize unique document coverage within the 21-doc budget.

The metric (`langProBe/hover/hover_utils.py`) evaluates whether all gold supporting documents appear in the top-21 retrieved results, applying a soft penalty for retrieval queries beyond a free budget of 2. It returns a `ScoreWithFeedback` object providing both a numeric score and textual feedback about missing documents for use with reflective optimizers.

The benchmark data (`langProBe/hover/hover_data.py`) loads the `hover-nlp/hover` HuggingFace dataset, filtered to examples requiring exactly 3 supporting documents (3-hop), and is registered via `__init__.py` as a `BenchmarkMeta` for the LangProBe evaluation framework.

## ARCHITECTURE DESCRIPTION:
**What the program does:** Given a factual claim, the pipeline performs three sequential retrieval hops against a Wikipedia ColBERT index to gather all supporting documents needed to verify the claim. The goal is coverage: retrieve every gold supporting document within a 21-document budget, using as few search queries as possible.

**Key modules and responsibilities:**

- `HoverMultiHopPipeline` (`langProBe/hover/hover_pipeline.py`): Top-level DSPy module invoked by the evaluator. Sets up `openai/gpt-5.4-nano` (low reasoning effort) and a `CountingRM`-wrapped ColBERTv2 retriever. Resets the per-thread retrieval counter before each forward pass, runs `HoverMultiHop` inside a scoped `dspy.context`, then stamps `search_count` onto the prediction before returning.

- `HoverMultiHop` (`langProBe/hover/hover_program.py`): Implements the 3-hop retrieval logic. **Hop 1:** retrieves `k=15` passages using the raw claim. **Hop 2:** uses `ChainOfThought("claim,summary_1->query")` to generate a refined query from the claim and a summarization of hop-1 passages, then retrieves 15 more. **Hop 3:** uses `ChainOfThought("claim,summary_1,summary_2->query")` to generate a third query using both prior summaries, retrieves 15 more. After all 3 hops (45 total docs), deduplicates by normalized title using `dspy.evaluate.normalize_text` and returns the first 21 unique documents as `retrieved_docs`. The two summarization modules (`summarize1`, `summarize2`) distill intermediate context to guide each subsequent hop.

- `CountingRM` (`langProBe/hover/counting_rm.py`): Thread-safe wrapper around ColBERTv2. Uses `threading.local()` for per-thread counting (safe under `dspy.Evaluate`'s thread pool). Also monkey-patches ColBERTv2's HTTP request function to extend the timeout to 60s and retry up to 2 times on connection errors from the remote Modal server.

- `hoverBench` (`langProBe/hover/hover_data.py`): Loads `hover-nlp/hover` from HuggingFace, filters to 3-unique-document examples (3-hop), shuffles with fixed seeds, and wraps rows as `dspy.Example(claim=..., supporting_facts=..., label=...)` with `claim` as the sole input field.

- `hover_utils.py`: Contains the metric hierarchy. `discrete_retrieval_eval` is the binary retrieval check (gold titles ⊆ top-21 retrieved titles, after normalizing). `discrete_retrieval_eval_with_resource_penalty` subtracts `0.002 * max(0, search_count - 2)` from the binary score. `discrete_retrieval_eval_with_resource_penalty_and_feedback` (the active metric) adds a `ScoreWithFeedback` wrapper with human-readable feedback listing missing documents, search count, penalty, and composite score — enabling reflective optimizers like GEPA.

- `tracing_setup.py`: Imported as a side-effect by `hover_pipeline.py`; instruments DSPy with OpenTelemetry via `openinference.instrumentation.dspy.DSPyInstrumentor` for span-level tracing across parallel threads.

- `LangProBeDSPyMetaProgram` (`langProBe/dspy_program.py`): Shared base class providing `setup_lm()` and `program_type()` conventions used across all LangProBe benchmarks.

**Data flow:** Evaluator calls `HoverMultiHopPipeline.forward(claim=<string>)` → `CountingRM.reset_count()` → 3 hops of `Retrieve(k=15)` interleaved with `ChainOfThought` query generation and summarization → 45 raw docs deduplicated by normalized title → `dspy.Prediction(retrieved_docs=[up to 21 unique passages])` augmented with `search_count=3` → metric receives `(output=prediction, supporting_facts=<gold list>)` → returns `ScoreWithFeedback(score ∈ [0,1], feedback=<string>)`.

**Hard constraint:** At most 21 documents may appear in `retrieved_docs`. **Soft constraint:** Searches beyond 2 incur a 0.002-per-search penalty on the composite score.
