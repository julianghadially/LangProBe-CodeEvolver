PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval_with_resource_penalty_and_feedback

## ARCHITECTURE TITLE: 3-Hop Iterative Retrieval Pipeline with ColBERT, DSPy ChainOfThought Summarizers, and Search-Count Penalty Metric

## ARCHITECTURE SUMMARY:
`HoverMultiHopPipeline` (`langProBe/hover/hover_pipeline.py`) is the evaluator-facing entry point. It wires together an LM (`openai/gpt-5.4-nano`, reasoning_effort="low"), a `CountingRM`-wrapped ColBERTv2 retriever, and the core `HoverMultiHop` program, then tags each prediction with the search count for downstream metric use.

`HoverMultiHop` (`langProBe/hover/hover_program.py`) implements a 3-hop Baleen-style retrieval loop: each hop generates a refined query via `dspy.ChainOfThought`, retrieves the top-7 passages via `dspy.Retrieve`, and summarizes them before feeding the summary into the next hop's query generator. The 21 passages collected across all three hops are returned as `retrieved_docs`.

The metric (`langProBe/hover/hover_utils.py`) scores predictions as retrieval success (all gold supporting-document titles found within the hard cap of 21 returned docs) minus a soft penalty for search queries beyond a 2-query free budget (0.002 per extra search), returning a `ScoreWithFeedback` object for reflective optimization.

## ARCHITECTURE DESCRIPTION:
**What the program does:** Given a factual claim (from the HoVer multi-hop fact-verification dataset), the pipeline must retrieve all documents that constitute the supporting evidence. The dataset is pre-filtered to examples requiring exactly 3 unique supporting documents (`hover_data.py` filters for `count_unique_docs == 3` in train, `<= 3` in test).

**Data flow:**
1. The evaluator calls `HoverMultiHopPipeline.forward(claim=...)`.
2. `CountingRM` is reset; the pipeline runs `HoverMultiHop` under a `dspy.context` binding the configured LM and counting RM.
3. **Hop 1** — `dspy.Retrieve(k=7)` is called with the raw claim, yielding 7 passages. A `ChainOfThought("claim,passages->summary")` produces `summary_1`.
4. **Hop 2** — `ChainOfThought("claim,summary_1->query")` generates a refined query; another `dspy.Retrieve(k=7)` fetches 7 more passages. `ChainOfThought("claim,context,passages->summary")` produces `summary_2`.
5. **Hop 3** — `ChainOfThought("claim,summary_1,summary_2->query")` generates a third query; a final `dspy.Retrieve(k=7)` adds 7 more passages.
6. All 21 passages are concatenated into `retrieved_docs` on a `dspy.Prediction`.
7. `search_count` (3, one per hop) is attached from `CountingRM.get_count()`.

**Key modules and responsibilities:**
- `hover_pipeline.py` — `HoverMultiHopPipeline`: top-level pipeline; owns LM/RM instantiation, DSPy context scoping, and `search_count` attachment. Uses `MODEL = "openai/gpt-5.4-nano"`.
- `hover_program.py` — `HoverMultiHop`: 3-hop ChainOfThought + Retrieve logic; the inner program CodeEvolver should modify to improve retrieval.
- `counting_rm.py` — `CountingRM`: thread-safe (threading.local) wrapper around `dspy.ColBERTv2` targeting a remote Modal ColBERT endpoint. Monkey-patches ColBERT's HTTP layer to use a shared connection-pooled `requests.Session` with 60s timeout and 2 retries.
- `hover_utils.py` — metric functions: `discrete_retrieval_eval` (binary subset check), `discrete_retrieval_eval_with_resource_penalty` (float), and the primary `discrete_retrieval_eval_with_resource_penalty_and_feedback` (returns `ScoreWithFeedback`).
- `hover_data.py` — `hoverBench`: loads `hover-nlp/hover` from HuggingFace, filters 3-hop examples, wraps in `dspy.Example` with `claim` as the sole input field.
- `tracing_setup.py` — imports `DSPyInstrumentor` on module load to emit OTel spans for all DSPy calls.
- `langProBe/dspy_program.py` — `LangProBeDSPyMetaProgram`: shared base class providing `setup_lm` and `program_type`.

**Metric detail:** `discrete_retrieval_eval_with_resource_penalty_and_feedback(output, supporting_facts, ...)` checks whether the normalized titles of all gold `supporting_facts` are a subset of the normalized titles extracted from `pred.retrieved_docs[:21]`. Success is 1.0 or 0.0. A penalty of `0.002 * max(0, search_count - 2)` is subtracted (2 free searches; the baseline uses exactly 3, incurring a 0.002 penalty). The result is a `ScoreWithFeedback` with `score` (composite float in [0,1]) and a textual `feedback` string listing missing documents and search statistics, enabling reflective/GEPA-style optimization.
