```
PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval_with_resource_penalty_and_feedback

## ARCHITECTURE TITLE: Entity-Focused 4-Hop Retrieval, k=7, Title-Only Coverage Check, No previous_queries, IdentifyNextTarget

## ARCHITECTURE SUMMARY:
`HoverMultiHopPipeline` (`langProBe/hover/hover_pipeline.py`) is the top-level DSPy module evaluated per row. It wraps `HoverMultiHop` (`langProBe/hover/hover_program.py`), a fixed 4-hop retrieval chain, inside a `CountingRM` (`langProBe/hover/counting_rm.py`) instrumented retrieval model that tracks how many ColBERT searches are fired per example.

The inner `HoverMultiHop` module performs four sequential hops using an entity-focused query strategy with k=7 documents per retrieval. Hop 1 retrieves directly on the raw claim. Hops 2, 3, and 4 each call a `ChainOfThought(IdentifyNextTarget)` module that scans the claim and already-retrieved passage titles to identify the single most important uncovered Wikipedia entity/article name, then retrieves against that concise entity name. The `previous_queries` field has been removed — the model instead relies on the updated prompt and retrieved passage titles to avoid repetition. After all hops, results are deduplicated by article title (first occurrence wins) and capped at 21 documents.

The metric (`hover_utils.py:discrete_retrieval_eval_with_resource_penalty_and_feedback`) checks whether all gold supporting-document titles appear in the top-21 retrieved documents, then applies a soft search-count penalty (0.002 per query beyond a free budget of 2), and returns a `ScoreWithFeedback` object carrying both the composite float score and a natural-language feedback string for reflective optimization.

## ARCHITECTURE DESCRIPTION:
**What the program does**: Given a factual claim from the HoVer multi-hop fact-checking dataset, the pipeline retrieves the set of Wikipedia articles that support or refute the claim. The dataset (`hover_data.py:hoverBench`) loads `hover-nlp/hover` from HuggingFace, filters to examples requiring exactly 2–3 supporting documents, and wraps each as a `dspy.Example(claim=..., supporting_facts=..., label=...)` with `claim` as the input key.

**Key modules and responsibilities**:
- `hover_pipeline.py / HoverMultiHopPipeline`: Outer pipeline; instantiates the LM (`openai/gpt-5.4-nano`, reasoning_effort="low"), wraps `dspy.ColBERTv2` in `CountingRM`, resets the per-thread search counter before each forward pass, then reads the count back into `result.search_count` after the inner program finishes. Inherits `LangProBeDSPyMetaProgram`.
- `hover_program.py / HoverMultiHop`: Core 4-hop DSPy module with k=7 documents per retrieval. Hop 1 retrieves directly on the claim. Hops 2, 3, and 4 each use `ChainOfThought(IdentifyNextTarget)` to identify the single most important uncovered entity. There is no `previous_queries` field — the model uses the retrieved passage titles (before ' | ') to determine what is already covered. Results are deduplicated by article title (case-insensitive prefix match on " | " separator) before being capped at 21 documents.
- `hover_program.py / IdentifyNextTarget`: DSPy Signature with two input fields (`claim`, `retrieved_passages`) that instructs the LM to enumerate named entities in the claim, check which are covered by dedicated article titles in retrieved passages (a mere mention inside another article's text does NOT count; disambiguation pages do NOT count), and output a single Wikipedia article title or entity name — explicitly forbidding question-style or sentence-style output.
- `counting_rm.py / CountingRM`: Thread-safe retrieval wrapper using `threading.local()` for per-thread counters (safe for `dspy.Evaluate` parallel mode). Also monkey-patches `dspy.dsp.colbertv2` to use a shared, connection-pooled `requests.Session` with configurable timeout (60 s default) and retry backoff, addressing DNS resolution failures under concurrent load.
- `tracing_setup.py`: On import, calls `DSPyInstrumentor().instrument()` (openinference) once, attaching OpenTelemetry spans to every DSPy Predict/Retrieve/LM call for CodeEvolver's IterationArchitect trace inspection.
- `hover_utils.py`: Metric logic. `discrete_retrieval_eval_with_resource_penalty_and_feedback` normalizes gold and found title sets, checks subset containment (binary success), applies `PENALTY_PER_SEARCH * max(0, search_count - 2)` soft penalty, and returns `ScoreWithFeedback(score, feedback)` where feedback details missing documents, counts, and penalty values.

**Data flow**: `claim` → `HoverMultiHopPipeline.forward` → reset counter → `HoverMultiHop.forward` (hop1: ColBERT on raw claim k=7; hop2: `IdentifyNextTarget`(no previous_queries) → entity name → ColBERT k=7; hop3: `IdentifyNextTarget`(passages from hops 1+2) → entity name → ColBERT k=7; hop4: `IdentifyNextTarget`(passages from hops 1+2+3) → entity name → ColBERT k=7; dedup by title → cap at 21) → `dspy.Prediction(retrieved_docs=[≤21 passages])` → attach `search_count` → metric compares `retrieved_docs[:21]` titles against `supporting_facts[*].key` → `ScoreWithFeedback`.

**Metric being optimized**: A composite float in [0, 1] equal to `1.0 - penalty` if all gold documents are found (0.0 otherwise), where penalty = 0.002 × max(0, searches − 2). The hard output constraint is ≤ 21 returned documents; there is no hard cap on search count, only the soft penalty.
```
