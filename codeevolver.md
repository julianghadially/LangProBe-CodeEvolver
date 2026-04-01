PARENT_MODULE_PATH: langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPipeline
METRIC_MODULE_PATH: langProPlus.hotpotGEPA.hotpot_metric_resource.hotpot_accuracy_with_resource_penalty_feedback

## ARCHITECTURE TITLE: DSPy Two-Hop Retrieval-Augmented QA Pipeline with GEPA Resource-Penalized Metric

## ARCHITECTURE SUMMARY:
The system is a DSPy-based multi-hop question answering pipeline over the HotpotQA "fullwiki" benchmark. The top-level entry point, `HotpotMultiHopPipeline`, wraps the core reasoning module (`HotpotMultiHop`) with a thread-safe retrieval counter (`CountingRM`) so that each forward pass records exactly how many retrieval queries were issued. The data layer (`HotpotQABench`) sources examples from HuggingFace's HotpotQA dataset, providing questions, gold answers, and supporting document titles.

The core reasoning module performs two retrieval hops followed by a direct answer generation step: the first hop retrieves passages for the raw question, summarizes them, then the second hop generates a refined query from the question and first summary, retrieves again, and produces a second summary. A final ChainOfThought step synthesizes both summaries into a short factoid answer. A `HotpotMultiHopPredict` variant substitutes `dspy.Predict` for `dspy.ChainOfThought` at every step, disabling intermediate reasoning traces.

The optimization target is `hotpot_accuracy_with_resource_penalty_feedback`, a GEPA-compatible metric that combines exact-match accuracy with a retrieval-cost penalty. It returns a `ScoreWithFeedback` object carrying both a numeric composite score and natural-language feedback, enabling gradient-free prompt optimization that simultaneously rewards correctness and discourages unnecessary retrieval calls.

## ARCHITECTURE DESCRIPTION:
**Pipeline entry point — `HotpotMultiHopPipeline` (`hotpot_pipeline.py`):**
Inherits from both `LangProBeDSPyMetaProgram` and `dspy.Module`. On each `forward(question)` call it (1) resets the per-thread retrieval counter, (2) executes `HotpotMultiHop` inside a `dspy.context(rm=self.rm)` block so all retrieval calls are intercepted by `CountingRM`, and (3) attaches `result.retrieval_count` to the prediction before returning it.

**Core reasoning — `HotpotMultiHop` (`hotpot_program.py`):**
Implements a three-stage pipeline:
- *Hop 1*: `dspy.Retrieve(k=7)` fetches passages for the raw question; `ChainOfThought("question,passages->summary")` condenses them into `summary_1`.
- *Hop 2*: `ChainOfThought("question,summary_1->query")` generates a targeted follow-up query; a second `Retrieve` call fetches additional passages; `ChainOfThought("question,context,passages->summary")` yields `summary_2`.
- *Hop 3*: `ChainOfThought(GenerateAnswer)` synthesizes `summary_1` and `summary_2` to produce a short factoid `answer`.
The `HotpotMultiHopPredict` variant mirrors this flow with `dspy.Predict` modules (no chain-of-thought scratchpad).

**Retrieval instrumentation — `CountingRM` (`counting_rm.py`):**
A lightweight decorator around any DSPy retrieval model (here, `dspy.ColBERTv2` pointing to a hosted Modal endpoint). Uses `threading.local()` so concurrent evaluation threads each maintain independent counts. Exposes `reset_count()` / `get_count()` for lifecycle management by the pipeline.

**Data — `HotpotQABench` (`hotpot_data.py`):**
Loads HotpotQA "fullwiki" split via HuggingFace `datasets`. Each example is a `dspy.Example` with fields `question`, `answer`, and `gold_titles` (set of Wikipedia article titles of supporting documents). Train and test sets are shuffled with fixed seeds for reproducibility.

**Metric — `hotpot_accuracy_with_resource_penalty_feedback` (`hotpot_metric_resource.py`):**
Computes `composite = max(0, exact_match - 0.0025 * max(0, retrieval_count - 2))`. The first two retrievals are free; each additional query incurs a 0.0025 penalty, incentivizing the optimizer to find prompts that answer correctly with fewer hops. The `_feedback` variant wraps the score in a `ScoreWithFeedback` object with a natural-language explanation (correctness verdict, query count, penalty, composite score, and required supporting documents), enabling the GEPA optimizer to use textual gradient signals during prompt search.

## DSPy Patterns and Guidelines

DSPy is an AI framework for defining a compound AI system across multiple modules. Instead of writing prompts, we define signatures. Signatures define the inputs and outputs to a module in an AI system, along with the purpose of the module in the docstring. DSPy leverages a prompt optimizer to convert the signature into an optimized prompt, which is stored as a JSON, and is loaded when compiling the program.

**DSPy docs**: https://dspy.ai/api/

Stick to DSPy for any AI modules you create, unless the client codebase does otherwise.

Defining signatures as classes is recommended. For example:

```python
class WebQueryGenerator(dspy.Signature):
    """Generate a query for searching the web."""
    question: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a query for searching the web")
```

Next, modules are used as nodes in the project, either as a single line:

```python
predict = dspy.Predict(WebQueryGenerator)
```

Or as a class:

```python
class WebQueryModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query_generator = dspy.Predict(WebQueryGenerator)

    def forward(self, question: str):
        return self.query_generator(question=question)
```

A module can represent a single module, or the module can act as a pipeline that calls a sequence of sub-modules inside `def forward`.

Common prebuilt modules include:
- `dspy.Predict`: for simple language model calls
- `dspy.ChainOfThought`: for reasoning first, followed by a response
- `dspy.ReAct`: for tool calling
- `dspy.ProgramOfThought`: for getting the LM to output code, whose execution results will dictate the response

