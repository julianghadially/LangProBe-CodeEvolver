PARENT_MODULE_PATH: langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPredictPipeline
METRIC_MODULE_PATH: langProPlus.hotpotGEPA.__init__.exact_match_metric

## Program Overview
HotpotGEPA is a multi-hop question answering system for the HotpotQA benchmark, implementing the GEPA paper architecture (arXiv:2507.19457). It answers complex questions requiring reasoning across multiple documents through a 3-hop pipeline with 2 retrieval steps.

## Key Modules

**HotpotMultiHopPredictPipeline** (`hotpot_pipeline.py`): Top-level pipeline wrapper that configures a ColBERTv2 retrieval model and delegates question processing to the core program.

**HotpotMultiHopPredict** (`hotpot_program.py`): Core DSPy program implementing the multi-hop reasoning architecture. Uses Predict signatures (no ChainOfThought) with k=7 documents per retrieval.

**HotpotQABench** (`hotpot_data.py`): Benchmark dataset loader that fetches HotpotQA fullwiki from HuggingFace, creates 150/300/300 train/dev/test splits (matching GEPA paper), and preserves gold supporting document titles.

## Data Flow

1. **Hop 1**: Retrieve k=7 documents for the question → Summarize passages → summary_1
2. **Hop 2**: Generate new query from (question, summary_1) → Retrieve k=7 documents → Summarize with context → summary_2
3. **Hop 3**: Generate final answer from (question, summary_1, summary_2)

Returns a dspy.Prediction containing the answer string.

## Optimization Metric

**exact_match_metric**: Uses `dspy.evaluate.answer_exact_match` to measure exact string matching between predicted answers and gold answers. The baseline performance target is 38% on GPT-4.1 mini (current: 34.3% with seed=6).

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

