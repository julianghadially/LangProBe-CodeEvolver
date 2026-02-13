PARENT_MODULE_PATH: langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPredictPipeline
METRIC_MODULE_PATH: langProPlus.hotpotGEPA.__init__.exact_match_metric

## Architecture Summary

**Purpose**: This program implements a multi-hop question-answering system for the HotpotQA benchmark. It answers complex questions requiring information retrieval and reasoning across multiple document sources using a three-hop pipeline with ColBERT-based retrieval.

**Key Modules**:
- **HotpotMultiHopPredictPipeline** (hotpot_pipeline.py): Top-level pipeline wrapper that configures a ColBERTv2 retrieval model via a remote API endpoint and delegates question-answering to the core program. Serves as the entry point for evaluation.
- **HotpotMultiHopPredict** (hotpot_program.py): Core multi-hop reasoning module implementing a three-stage process: (1) retrieve documents for initial question, (2) create second query from first summary and retrieve more documents, (3) generate final answer from both summaries using DSPy Predict modules.
- **HotpotQABench** (hotpot_data.py): Data loader that fetches the HotpotQA fullwiki dataset from HuggingFace, preprocesses training and validation splits into DSPy Examples with questions, answers, and gold supporting document titles.

**Data Flow**:
1. Question input → HotpotMultiHopPredictPipeline configures ColBERT retriever context
2. First hop: Retrieve k=7 documents for question → Summarize passages
3. Second hop: Generate refined query from first summary → Retrieve k=7 more documents → Summarize with context
4. Third hop: Generate answer from both summaries using GenerateAnswer signature
5. Return DSPy Prediction with final answer

**Metric**: Uses `dspy.evaluate.answer_exact_match` (exact_match_metric) which compares predicted answers against gold standard answers for exact string matching to measure QA accuracy.

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

