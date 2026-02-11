PARENT_MODULE_PATH: langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPredictPipeline
METRIC_MODULE_PATH: langProPlus.hotpotGEPA.__init__.exact_match_metric

## Program Overview
This is a multi-hop question answering system built on DSPy for the HotpotQA benchmark. It uses retrieval-augmented generation with a three-hop pipeline to answer complex questions requiring reasoning across multiple documents.

## Key Modules

**HotpotMultiHopPredictPipeline** (`hotpot_pipeline.py`): Top-level orchestrator that configures the ColBERTv2 retrieval model and delegates to the core program.

**HotpotMultiHopPredict** (`hotpot_program.py`): Core three-hop reasoning module that:
- Hop 1: Retrieves k=7 documents for the original question and generates summary_1
- Hop 2: Creates a refined query from summary_1, retrieves additional documents, and generates summary_2
- Hop 3: Generates the final answer using both summaries

**HotpotQABench** (`hotpot_data.py`): Dataset loader that fetches HotpotQA fullwiki data, creates train/validation splits with shuffling.

## Data Flow
Question → Retrieve docs (hop 1) → Summarize → Generate query (hop 2) → Retrieve docs → Summarize with context → Generate answer. The pipeline uses ColBERTv2 for neural retrieval and DSPy's Predict modules for summarization and answer generation.

## Metric
The system is optimized using `dspy.evaluate.answer_exact_match`, which measures exact string matching between predicted and gold answers, providing a strict accuracy measure for factoid question answering.

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

