PARENT_MODULE_PATH: langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPredictPipeline
METRIC_MODULE_PATH: langProPlus.hotpotGEPA.__init__.exact_match_metric

## Program Purpose
This program implements a multi-hop question answering system for the HotpotQA benchmark using DSPy. It answers complex questions that require reasoning over multiple retrieved documents through iterative retrieval and summarization steps.

## Key Modules

**HotpotMultiHopPredictPipeline** (`hotpot_pipeline.py`): Top-level wrapper that configures the ColBERTv2 retrieval model and orchestrates the entire pipeline execution.

**HotpotMultiHopPredict** (`hotpot_program.py`): Core multi-hop reasoning module that implements a 3-hop process:
- Hop 1: Retrieves k=7 documents for the question and generates initial summary
- Hop 2: Creates a refined query from the first summary, retrieves more documents, and generates a second summary
- Hop 3: Generates the final answer from both summaries

**HotpotQABench** (`hotpot_data.py`): Data loader for the HotpotQA "fullwiki" dataset, handling train/validation splits with deterministic shuffling.

## Data Flow
1. Question input → Pipeline wrapper initializes ColBERTv2 retriever context
2. First retrieval → Summarization → Query refinement
3. Second retrieval → Context-aware summarization  
4. Both summaries → Answer generation via `GenerateAnswer` signature
5. Final answer prediction returned

## Metric
Uses `dspy.evaluate.answer_exact_match` metric to measure exact string match between predicted and gold standard answers, evaluating the system's factual accuracy on multi-hop reasoning tasks.

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

