PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements a multi-hop document retrieval system for the HoVer (Hop-VERification) dataset, which involves verifying claims by retrieving supporting documents across multiple reasoning hops. The system uses DSPy framework with ColBERTv2 retrieval to find relevant documents that support or refute claims requiring up to 3-hop reasoning.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): The top-level wrapper that initializes the ColBERTv2 retrieval model and orchestrates the forward pass through the program.

**HoverMultiHop** (`hover_program.py`): The core multi-hop retrieval logic implementing a 3-hop retrieval strategy:
- Hop 1: Retrieves k=7 documents based on the original claim
- Hop 2: Generates a new query from the claim + summary of hop 1 results, retrieves k more documents
- Hop 3: Generates a final query from claim + both summaries, retrieves k more documents
Each hop uses ChainOfThought modules for query generation and summarization.

**hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` which checks if all gold supporting document titles are present in the retrieved set (max 21 documents).

**hoverBench** (`hover_data.py`): Loads and preprocesses the HoVer dataset, filtering for 3-hop examples.

## Data Flow
1. Input claim enters HoverMultiHopPipeline
2. HoverMultiHop performs 3 sequential retrieval hops, each refining the search query based on previous results
3. Returns combined list of up to 21 documents (7 from each hop)
4. Metric evaluates whether all gold supporting documents are present in retrieved set

## Metric
`discrete_retrieval_eval` returns True if all gold supporting document titles (normalized) are found within the retrieved documents (subset evaluation), measuring retrieval recall.

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

