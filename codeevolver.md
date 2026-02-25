PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements a multi-hop document retrieval system for the HoVer (Hover-nlp) claim verification benchmark using DSPy. The system performs iterative retrieval across three hops to find supporting documents for fact-checking claims that require evidence from multiple sources.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level pipeline wrapper that initializes the ColBERTv2 retrieval model and manages the execution context. Serves as the entry point for the evaluation framework.

**HoverMultiHop** (`hover_program.py`): Core retrieval logic implementing a 3-hop iterative retrieval strategy. Each hop retrieves k=7 documents, uses Chain-of-Thought prompting to summarize findings, and generates refined queries for subsequent hops.

**hover_utils**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting documents are found within the top 21 retrieved documents.

**hover_data**: Loads and preprocesses the HoVer dataset, filtering for 3-hop examples and formatting them for DSPy evaluation.

## Data Flow
1. Input claim enters via `HoverMultiHopPipeline.forward()`
2. Hop 1: Retrieve k documents directly from claim, generate summary
3. Hop 2: Create refined query from claim+summary_1, retrieve k more documents, summarize
4. Hop 3: Create query from claim+both summaries, retrieve k final documents
5. Return all 21 documents (3 hops × 7 docs) as `retrieved_docs`

## Metric
The `discrete_retrieval_eval` metric computes recall@21: whether all gold supporting document titles from `supporting_facts` are present in the retrieved set. Success requires the retrieval pipeline to discover all necessary evidence documents within the 21-document budget.

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

