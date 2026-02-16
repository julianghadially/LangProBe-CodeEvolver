PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This program implements a multi-hop document retrieval system for fact-checking claims using the HoVer dataset. It performs iterative retrieval across three hops to gather relevant supporting documents for verifying factual claims.

**Key Modules**:
- **HoverMultiHopPipeline** (`hover_pipeline.py`): The top-level wrapper that initializes the ColBERTv2 retrieval model and orchestrates the forward pass through the program.
- **HoverMultiHop** (`hover_program.py`): The core multi-hop retrieval logic that performs three iterative retrieval steps, each retrieving k=7 documents per hop (21 total).
- **hover_data.py**: Dataset loader that processes the HoVer dataset, filtering for 3-hop examples and reformatting for DSPy.
- **hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` that validates retrieval quality.

**Data Flow**:
1. A claim is passed to HoverMultiHopPipeline
2. HOP 1: Direct retrieval on the claim, followed by summarization of top-k documents
3. HOP 2: Generate new query using claim + summary_1, retrieve documents, summarize with context
4. HOP 3: Generate final query using claim + both summaries, retrieve documents
5. Returns all 21 documents (7 per hop) as `retrieved_docs`

**Metric**: `discrete_retrieval_eval` checks if all gold supporting document titles are present in the retrieved set (subset matching). The system is constrained to return at most 21 documents, and success requires all ground truth documents to be found within this set.

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

