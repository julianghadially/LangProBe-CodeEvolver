PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPredictPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This program implements a multi-hop retrieval system for fact-checking claims using the HoVer (Hop VERification) dataset. It performs iterative document retrieval across three hops to gather supporting evidence for verifying factual claims.

**Key Modules**:
- **HoverMultiHopPredictPipeline** (entry point): Orchestrates the pipeline by configuring a ColBERTv2 retrieval model and delegating to the core program. Inherits from LangProBeDSPyMetaProgram for DSPy framework integration.
- **HoverMultiHopPredict**: Implements the three-hop retrieval strategy. For each hop, it generates search queries, retrieves top-k documents (k=7), and creates summaries to inform subsequent hops. Returns concatenated documents from all three hops.
- **hover_data (hoverBench)**: Loads and preprocesses the HoVer dataset, filtering for 3-hop examples and reformatting into DSPy Example objects with claims and supporting facts.
- **hover_utils**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting document titles are present in the retrieved documents (max 21).

**Data Flow**: Input claim → Hop 1 (retrieve + summarize) → Hop 2 (generate query from summary + retrieve + summarize) → Hop 3 (generate query from both summaries + retrieve) → Output combined retrieved documents.

**Optimization Metric**: `discrete_retrieval_eval` returns True if all ground-truth supporting document titles are found within the top 21 retrieved documents, measuring retrieval recall success.

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

