PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

This is a multi-hop document retrieval system built with DSPy that retrieves supporting documents for fact-checking claims from the HoVer dataset. The system performs iterative retrieval across three hops to find relevant evidence documents.

**Key Modules:**

1. **HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level pipeline wrapper that initializes the ColBERTv2 retrieval model and orchestrates the multi-hop program execution. Inherits from `LangProBeDSPyMetaProgram` for optimization compatibility.

2. **HoverMultiHop** (`hover_program.py`): Core retrieval logic implementing three-hop iterative retrieval. Each hop retrieves k=7 documents and generates summaries to inform subsequent query generation. Uses DSPy ChainOfThought modules for query generation and summarization.

3. **hover_data.py**: Dataset loader for the HoVer benchmark, filtering examples to those requiring exactly 3 document hops for training and up to 3 hops for testing.

4. **hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting document titles are present in the retrieved results (max 21 documents).

**Data Flow:**
Claim → Hop1: Retrieve(claim) → Summarize → Hop2: Generate Query → Retrieve → Summarize → Hop3: Generate Query → Retrieve → Aggregate all documents (21 total)

**Metric:** The system is evaluated on whether it successfully retrieves all gold supporting documents (subset match) within the 21-document limit across the three retrieval hops.

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

