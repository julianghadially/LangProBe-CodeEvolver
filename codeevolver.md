PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system designed to find supporting documents for fact-checking claims. The system uses iterative retrieval and summarization to discover relevant documents across multiple reasoning hops, optimized for the HoVer (Hop-VERification) benchmark.

**Key Modules**:

1. **HoverMultiHopPipeline** (`hover_pipeline.py`): The entry point that wraps the core program with a ColBERTv2 retriever. Initializes the retrieval model with a remote ColBERT API endpoint and provides the forward pass interface.

2. **HoverMultiHop** (`hover_program.py`): The core multi-hop reasoning module implementing a 3-hop retrieval strategy:
   - Hop 1: Retrieves k=7 documents directly from the claim, then summarizes them
   - Hop 2: Generates a refined query from claim + summary_1, retrieves k documents, and creates summary_2
   - Hop 3: Generates final query from claim + both summaries, retrieves k documents
   - Combines all retrieved documents (up to 21 total) for evaluation

3. **hover_data.py**: Loads and preprocesses the HoVer dataset, filtering for examples with exactly 3 supporting document hops for training and ≤3 hops for testing.

**Data Flow**: Claim → Retrieve docs → Summarize → Generate refined query → Retrieve more docs → Summarize → Generate final query → Retrieve final docs → Return all 21 documents

**Metric**: `discrete_retrieval_eval` checks if all gold supporting document titles (from ground truth) are found within the predicted retrieved documents (max 21), using normalized text matching. Returns True if all gold documents are successfully retrieved.

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

