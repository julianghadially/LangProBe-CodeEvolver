PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: HoverMultiHopPipeline is a multi-hop fact-checking system that retrieves relevant documents for claims using iterative retrieval and summarization. The system performs 3 sequential "hops" to progressively gather supporting evidence from a document corpus using the HoVer dataset.

**Key Modules**:
- **HoverMultiHopPipeline**: Top-level wrapper that initializes a ColBERTv2 retrieval model (via remote API) and delegates execution to HoverMultiHop program
- **HoverMultiHop**: Core multi-hop retrieval program implementing a 3-hop architecture:
  - Hop 1: Retrieves k=7 documents for the original claim and generates summary
  - Hop 2: Creates refined query from claim+summary_1, retrieves k=7 more docs, generates summary_2
  - Hop 3: Creates final query from claim+both summaries, retrieves k=7 final docs
  - Uses DSPy ChainOfThought modules for query generation and summarization
- **hover_data.py**: Loads HoVer dataset from Hugging Face, filters to 3-hop examples only (where count_unique_docs == 3), creates train/test splits with DSPy Examples
- **hover_utils.py**: Contains evaluation metric and helper functions

**Data Flow**: Claim → Retrieve(claim) → Summarize → Generate Query2 → Retrieve → Summarize → Generate Query3 → Retrieve → Return 21 documents (7×3 hops)

**Metric**: discrete_retrieval_eval checks if all gold supporting documents (from example.supporting_facts) are found within the top 21 retrieved documents (normalized title matching). Returns boolean success/failure for exact document retrieval.

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

