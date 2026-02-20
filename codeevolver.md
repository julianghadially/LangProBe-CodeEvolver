PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop fact-checking system for the HoVer benchmark that retrieves supporting documents for fact verification claims. The system performs iterative document retrieval across three hops to find relevant evidence documents.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level pipeline wrapper that initializes the ColBERTv2 retrieval model and delegates to the core program
- `HoverMultiHop`: Core multi-hop retrieval program implementing the three-hop query-retrieve-summarize pattern using DSPy modules
- `hoverBench`: Dataset loader that filters HoVer dataset to 3-hop examples and formats them for evaluation
- `hover_utils`: Contains the evaluation metric `discrete_retrieval_eval`

**Data Flow**:
1. Input claim is processed through three sequential hops
2. Hop 1: Direct retrieval (k=7 docs) + summarization
3. Hop 2: Generate new query from claim+summary1, retrieve k docs, summarize with context
4. Hop 3: Generate query from claim+both summaries, retrieve k docs
5. Output: Concatenated list of all 21 retrieved documents (7×3 hops)

**Metric**: `discrete_retrieval_eval` checks if all gold supporting fact documents are present in the retrieved set (subset matching). Returns True if all required documents are found within the 21-document limit, False otherwise. The metric normalizes document titles for comparison.

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

