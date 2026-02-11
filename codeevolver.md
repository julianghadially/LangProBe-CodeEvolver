PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPredictPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop question answering system for the HoVer (Hop Verification) benchmark, designed to retrieve supporting documents across multiple reasoning steps to verify fact-based claims.

**Key Modules**:
- `HoverMultiHopPredictPipeline`: Top-level pipeline wrapper that initializes the ColBERTv2 retrieval model and orchestrates the prediction flow
- `HoverMultiHopPredict`: Core program implementing 3-hop retrieval strategy with iterative query refinement and summarization
- `hover_data.py`: Data loader filtering HoVer dataset for 3-hop examples from train/validation splits
- `hover_utils.py`: Evaluation utilities including the `discrete_retrieval_eval` metric

**Data Flow**:
1. Pipeline receives a claim as input
2. Hop 1: Retrieves top-k documents directly from claim, generates summary
3. Hop 2: Creates new query from claim+summary1, retrieves more documents, generates summary2
4. Hop 3: Creates final query from claim+both summaries, retrieves final documents
5. Returns combined retrieved_docs from all three hops (up to 21 documents)

**Metric**: `discrete_retrieval_eval` measures whether all gold supporting fact titles are found within the top 21 retrieved documents. Returns True if all required documents are retrieved (exact subset match), optimizing for complete multi-hop document coverage.

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

