PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPredictPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for the HoVer (Hover Verification) fact-checking benchmark. The system performs iterative retrieval across three hops to collect supporting documents for claim verification, optimizing for retrieval coverage of gold supporting facts.

**Key Modules**:
- `HoverMultiHopPredictPipeline`: Top-level pipeline wrapper that initializes a ColBERTv2 retriever and orchestrates the forward pass through the multi-hop prediction program
- `HoverMultiHopPredict`: Core retrieval program implementing a 3-hop iterative retrieval strategy with query refinement and summarization between hops
- `hover_utils.discrete_retrieval_eval`: Evaluation metric that checks if all gold supporting documents are present in the retrieved set (subset matching with max 21 documents)
- `hover_data.hoverBench`: Dataset loader that filters HoVer dataset for 3-hop examples and formats them for DSPy

**Data Flow**:
1. Input claim is passed to the pipeline
2. **Hop 1**: Retrieve k=7 documents using the claim as query, then summarize them
3. **Hop 2**: Generate new query from claim + summary_1, retrieve k=7 more documents, summarize with context
4. **Hop 3**: Generate final query from claim + both summaries, retrieve k=7 final documents
5. Return combined set of all retrieved documents (21 total)
6. Evaluation checks if gold supporting fact keys are subset of retrieved document keys

**Optimization Target**: The `discrete_retrieval_eval` metric returns binary success/failure based on whether all supporting documents are retrieved within the 21-document limit.

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

