PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system built on DSPy that retrieves supporting documents for fact-checking claims. The system performs iterative retrieval across three hops to find all relevant documents that support or relate to a given claim.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level pipeline wrapper that initializes the ColBERTv2 retriever and orchestrates the program execution
- `HoverMultiHop`: Core program implementing the 3-hop retrieval strategy with query generation and summarization at each hop
- `hover_utils.discrete_retrieval_eval`: Evaluation metric that checks if all gold-standard supporting documents are found within the top 21 retrieved documents
- `hover_data.hoverBench`: Dataset loader for the HoVer fact-checking dataset, filtering for 3-hop examples

**Data Flow**:
1. A claim is input to the pipeline
2. **Hop 1**: Direct retrieval on claim (k=7 docs), then summarize results
3. **Hop 2**: Generate new query from claim + summary_1, retrieve k docs, summarize with context
4. **Hop 3**: Generate query from claim + both summaries, retrieve k docs
5. All 21 documents (7×3 hops) are returned as `retrieved_docs`

**Metric**: The `discrete_retrieval_eval` function returns a binary score checking whether all supporting document titles from the gold standard are present in the retrieved set (subset relationship). Success requires finding all relevant documents within the 21-document limit.

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

