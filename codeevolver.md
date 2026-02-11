PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPredictPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This system implements a multi-hop document retrieval pipeline for fact verification using the HoVer (Hop Verification) dataset. It performs iterative retrieval across three hops to gather supporting documents for claim verification.

**Key Modules**:
- **HoverMultiHopPredictPipeline** (entry point): Orchestrates the pipeline, initializing ColBERTv2 retrieval model and delegating to the core program
- **HoverMultiHopPredict**: Core multi-hop retrieval logic implementing a three-stage iterative process
- **hover_data.py/hoverBench**: Loads and preprocesses the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval`

**Data Flow**:
1. Input: A claim to be verified
2. **Hop 1**: Retrieve k=7 documents based on the claim, summarize them
3. **Hop 2**: Generate a new query from claim + summary_1, retrieve k=7 more documents, create summary_2
4. **Hop 3**: Generate query from claim + both summaries, retrieve k=7 final documents
5. Output: Combined 21 retrieved documents (7 per hop)

**Metric**: `discrete_retrieval_eval` evaluates whether all gold supporting document titles (from ground truth) are present in the retrieved documents (up to 21 max). Returns True if all required documents are found (gold_titles ⊆ found_titles), False otherwise. This measures retrieval recall for multi-hop reasoning.

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

