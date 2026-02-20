PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This system performs multi-hop document retrieval for claim verification using the HoVer dataset. It retrieves relevant supporting documents through an iterative 3-hop retrieval process with gap analysis feedback loops, where each hop refines the search based on identified gaps in coverage.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level pipeline wrapper that initializes the ColBERTv2 retrieval model and coordinates execution
- `HoverMultiHop`: Core program implementing the 3-hop retrieval strategy with gap analysis, query generation, and summarization at each hop
- `GapAnalysis`: DSPy Signature for analyzing gaps between claim requirements and retrieved passages, outputting missing entities and coverage assessment
- `CreateQueryHop2` & `CreateQueryHop3`: DSPy Signatures for generating targeted queries informed by missing entities from gap analysis
- `hover_utils.py`: Contains the evaluation metric `discrete_retrieval_eval` that validates retrieval quality
- `hover_data.py`: Benchmark class managing the HoVer dataset, filtering for 3-hop examples

**Data Flow**:
1. Input claim is used for initial retrieval (Hop 1), fetching k=7 documents
2. Retrieved documents are summarized using ChainOfThought
3. **Gap Analysis after Hop 1**: Analyzes retrieved passages to identify missing entities and coverage gaps
4. Gap-informed query generation for Hop 2, using missing entities to generate targeted queries, fetching 7 more documents
5. Hop 2 results are summarized with previous context
6. **Gap Analysis after Hop 2**: Analyzes all retrieved documents (Hop 1 + Hop 2) to identify remaining gaps
7. Highly targeted Hop 3 query generation using remaining missing entities, fetching final 7 documents
8. All retrieved documents (21 total) are concatenated and returned

The gap analysis feedback loop ensures that each subsequent hop explicitly searches for what is missing, improving retrieval precision by identifying and targeting coverage gaps rather than simply refining based on summaries alone.

**Metric**: `discrete_retrieval_eval` checks if all gold-standard supporting document titles are present in the retrieved set (up to 21 documents). Returns True only when all required documents are successfully retrieved, evaluating as a strict subset match.

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

