PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims using the HoVer (Hop Verification) dataset. The system uses a targeted three-phase retrieval strategy to identify and retrieve specific documents relevant to a given claim, focusing on concrete entities rather than broad concepts.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level wrapper that implements a three-phase targeted retrieval strategy with document analysis, precise query generation, and relevance scoring. Initializes the ColBERTv2 retriever and orchestrates the retrieval pipeline.
- `HoverMultiHop`: Core program implementing original 3-hop retrieval logic with query generation and summarization at each hop (currently not used in the pipeline)
- `hover_data.py`: Data loader that filters HoVer dataset to 3-hop examples (claim-fact pairs requiring 3 documents)
- `hover_utils.py`: Evaluation utilities including the `discrete_retrieval_eval` metric

**Data Flow (Three-Phase Targeted Retrieval)**:
1. **Phase 1 - Document Analysis**: DSPy signature `RequiredDocumentAnalysis` analyzes the claim to identify 2-4 specific document titles/topics that must be retrieved (e.g., "The Dinner Party artwork", "Sojourner Truth biography"), each with a rationale
2. **Phase 2 - Precise Query Generation**: DSPy signature `PreciseQueryGeneration` creates up to 3 highly specific queries optimized for Wikipedia title/abstract matching, each targeting a required document
3. **Phase 3 - Retrieval & Scoring**: Retrieves k=25 documents per query (up to 75 total), then DSPy signature `DocumentRelevanceScoring` scores each document (0-10) based on relevance to the required documents list
4. The top 21 highest-scoring unique documents are selected and returned as `retrieved_docs`

**Metric**: `discrete_retrieval_eval` checks if all gold supporting document titles (from `supporting_facts`) are present in the retrieved documents (max 21). Returns True if gold titles are a subset of retrieved titles after normalization.

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

