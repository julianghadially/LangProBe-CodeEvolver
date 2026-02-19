PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: HoverMultiHopPipeline is a multi-hop document retrieval system that retrieves relevant supporting documents for a given claim using a parallel diverse query expansion architecture with semantic deduplication. The system is evaluated on its ability to retrieve all gold-standard documents that support a claim.

**Key Modules**:
1. **HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level wrapper that initializes a ColBERTv2 retrieval model and orchestrates the HoverMultiHop program execution with the retrieval context.

2. **HoverMultiHop** (`hover_program.py`): Core retrieval logic implementing a 2-stage parallel approach with 3 total retrievals (k=15 each). Uses three custom DSPy Signatures:
   - **ParallelQueryGenerator**: Generates 3 diverse queries targeting different semantic aspects (entities/actors, relationships/actions, temporal/contextual details)
   - **CoverageAnalyzer**: Analyzes claim and current documents to identify 2 distinct missing information angles for adaptive expansion
   - **UtilityReranker**: Scores and ranks documents by relevance to select top 21 unique documents

3. **Data Module** (`hover_data.py`): Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples with claims and supporting facts.

4. **Evaluation Metric** (`hover_utils.py`): The `discrete_retrieval_eval` function checks if all gold supporting document titles are present in the retrieved documents (maximum 21 documents).

**Data Flow**:
Claim → **Stage 1**: Generate 3 diverse queries (1 LLM call), retrieve k=15 docs for first query → Track unique docs by title → **Stage 2**: Analyze coverage gaps, generate 2 adaptive queries, retrieve k=15 docs for each (2 searches) → **Stage 3**: Deduplicate all docs by title, rerank by utility score if >21 docs, select top 21 → Evaluate against gold supporting facts using subset matching.

**Metric**: Binary success metric that returns True if all gold-standard supporting document titles are found within the top 21 retrieved documents.

**Key Innovation**: Parallel diverse query generation maximizes coverage upfront (rather than sequential reaction), semantic deduplication eliminates redundancy, and utility-based reranking optimizes the final 21-document selection. Uses exactly 3 retrievals as constrained.

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

