PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: HoverMultiHopPipeline is a two-stage targeted entity retrieval system that retrieves relevant supporting documents for a given claim using entity extraction and constraint-aware reranking. The system is evaluated on its ability to retrieve all gold-standard documents that support a claim.

**Key Modules**:
1. **HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level pipeline implementing a two-stage retrieval strategy:
   - **Stage 1 - Entity Extraction & Direct Retrieval**: Uses ClaimEntityParser signature to extract 2-4 specific named entities from the claim, constructs a combined query, and retrieves k=60 documents in a single search.
   - **Stage 2 - Constraint-Aware Reranking**: Uses EntityCoverageReranker signature to score all 60 documents (0-10) based on exact entity matches, definitional content, and relationship information. Returns top 21 documents.

2. **ClaimEntityParser** (`hover_pipeline.py`): DSPy Signature that extracts 2-4 specific named entities, phrases, or titles from the claim that are most likely to appear in supporting documents.

3. **EntityCoverageReranker** (`hover_pipeline.py`): DSPy Signature that scores documents based on entity coverage, prioritizing documents that directly define or describe key entities.

4. **HoverMultiHop** (`hover_program.py`): Legacy 3-hop retrieval logic (not currently used by pipeline). Implements iterative retrieval and summarization.

5. **Data Module** (`hover_data.py`): Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples with claims and supporting facts.

6. **Evaluation Metric** (`hover_utils.py`): The `discrete_retrieval_eval` function checks if all gold supporting document titles are present in the retrieved documents (maximum 21 documents).

**Data Flow**:
Claim → Extract entities (LLM) → Construct combined query → Single retrieval (k=60) → Score documents by entity coverage (LLM) → Sort by score → Return top 21 documents → Evaluate against gold supporting facts using subset matching.

**Metric**: Binary success metric that returns True if all gold-standard supporting document titles are found within the top 21 retrieved documents.

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

