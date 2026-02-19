PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: HoverMultiHopPipeline is a multi-hop document retrieval system that retrieves relevant supporting documents for a given claim using a claim decomposition strategy with mandatory deduplication. The system is evaluated on its ability to retrieve all gold-standard documents that support a claim.

**Key Modules**:
1. **HoverMultiHopPipeline** (`hover_pipeline.py`): Implements a 3-stage retrieval strategy: (1) Claim Decomposition - decomposes claims into 3 sub-questions (entities, relationships, attributes) and performs a single k=50 retrieval; (2) Targeted Gap Filling - identifies missing entities and performs 2 additional k=25 retrievals; (3) Aggressive Deduplication & Reranking - deduplicates by title, scores documents for relevance, and returns top 21 unique documents.

2. **DSPy Signatures** (`hover_pipeline.py`):
   - `ClaimDecomposer`: Generates 3 sub-questions focusing on entities, relationships, and attributes
   - `MissingEntityIdentifier`: Analyzes retrieved documents to identify missing entities for gap filling
   - `DocumentRelevanceScorer`: Scores documents 0-100 for relevance to the claim

3. **Data Module** (`hover_data.py`): Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples with claims and supporting facts.

4. **Evaluation Metric** (`hover_utils.py`): The `discrete_retrieval_eval` function checks if all gold supporting document titles are present in the retrieved documents (maximum 21 documents).

**Data Flow**:
Claim → Decompose into 3 sub-questions → Stage 1: Combined retrieval (k=50) → Stage 2: Gap analysis & targeted retrieval (k=25 × 2) → Stage 3: Deduplicate by title → Score unique documents for relevance → Return top 21 unique documents → Evaluate against gold supporting facts using subset matching.

**Metric**: Binary success metric that returns True if all gold-standard supporting document titles are found within the top 21 retrieved unique documents.

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

