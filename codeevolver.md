PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

This program implements a self-reflective multi-hop document retrieval system for the HoVer (Hover-nlp) fact verification task using DSPy. The system retrieves supporting documents for a given claim through a five-stage retrieval, gap analysis, and hybrid scoring strategy.

**Key Modules:**

1. **HoverMultiHopPipeline** (`hover_pipeline.py`): The top-level pipeline that implements a five-stage retrieval strategy:
   - **Stage 1 - Initial Retrieval**: Generates 2 diverse queries from the claim (entity-focused via `GenerateEntityQuery` and relationship-focused via `GenerateRelationshipQuery`), retrieving k=25 documents per query for 50 total documents
   - **Stage 2 - Gap Analysis**: Uses `AnalyzeRetrievalGaps` with `ChainOfThought` to analyze the initial 50 documents and identify which key entities/facts are well-covered vs. missing or poorly covered, then generates a targeted gap-filling query
   - **Stage 3 - Targeted Retrieval**: Executes the gap-filling query with k=10 to retrieve additional documents, bringing total raw retrieval to 60 documents
   - **Stage 4 - Deduplication**: Performs deterministic deduplication by normalized title (lowercase, stripped) to remove duplicate documents from the pool
   - **Stage 5 - Hybrid Scoring**: Uses `ScoreDocumentRelevance` signature to score each unique document (0-10) with justification, then selects top 21 documents by score
   - Initializes the ColBERTv2 retriever with a remote API endpoint

2. **HoverMultiHop** (`hover_program.py`): The legacy multi-hop retrieval program (not currently used by HoverMultiHopPipeline) that performs three sequential retrieval hops with summaries.

3. **hover_data.py**: Manages dataset loading from the HoVer benchmark, filtering for 3-hop examples and creating DSPy Example objects.

4. **hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` which checks if all gold supporting document titles are present in the retrieved documents (subset match).

**Data Flow:**
Claim → Generate Entity Query → Retrieve 25 Docs → Generate Relationship Query → Retrieve 25 Docs → Combine 50 Docs → Gap Analysis (CoT) → Generate Gap-Filling Query → Retrieve 10 Docs → Combine 60 Docs → Deduplicate by Title → Score All Unique Docs → Select Top 21 by Score

**Metric:** The `discrete_retrieval_eval` metric returns True if all gold standard supporting document titles are found within the predicted retrieved documents (max 21), using normalized text matching.

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

