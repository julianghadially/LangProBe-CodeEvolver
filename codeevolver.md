PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements a two-stage adaptive retrieval system for the HoVer (Hover-nlp) claim verification benchmark using DSPy. The system uses entity-aware multi-query generation followed by LLM-based listwise reranking to find supporting documents for fact-checking claims that require evidence from multiple sources. This approach replaces the sequential 3-hop strategy with a more flexible parallel retrieval and intelligent reranking architecture.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level pipeline wrapper that initializes the ColBERTv2 retrieval model and manages the execution context. Serves as the entry point for the evaluation framework.

**HoverMultiHop** (`hover_program.py`): Core retrieval logic implementing a two-stage adaptive retrieval architecture:
- Stage 1: Entity-aware multi-query generation using `EntityQueryDecomposition` signature (via dspy.ChainOfThought) that creates 2-3 diverse sub-queries from the claim, then retrieves k=12-15 documents per query (staying under k=35 total per retrieval call).
- Stage 2: LLM-based listwise reranking using `DocumentReranker` signature (via dspy.ChainOfThought) that takes all retrieved documents (24-45 total) and outputs exactly 21 document IDs ranked by relevance and coverage, ensuring diverse evidence collection.

**EntityQueryDecomposition**: DSPy signature for decomposing claims into 2-3 diverse entity-aware sub-queries that target different aspects, entities, or perspectives relevant to the claim.

**DocumentReranker**: DSPy signature for reasoning about and ranking all retrieved documents by relevance and coverage, outputting exactly 21 document IDs in ranked order.

**hover_utils**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting documents are found within the top 21 retrieved documents.

**hover_data**: Loads and preprocesses the HoVer dataset, filtering for 3-hop examples and formatting them for DSPy evaluation.

## Data Flow
1. Input claim enters via `HoverMultiHopPipeline.forward()`
2. Stage 1 - Multi-Query Retrieval:
   - Decompose claim into 2-3 diverse sub-queries using EntityQueryDecomposition
   - Retrieve k=15 documents per query (2 queries) or k=12 per query (3 queries)
   - Deduplicate retrieved documents (typically 24-45 unique documents)
3. Stage 2 - Listwise Reranking:
   - If ≤21 unique documents, return directly
   - Otherwise, use DocumentReranker to analyze all documents and rank by relevance/coverage
   - Return top 21 documents based on reranker output
4. Return exactly 21 documents as `retrieved_docs`

## Metric
The `discrete_retrieval_eval` metric computes recall@21: whether all gold supporting document titles from `supporting_facts` are present in the retrieved set. Success requires the retrieval pipeline to discover all necessary evidence documents within the 21-document budget.

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

