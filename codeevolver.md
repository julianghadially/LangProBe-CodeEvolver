PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is an entity-focused two-stage retrieval system for fact-checking claims from the HoVer (Fact Extraction and VERification over Unstructured and Structured information) dataset. The system extracts specific named entities from claims, generates targeted entity-specific queries, retrieves a larger pool of candidate documents, and applies LLM-based reranking to select the most relevant documents. This entity-centric approach with LLM reranking dramatically improves precision by targeting specific facts rather than broad topics.

**Key Modules**:
- **HoverMultiHopPipeline** (hover_pipeline.py): Top-level pipeline wrapper that initializes the ColBERTv2 retrieval model and orchestrates the program execution. Inherits from LangProBeDSPyMetaProgram and dspy.Module.
- **HoverMultiHop** (hover_program.py): Core retrieval logic implementing entity-focused queries with LLM-based reranking. Extracts 3-5 named entities from the claim, generates entity-specific retrieval queries, retrieves k=50 documents per entity query (up to 150 total across 3 queries max), applies LLM reranking in sliding windows of 10 documents, and selects top 21 documents based on reranking scores.
- **ExtractEntities** (hover_program.py): DSPy signature that extracts 3-5 specific named entities (people, works, places, organizations) from the claim for targeted retrieval.
- **EntityToQuery** (hover_program.py): DSPy signature that generates entity-specific search queries designed to retrieve factual information about each entity in the context of the claim.
- **LLMReranker** (hover_program.py): DSPy ChainOfThought signature that analyzes batches of 10 candidate documents and outputs relevance scores (1-10) for each document, enabling intelligent document reranking.
- **ClaimDecomposition** (hover_program.py): Legacy DSPy signature that decomposes claims into 2-3 focused sub-queries (kept for compatibility).
- **RelevanceScorer** (hover_program.py): Legacy DSPy ChainOfThought signature for document scoring (kept for compatibility).
- **hoverBench** (hover_data.py): Dataset loader that filters HoVer dataset examples to only include 3-hop cases, formats them as DSPy examples with claims and supporting facts.
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting document titles are present in the retrieved documents (max 21 documents).

**Data Flow**:
1. Input claim undergoes entity extraction to identify 3-5 specific named entities (people, works, places, organizations)
2. For each extracted entity (max 3 entities), a targeted entity-specific search query is generated
3. Each entity query retrieves k=50 documents (up to 150 total documents across 3 queries)
4. Retrieved documents are deduplicated to create a unique candidate pool
5. LLM-based reranking is applied in sliding windows of 10 documents, scoring each document's relevance to the claim on a 1-10 scale
6. Documents are sorted by LLM reranking scores (descending)
7. Final deduplication by normalized title is performed
8. Top 21 unique documents by reranking score are selected and returned as retrieved_docs

**Metric**: discrete_retrieval_eval evaluates whether all gold supporting fact documents are subset of retrieved documents, enforcing a maximum of 21 retrieved documents.

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

