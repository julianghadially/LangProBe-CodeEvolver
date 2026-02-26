PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements Query Decomposition with Iterative Entity Discovery and Bridging Entity Retrieval for multi-hop document retrieval on the HoVer claim verification benchmark using DSPy. The system transforms retrieval from similarity search to structured reasoning by decomposing claims into sub-questions, discovering entities/relationships iteratively, identifying bridging entities for dedicated retrieval, and using gap analysis for self-correction. It uses a two-stage retrieval strategy with intermediate reranking: initial broad retrieval (k=20) for comprehensive coverage, early reranking to identify top-15 most relevant documents for entity extraction, and targeted retrieval (k=10) for gap-filling queries. It retrieves documents across 3 iterations plus bridging phase, then scores with LLM to return top 21 most relevant documents.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Main pipeline implementing iterative entity discovery with bridging entity retrieval and two-stage retrieval strategy. Decomposes claims into sub-questions, retrieves k=20 docs per question for broad coverage. Applies intermediate reranking to score and keep top-15 documents for entity extraction and bridging entity identification. After iteration 1, identifies bridging entities (people, organizations, events) in top-15 reranked documents, retrieves k=10 docs per bridging entity. Performs gap analysis twice to identify missing info and generate targeted queries with k=10 retrieval. Deduplicates documents, scores with LLM, returns top 21. Entry point for evaluation.

**ClaimDecomposition** (`hover_pipeline.py`): Signature decomposing claims into 2-3 answerable sub-questions.

**EntityExtractor** (`hover_pipeline.py`): Signature extracting entities/relationships from documents for structured knowledge.

**GapAnalysis** (`hover_pipeline.py`): Signature analyzing missing information, generating targeted queries.

**BridgingEntityIdentifier** (`hover_pipeline.py`): Signature identifying 3-5 specific bridging entities (people, organizations, events) in retrieved documents that appear as important intermediate connections but need standalone retrieval.

**DocumentRelevanceScorer** (`hover_pipeline.py`): ChainOfThought module scoring document relevance (1-10).

**hover_utils**: Contains `discrete_retrieval_eval` metric for recall@21 evaluation.

**hover_data**: Loads HoVer dataset with 3-hop examples.

## Data Flow
1. Input claim enters `HoverMultiHopPipeline.forward()`
2. **Iteration 1**: Decompose claim → 2-3 sub-questions → retrieve k=20 docs per question for broad coverage
3. **Intermediate Reranking**: Score iteration 1 docs with DocumentRelevanceScorer → keep top 15 highest-scoring documents for entity extraction
4. **Entity Extraction**: Extract entities/relationships from top-15 reranked documents
5. **Bridging Entity Discovery**: Identify 3-5 bridging entities from top-15 reranked docs → retrieve k=10 docs per entity using entity name as targeted query (e.g., 'Lisa Raymond', 'Ellis Ferreira') → add to all_retrieved_docs
6. **Iteration 2**: GapAnalysis identifies missing info → generate 3 targeted queries → retrieve k=10 docs per query → update entities/relationships
7. **Iteration 3**: Final GapAnalysis → generate 3 queries for remaining gaps → retrieve k=10 docs per query
8. **Post-Iteration**: Deduplicate by title → score with DocumentRelevanceScorer (LLM reasoning) → sort by score → return top 21 documents

## Metric
The `discrete_retrieval_eval` metric computes recall@21: whether all gold supporting document titles are in the retrieved set. The iterative entity discovery architecture with two-stage retrieval and intermediate reranking maximizes recall through: claim decomposition, broad initial retrieval (k=20), early reranking to focus on top-15 most promising documents, structured entity/relationship tracking from high-quality documents, bridging entity identification for dedicated retrieval (k=10) of implicit entities discovered through reranked docs, gap analysis with targeted retrieval (k=10) for missing information, and final LLM scoring to select the most relevant 21 documents.

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

