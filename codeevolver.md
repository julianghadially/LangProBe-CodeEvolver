PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements Query Decomposition with Iterative Entity Discovery and Bridging Entity Retrieval for multi-hop document retrieval on the HoVer claim verification benchmark using DSPy. The system transforms retrieval from similarity search to structured reasoning by decomposing claims into sub-questions, discovering entities/relationships iteratively, identifying bridging entities for dedicated retrieval, and using gap analysis for self-correction. It retrieves ~96-120 documents across 3 iterations plus bridging phase (k=12 per query), then scores with LLM to return top 21 most relevant documents.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Main pipeline implementing iterative entity discovery with bridging entity retrieval. Decomposes claims into sub-questions, retrieves k=12 docs per question, extracts entities/relationships. After iteration 1, identifies bridging entities (people, organizations, events) in retrieved documents that need dedicated retrieval, retrieves k=12 docs per bridging entity. Performs gap analysis twice to identify missing info and generate targeted queries. Deduplicates ~96-120 documents, scores with LLM, returns top 21. Entry point for evaluation.

**ClaimDecomposition** (`hover_pipeline.py`): Signature decomposing claims into 2-3 answerable sub-questions.

**EntityExtractor** (`hover_pipeline.py`): Signature extracting entities/relationships from documents for structured knowledge.

**GapAnalysis** (`hover_pipeline.py`): Signature analyzing missing information, generating targeted queries.

**BridgingEntityIdentifier** (`hover_pipeline.py`): Signature identifying 3-5 specific bridging entities (people, organizations, events) in retrieved documents that appear as important intermediate connections but need standalone retrieval.

**DocumentRelevanceScorer** (`hover_pipeline.py`): ChainOfThought module scoring document relevance (1-10).

**hover_utils**: Contains `discrete_retrieval_eval` metric for recall@21 evaluation.

**hover_data**: Loads HoVer dataset with 3-hop examples.

## Data Flow
1. Input claim enters `HoverMultiHopPipeline.forward()`
2. **Iteration 1**: Decompose claim → 2-3 sub-questions → retrieve k=12 docs per question → extract entities/relationships
3. **Bridging Entity Discovery**: Identify 3-5 bridging entities from iteration 1 docs → retrieve k=12 docs per entity using entity name as direct query (e.g., 'Lisa Raymond', 'Ellis Ferreira') → add to all_retrieved_docs
4. **Iteration 2**: GapAnalysis identifies missing info → generate 3 targeted queries → retrieve k=12 docs per query → update entities/relationships
5. **Iteration 3**: Final GapAnalysis → generate 3 queries for remaining gaps → retrieve k=12 docs per query (total ~96-120 docs)
6. **Post-Iteration**: Deduplicate by title → score with DocumentRelevanceScorer (LLM reasoning) → sort by score → return top 21 documents

## Metric
The `discrete_retrieval_eval` metric computes recall@21: whether all gold supporting document titles are in the retrieved set. The iterative entity discovery architecture with bridging entity retrieval maximizes recall through claim decomposition, structured entity/relationship tracking, bridging entity identification for dedicated retrieval of implicit entities discovered through initial docs, gap analysis for missing information, and LLM scoring to select the most relevant 21 documents.

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

