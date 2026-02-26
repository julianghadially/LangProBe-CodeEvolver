PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements a Triple-Query Architecture with Attribute-Enhanced Retrieval for multi-hop document retrieval on the HoVer claim verification benchmark using DSPy. The system combines semantic retrieval, explicit entity-title queries, and attribute-enhanced queries to maximize recall. Each iteration generates THREE types of queries in parallel: (1) semantic queries from claim decomposition and gap analysis for contextual retrieval (max 3, k=5), (2) entity-focused queries that are literal Wikipedia article titles (max 3, k=3), and (3) attribute-enhanced queries combining entities with temporal/relational qualifiers (max 4, k=3). This triple approach ensures semantic relevance, direct entity article retrieval, and temporally/relationally-specific document capture across 3 iterations, retrieving and deduplicating documents before scoring with LLM to return the top 21 most relevant.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Main pipeline implementing triple-query architecture. For each of 3 iterations: (1) generates max 3 semantic queries (k=5 docs each) via ClaimDecomposition/GapAnalysis, (2) generates max 3 entity-title queries (k=3 docs each) via EntityTitleExtractor, (3) generates max 4 attribute-enhanced queries (k=3 docs each) via AttributeExtractor, (4) retrieves documents for all three query types in parallel, (5) deduplicates by title after each iteration, (6) extracts entities/relationships for next iteration. After 3 iterations, scores all unique documents with LLM and returns top 21. Entry point for evaluation.

**ClaimDecomposition** (`hover_pipeline.py`): Signature decomposing claims into 2-3 semantic sub-questions for contextual retrieval.

**EntityTitleExtractor** (`hover_pipeline.py`): Signature extracting 3 potential Wikipedia article titles (proper nouns: people, places, events, works, organizations) from claim and context. Acts as lexical anchor for direct entity article retrieval.

**AttributeExtractor** (`hover_pipeline.py`): NEW signature identifying temporal qualifiers (years, dates, seasons), relational qualifiers (films, albums, TV shows), and other specific attributes from claim and context. Generates 3-4 entity-attribute pairs for enhanced queries (e.g., 'Rosario Dawson 1995 film debut', 'New York Islanders 1974 season').

**EntityExtractor** (`hover_pipeline.py`): Signature extracting entities/relationships from documents for structured knowledge tracking.

**GapAnalysis** (`hover_pipeline.py`): Signature analyzing missing information, generating 2-3 targeted semantic queries for next iteration.

**DocumentRelevanceScorer** (`hover_pipeline.py`): ChainOfThought module scoring document relevance (1-10).

**hover_utils**: Contains `discrete_retrieval_eval` metric for recall@21 evaluation.

**hover_data**: Loads HoVer dataset with 3-hop examples.

## Data Flow
1. Input claim enters `HoverMultiHopPipeline.forward()`
2. **Iteration 1**: (A) Decompose claim → max 3 semantic queries → retrieve k=5 docs per query; (B) Extract entity titles from claim → max 3 entity queries → retrieve k=3 docs per entity; (C) Extract attributes from claim → max 4 attribute-enhanced queries → retrieve k=3 docs per query; (D) Deduplicate by title → extract entities/relationships
3. **Iteration 2**: (A) GapAnalysis on iteration 1 results → max 3 semantic queries → retrieve k=5 docs per query; (B) Extract entity titles from accumulated context → max 3 entity queries → retrieve k=3 docs per entity; (C) Extract attributes from context → max 4 attribute-enhanced queries → retrieve k=3 docs per query; (D) Deduplicate by title → update entities/relationships
4. **Iteration 3**: (A) Final GapAnalysis → max 3 semantic queries → retrieve k=5 docs per query; (B) Extract entity titles from full context → max 3 entity queries → retrieve k=3 docs per entity; (C) Extract attributes from full context → max 4 attribute-enhanced queries → retrieve k=3 docs per query; (D) Deduplicate by title
5. **Post-Iteration**: Score all unique documents with DocumentRelevanceScorer (LLM reasoning) → sort by score descending → return top 21 documents

## Metric
The `discrete_retrieval_eval` metric computes recall@21: whether all gold supporting document titles are in the retrieved set. The iterative entity discovery architecture maximizes recall through claim decomposition, structured entity/relationship tracking, gap analysis for missing information, and LLM scoring to select the most relevant 21 documents.

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

