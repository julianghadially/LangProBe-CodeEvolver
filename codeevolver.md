PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements a Reciprocal Rank Fusion (RRF) reranking architecture for multi-hop document retrieval on the HoVer claim verification benchmark using DSPy. The system uses query perspective generation to create 4-5 diverse query reformulations per iteration from different angles (entity-focused, relationship-focused, temporal-focused, comparison-focused, context-focused). Each iteration retrieves k=7 documents per query (~35 documents per iteration). After retrieval, RRF formula (score = sum(1/(60 + rank))) mathematically prioritizes documents that consistently rank well across multiple query perspectives. The top 30 documents by RRF score are retained after each iteration to provide context for the next iteration. After 3 iterations, final RRF scoring across all accumulated query results returns the top 21 documents, eliminating expensive LLM-based scoring.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Main pipeline implementing RRF reranking architecture. For each of 3 iterations: (1) generates 4-5 diverse query perspectives via QueryPerspectiveGenerator, (2) retrieves k=7 documents per query, (3) applies RRF formula across all query results, (4) keeps top 30 documents by RRF score for next iteration's context. After 3 iterations, applies final RRF scoring across all accumulated query results and returns top 21 documents. Entry point for evaluation.

**QueryPerspectiveGenerator** (`hover_pipeline.py`): Signature generating 4-5 diverse query reformulations from different perspectives: entity-focused (targeting specific entities), relationship-focused (connections between entities), temporal-focused (time periods/dates), comparison-focused (contrasting aspects), and context-focused (background information). Maximizes retrieval coverage by approaching claim from multiple angles.

**EntityExtractor** (`hover_pipeline.py`): Signature extracting entities/relationships from documents for structured knowledge tracking (retained but not actively used in RRF flow).

**GapAnalysis** (`hover_pipeline.py`): Signature analyzing missing information, generating targeted queries for next iteration (retained but not actively used in RRF flow).

**hover_utils**: Contains `discrete_retrieval_eval` metric for recall@21 evaluation.

**hover_data**: Loads HoVer dataset with 3-hop examples.

## Data Flow
1. Input claim enters `HoverMultiHopPipeline.forward()`
2. **Iteration 1**: (A) Generate 4-5 diverse query perspectives from claim; (B) Retrieve k=7 documents per query (~35 docs total); (C) Apply RRF formula across all query results; (D) Keep top 30 documents by RRF score; (E) Use top 10 docs as context for next iteration
3. **Iteration 2**: (A) Generate 4-5 diverse query perspectives informed by iteration 1 context; (B) Retrieve k=7 documents per query; (C) Apply RRF across ALL accumulated query results (from iterations 1+2); (D) Keep top 30 documents by RRF score; (E) Use top 10 docs as context for next iteration
4. **Iteration 3**: (A) Generate 4-5 diverse query perspectives informed by iteration 2 context; (B) Retrieve k=7 documents per query; (C) Accumulate all query results
5. **Final RRF**: Apply RRF scoring across ALL accumulated query results from all 3 iterations (~15 queries total) → return top 21 documents by final RRF score

## Metric
The `discrete_retrieval_eval` metric computes recall@21: whether all gold supporting document titles are in the retrieved set. The RRF reranking architecture maximizes recall by generating diverse query perspectives that approach the claim from multiple angles, then mathematically prioritizing documents that consistently rank well across these perspectives. This eliminates the need for expensive LLM-based scoring while improving document diversity and recall.

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

