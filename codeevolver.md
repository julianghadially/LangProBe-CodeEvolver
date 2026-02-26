PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements a Two-Stage Ensemble Reranking architecture for multi-hop document retrieval on the HoVer claim verification benchmark using DSPy. The system uses query expansion to generate diverse query formulations, performs a single retrieval phase, then applies listwise reranking with sliding windows to score documents. Final ranking combines LLM listwise scores with ColBERT retrieval scores using weighted averaging (0.6 LLM + 0.4 ColBERT) to return the top 21 most relevant documents.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Main pipeline implementing two-stage ensemble reranking. Expands claim into 5-7 diverse queries (entity-focused, relationship-focused, negation), retrieves k=5 docs per query in single phase (~35 docs total). Applies listwise reranking with sliding windows (10 docs, 3-doc overlap). Combines LLM listwise scores (0.6 weight) with ColBERT retrieval scores (0.4 weight) to rank documents. Returns top 21 by combined score. Entry point for evaluation.

**QueryExpander** (`hover_pipeline.py`): Signature generating 5-7 diverse query formulations from the claim, including entity-focused queries, relationship-focused queries, and negation queries.

**ListwiseReranker** (`hover_pipeline.py`): Signature ranking a list of 10 documents at a time by relevance to the claim, outputting ranked indices and relevance scores (0-100).

**hover_utils**: Contains `discrete_retrieval_eval` metric for recall@21 evaluation.

**hover_data**: Loads HoVer dataset with 3-hop examples.

## Data Flow
1. Input claim enters `HoverMultiHopPipeline.forward()`
2. **Stage 1 - Query Expansion**: Generate 5-7 diverse query formulations from claim (entity-focused, relationship-focused, negation queries)
3. **Stage 2 - Batch Retrieval**: Single retrieval phase, retrieve k=5 docs per query (~35 docs total), assign ColBERT scores based on retrieval rank (100 to 50)
4. **Stage 3 - Deduplication**: Deduplicate documents by title
5. **Stage 4 - Listwise Reranking**: Process unique docs through sliding windows of 10 documents with 3-document overlap, get LLM listwise scores (0-100) for each document, average scores for docs appearing in multiple windows
6. **Stage 5 - Ensemble Scoring**: Combine scores using weighted average: 0.6 × LLM_score + 0.4 × ColBERT_score
7. **Stage 6 - Final Selection**: Sort by combined score descending, return top 21 documents

## Metric
The `discrete_retrieval_eval` metric computes recall@21: whether all gold supporting document titles are in the retrieved set. The ensemble reranking architecture maximizes recall through diverse query expansion to capture different aspects of multi-hop claims, single-phase retrieval for efficiency, listwise reranking to leverage LLM comparative judgment across documents, and ensemble scoring that balances semantic understanding (LLM) with lexical relevance (ColBERT).

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

