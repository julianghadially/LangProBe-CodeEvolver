PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims using the HoVer (Hop Verification) dataset. The system performs hybrid literal + generated query retrieval to find supporting documents relevant to a given claim, combining exact phrase matching with semantic bridging queries.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level pipeline implementing hybrid retrieval strategy with key phrase extraction, bridging queries, and coverage-based reranking
- `KeyPhraseExtraction`: DSPy signature that extracts 2-3 quoted phrases/entity names from claims for exact matching
- `BridgingQuery`: DSPy signature that generates queries to find connecting/bridging documents
- `CoverageReranker`: DSPy signature that reranks documents by exact phrase matches, entity coverage, and claim relevance
- `HoverMultiHop`: Original 3-hop retrieval program (no longer used by pipeline)
- `hover_data.py`: Data loader that filters HoVer dataset to 3-hop examples (claim-fact pairs requiring 3 documents)
- `hover_utils.py`: Evaluation utilities including the `discrete_retrieval_eval` metric

**Data Flow**:
1. Extract 2-3 key phrases/entity names from the input claim using KeyPhraseExtraction
2. Query 1: Use first extracted phrase to retrieve k=15 documents (literal matching)
3. Query 2: Generate bridging query from claim + initial results to retrieve k=10 connecting documents
4. Query 3: Use second extracted phrase to retrieve k=10 documents (literal matching)
5. Concatenate all 35 documents (15+10+10)
6. Rerank using CoverageReranker based on: (a) exact phrase matches, (b) entity coverage, (c) claim relevance
7. Return top 21 documents from reranking as `retrieved_docs`

**Metric**: `discrete_retrieval_eval` checks if all gold supporting document titles (from `supporting_facts`) are present in the retrieved documents (max 21). Returns True if gold titles are a subset of retrieved titles after normalization.

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

