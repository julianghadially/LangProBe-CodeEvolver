PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims using the HoVer (Hop Verification) dataset. The system performs iterative retrieval and summarization to find supporting documents relevant to a given claim through a 3-hop reasoning process with entity-aware prioritization.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level pipeline that implements entity-based retrieval with adaptive k values (15+15+5) and score-based reranking to return top 21 documents. Uses custom DSPy signatures for entity identification, query generation, and document relevance scoring.
- `HoverMultiHop`: Legacy core program implementing 3-hop retrieval logic (not currently used by pipeline)
- `hover_data.py`: Data loader that filters HoVer dataset to 3-hop examples (claim-fact pairs requiring 3 documents)
- `hover_utils.py`: Evaluation utilities including the `discrete_retrieval_eval` metric

**Data Flow**:
1. **Entity Identification**: Extract proper nouns (people, bands, works, organizations, places) from the claim using `EntityIdentification` signature
2. **Hop 1 (k=15)**: Generate entity-specific queries formatted as Wikipedia article titles using `EntityQueryGenerator`, retrieve 15 documents, and summarize
3. **Hop 2 (k=15)**: Generate bridging query connecting entities using `BridgingQueryGenerator`, retrieve 15 documents, and summarize with Hop 1 context
4. **Hop 3 (k=5)**: Generate verification query using `VerificationQueryGenerator`, retrieve 5 documents
5. **Reranking**: Score all 35 documents (15+15+5) using `DocumentRelevanceScorer` based on entity relevance and claim relationships
6. Return top 21 highest-scoring documents as `retrieved_docs`

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

