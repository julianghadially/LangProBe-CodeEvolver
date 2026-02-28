PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims using the HoVer (Hop Verification) dataset. The system performs parallel multi-query retrieval with coverage-based reranking to find supporting documents relevant to a given claim, ensuring comprehensive entity coverage across multi-hop claims.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level module that implements parallel multi-query retrieval with coverage-based reranking. Generates 3 diverse queries, retrieves 35 documents per query (105 total), then iteratively reranks to select 21 documents with maximum entity coverage diversity.
- `DiverseQueryGeneration`: DSPy signature for generating 3 diverse search queries targeting different entities/aspects of the claim
- `CoverageScoring`: DSPy signature for scoring documents (0-10) based on unique entity/fact coverage relative to already-selected documents
- `HoverMultiHop`: Legacy 3-hop retrieval program (no longer used in current pipeline)
- `hover_data.py`: Data loader that filters HoVer dataset to 3-hop examples (claim-fact pairs requiring 3 documents)
- `hover_utils.py`: Evaluation utilities including the `discrete_retrieval_eval` metric

**Data Flow**:
1. Input claim is used to generate 3 diverse search queries via `DiverseQueryGeneration` signature
2. Each query retrieves k=35 documents using ColBERTv2 retriever (total 105 documents)
3. Duplicates are removed from the 105 documents
4. Iterative reranking loop selects 21 documents:
   - For each of 21 iterations, score all remaining documents using `CoverageScoring`
   - Select the highest-scoring document that adds new entity coverage
   - Track covered entities to guide subsequent selections
5. Final 21 documents with maximum entity coverage diversity are returned as `retrieved_docs`

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

