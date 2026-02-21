PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims using the HoVer (Hover-nlp) dataset. The system uses claim decomposition with focused sub-query retrieval and relevance scoring to find the most relevant supporting documents for veracity assessment of claims.

**Key Modules**:
- **HoverMultiHopPipeline** (hover_pipeline.py): Top-level pipeline implementing the complete claim decomposition retrieval strategy with sub-query generation and relevance-based scoring
- **ClaimDecomposer** (hover_pipeline.py): DSPy Signature for decomposing claims into 2-3 focused sub-queries targeting different aspects needed for multi-hop reasoning (main entities, relationships, comparative facts)
- **RelevanceScorer** (hover_pipeline.py): DSPy Signature for scoring individual documents on a 0-10 scale based on relevance to the claim
- **HoverMultiHop** (hover_program.py): Legacy DSPy module implementing 3-hop retrieval logic (not currently used)
- **hoverBench** (hover_data.py): Dataset handler that loads and filters HoVer dataset to 3-hop examples, creating train/test splits
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting document titles are retrieved (maximum 21 documents)

**Data Flow**:
1. **Stage 1 - Initial Retrieval**: Retrieve k=100 documents using the original claim as query (broad search)
2. **Stage 2 - Claim Decomposition**: Decompose the claim directly into 2 focused sub-queries that target different aspects needed for multi-hop reasoning, independent of initially retrieved documents
3. **Stage 3 - Sub-Query Retrieval**: Retrieve k=50 documents for each of the 2 sub-queries (total 3 searches: 1 broad + 2 focused = up to 200 documents)
4. **Stage 4 - Document Deduplication**: Combine all retrieved documents (initial 100 + sub-query 100) and deduplicate by normalized document title
5. **Stage 5 - Relevance Scoring**: Apply simple single-pass RelevanceScorer to each document, assigning a score from 0-10, then select top 21 documents by score (avoids expensive list generation of listwise reranker)
6. Output: Final 21 highest-scored documents as retrieved_docs prediction

**Metric**: discrete_retrieval_eval compares normalized gold document titles against retrieved document titles, returning True if all gold titles are found within the retrieved set (subset check).

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

