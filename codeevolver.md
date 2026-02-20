PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This system performs multi-hop document retrieval for claim verification using the HoVer dataset. It retrieves relevant supporting documents through a three-stage query decomposition strategy: (1) decomposing claims into diverse queries, (2) broad retrieval, and (3) LLM-based semantic reranking for optimal coverage.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level pipeline that implements the three-stage retrieval strategy directly in its forward() method using query decomposition, multi-query retrieval, and semantic reranking
- `DecomposeClaimToQueries`: DSPy Signature that decomposes a claim into 3 distinct search queries by identifying different entities/concepts and generating synonyms or related terms for maximum coverage
- `RerankDocumentsByRelevance`: DSPy Signature that scores and reranks documents using LLM reasoning to identify which documents contain the most relevant supporting facts for the claim
- `HoverMultiHop`: Legacy program implementing 3-hop retrieval with gap analysis (not currently used by the pipeline)
- `hover_utils.py`: Contains the evaluation metric `discrete_retrieval_eval` that validates retrieval quality
- `hover_data.py`: Benchmark class managing the HoVer dataset, filtering for 3-hop examples

**Data Flow**:
1. **Stage 1 - Query Decomposition**: The input claim is decomposed into 3 distinct search queries using `DecomposeClaimToQueries`. Each query targets different entities, concepts, or aspects with variations and synonyms (e.g., for "Johnny Tremain film", generates queries like "Johnny Tremain 1957 movie", "Hal Stalmaster American Revolution film", "Walt Disney historical films")
2. **Stage 2 - Multi-Query Retrieval**: For each of the 3 queries, retrieve k=50 documents using `dspy.Retrieve(k=50)`, resulting in a combined pool of up to 150 documents
3. **Stage 3 - Semantic Reranking**: The combined document pool is passed to `RerankDocumentsByRelevance`, which uses LLM reasoning to score each document's relevance to the claim and returns the top 21 most relevant documents

This approach maintains the 3-query limit and 21-document output requirement while improving coverage of multi-hop reasoning chains through better query diversity and semantic reranking. The decomposition strategy ensures queries target different aspects of the claim, while semantic reranking ensures the most relevant supporting documents are prioritized.

**Metric**: `discrete_retrieval_eval` checks if all gold-standard supporting document titles are present in the retrieved set (up to 21 documents). Returns True only when all required documents are successfully retrieved, evaluating as a strict subset match.

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

