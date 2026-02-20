PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This system performs multi-query diversified document retrieval for claim verification using the HoVer dataset. It retrieves relevant supporting documents through a parallel query generation strategy that analyzes claims from multiple perspectives, then applies diversity-based reranking to select the most relevant and diverse documents.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level pipeline implementing Parallel Multi-Query Diversified Retrieval architecture with three specialized query generators and diversity-based reranking
- `EntityFocusedQuerySignature`, `ComparativeQuerySignature`, `ContextualQuerySignature`: Three parallel query generation signatures that extract different aspects from claims
- `DocumentScoringSignature`: Reranking signature that scores documents based on relevance and diversity
- `hover_utils.py`: Contains the evaluation metric `discrete_retrieval_eval` that validates retrieval quality
- `hover_data.py`: Benchmark class managing the HoVer dataset, filtering for 3-hop examples

**Data Flow**:
1. Input claim is analyzed by three parallel query generators:
   - EntityFocusedQuery: Extracts and queries key entities mentioned in the claim
   - ComparativeQuery: Generates queries for comparative/contrasting elements
   - ContextualQuery: Generates broader contextual queries for background information
2. Each query generator retrieves k=21 documents independently (total 63 documents across 3 searches)
3. Retrieved documents are deduplicated by title to remove redundant entries
4. Diversity-based reranking module iteratively selects documents using ChainOfThought scoring:
   - Each document is scored on (a) relevance to the claim and (b) uniqueness/diversity compared to already-selected documents
   - The reranker selects exactly 21 diverse, high-quality documents from the candidate pool
5. Final set of 21 documents is returned for evaluation

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

