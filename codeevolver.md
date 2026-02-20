PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This system performs multi-hop document retrieval for claim verification using the HoVer dataset. It uses a "Retrieve-Many, Rerank-to-21" architecture that leverages unlimited retrieval capacity to gather a large pool of candidate documents (150 total), then uses LLM-based listwise reranking to intelligently select the top 21 most relevant documents.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level pipeline implementing the Retrieve-Many, Rerank-to-21 strategy. Initializes ColBERTv2 retrieval model, performs 3 hops with k=50 each (150 total documents), then uses LLM reranking to select top 21 most relevant documents
- `RankDocumentRelevance`: DSPy Signature for listwise reranking that takes claim and document pool, outputs indices of top 21 most relevant documents
- `HoverMultiHop`: Core program implementing the 3-hop retrieval strategy with gap analysis, query generation, and summarization at each hop (currently not used in pipeline forward method)
- `GapAnalysis`: DSPy Signature for analyzing gaps between claim requirements and retrieved passages, outputting missing entities and coverage assessment
- `CreateQueryHop2` & `CreateQueryHop3`: DSPy Signatures for generating targeted queries informed by missing entities from gap analysis
- `hover_utils.py`: Contains the evaluation metric `discrete_retrieval_eval` that validates retrieval quality
- `hover_data.py`: Benchmark class managing the HoVer dataset, filtering for 3-hop examples

**Data Flow**:
1. **Hop 1**: Input claim is used for initial retrieval, fetching k=50 documents
2. **Hop 2**: Claim is used again for second retrieval pass, fetching k=50 additional documents
3. **Hop 3**: Claim is used for final retrieval pass, fetching k=50 more documents
4. All retrieved documents are combined into a pool of 150 documents
5. **Listwise Reranking**: ChainOfThought module with RankDocumentRelevance signature analyzes claim and all 150 documents, performing intelligent reranking to output indices of top 21 most relevant documents
6. Top 21 documents are selected based on reranked indices and returned

The Retrieve-Many, Rerank-to-21 architecture maximizes recall in the retrieval phase by casting a wide net (150 documents), then leverages LLM intelligence to optimize precision by selecting the 21 most relevant documents through listwise comparison and reasoning.

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

