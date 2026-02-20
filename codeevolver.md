PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This system performs multi-hop document retrieval for claim verification using the HoVer dataset. It retrieves relevant supporting documents through a three-stage entity-focused retrieval strategy with intelligent reranking, targeting all key entities in multi-hop claims before selecting the most relevant documents.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level pipeline implementing the three-stage retrieval strategy with entity extraction, targeted querying, and reranking
- `AnalyzeClaimEntities`: DSPy Signature using CoT reasoning to extract 2-3 key entities and concepts from the claim
- `GenerateEntityQueries`: DSPy Signature that generates 2-3 targeted search queries, one per entity/concept
- `RerankDocuments`: DSPy Signature using CoT reasoning to assign relevance scores (0-100) to all retrieved documents
- `HoverMultiHop`: Legacy core program implementing the 3-hop retrieval strategy (no longer used by pipeline)
- `hover_utils.py`: Contains the evaluation metric `discrete_retrieval_eval` that validates retrieval quality
- `hover_data.py`: Benchmark class managing the HoVer dataset, filtering for 3-hop examples

**Data Flow**:
1. **Stage 1 - Entity Extraction**: Analyze the claim using CoT reasoning to extract 2-3 key entities, concepts, or topics central to verification
2. **Stage 2 - Query Generation**: Generate 2-3 targeted search queries, one focused on each extracted entity in the context of the claim
3. **Stage 3 - Bulk Retrieval**: For each query, retrieve k=50 documents using dspy.Retrieve, yielding up to 150 total documents (deduplicated)
4. **Stage 4 - Intelligent Reranking**: Use CoT reasoning to analyze all retrieved documents and assign relevance scores (0-100) based on their importance for claim verification
5. **Stage 5 - Top-K Selection**: Sort all documents by relevance score (descending) and select the top 21 most relevant documents

The entity-focused strategy ensures that all key entities in multi-hop claims receive dedicated retrieval queries before reranking, addressing the limitation of sequential hop-based approaches. The large-scale retrieval (k=50 per query) followed by intelligent reranking maximizes recall before precision optimization, ensuring critical documents aren't missed.

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

