PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-entity parallel document retrieval system for fact-checking claims from the HoVer (Fact Extraction and VERification over Unstructured and Structured information) dataset. The system uses query decomposition to retrieve documents about all entities mentioned in multi-entity claims rather than following a single sequential path, then applies two-stage reranking to select the most relevant documents.

**Key Modules**:
- **HoverMultiHopPipeline** (hover_pipeline.py): Top-level pipeline wrapper that initializes the ColBERTv2 retrieval model and orchestrates the program execution. Inherits from LangProBeDSPyMetaProgram and dspy.Module.
- **HoverMultiHop** (hover_program.py): Core retrieval logic implementing parallel multi-entity query decomposition with two-stage reranking. Decomposes claims into 2-3 focused sub-queries, retrieves k=100 documents per sub-query (up to 300 total), applies two-stage reranking (fast filtering then detailed scoring), deduplicates by title, and selects top 21 unique documents by relevance score.
- **ClaimDecomposition** (hover_program.py): DSPy signature that takes a claim and outputs 2-3 focused sub-queries targeting different entities or concepts within the claim for parallel retrieval.
- **QuickRelevanceFilter** (hover_program.py): DSPy Predict signature for first-stage reranking that quickly filters documents with binary yes/no relevance decisions, reducing ~300 documents to ~60 using lightweight filtering before expensive scoring.
- **RelevanceScorer** (hover_program.py): DSPy ChainOfThought signature for second-stage reranking that scores each filtered document's relevance to the original claim on a 1-10 scale with detailed reasoning, enabling intelligent final selection.
- **hoverBench** (hover_data.py): Dataset loader that filters HoVer dataset examples to only include 3-hop cases, formats them as DSPy examples with claims and supporting facts.
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting document titles are present in the retrieved documents (max 21 documents).

**Data Flow**:
1. Input claim is decomposed into 2-3 focused sub-queries targeting different entities/concepts
2. Each sub-query retrieves k=100 documents in parallel (up to 300 total documents)
3. All retrieved documents are combined into a single pool
4. First-stage reranking: QuickRelevanceFilter applies fast binary yes/no filtering to reduce ~300 documents to ~60 potentially relevant documents
5. Second-stage reranking: RelevanceScorer applies detailed chain-of-thought scoring on the ~60 filtered documents, assigning 1-10 relevance scores
6. Documents are deduplicated by normalized title
7. Top 21 unique documents by relevance score are selected and returned as retrieved_docs

**Metric**: discrete_retrieval_eval evaluates whether all gold supporting fact documents are subset of retrieved documents, enforcing a maximum of 21 retrieved documents.

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

