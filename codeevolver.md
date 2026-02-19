PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is an entity-focused document retrieval system with MMR reranking for fact-checking claims from the HoVer dataset. The system uses a two-stage approach: (1) entity extraction and focused retrieval, and (2) Maximal Marginal Relevance (MMR) reranking to select diverse and relevant documents for a given claim using ColBERTv2 retriever.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level pipeline that implements entity-focused retrieval with MMR reranking
- **EntityExtractionSignature**: DSPy signature for extracting 2-3 key entities from claims
- **EntityQueryGeneratorSignature**: DSPy signature for generating focused search queries per entity
- **MMRReranker**: Implements Maximal Marginal Relevance algorithm for document diversity (lambda=0.7)
- **HoverMultiHop**: The original 3-hop retrieval logic (not currently used)
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Stage 1: Extract 2-3 key entities from the claim using EntityExtractionSignature
3. Stage 2: For each of the first 2 entities (to stay under 3-query limit):
   - Generate a focused search query using EntityQueryGeneratorSignature
   - Retrieve k=100 documents per query using ColBERTv2
   - Deduplicate documents across queries
4. Stage 3: Apply MMR reranking to select final 21 documents:
   - Calculate relevance scores (cosine similarity between document embeddings and claim)
   - Iteratively select documents that maximize: λ * relevance - (1-λ) * max_similarity_to_selected
   - Uses sentence-transformers ('all-MiniLM-L6-v2') for embeddings
   - Lambda=0.7 balances relevance (70%) vs diversity (30%)
5. Return exactly 21 diverse, relevant documents as retrieved_docs

**Metric**: The discrete_retrieval_eval metric checks if all gold-standard supporting document titles (from supporting_facts) are present in the top 21 retrieved documents. Success requires 100% recall of gold documents within the 21-document limit. Documents are compared using normalized text matching on title keys.

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

