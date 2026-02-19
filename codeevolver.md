PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims from the HoVer dataset. The system uses entity-based retrieval with BM25-style lexical reranking to find relevant supporting documents for a given claim, using a ColBERTv2 retriever and entity extraction to focus on documents containing specific entities and terminology mentioned in the claim.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level wrapper that initializes the ColBERTv2 retriever and implements entity-based retrieval with lexical reranking
- **EntityExtractionSignature**: DSPy signature for extracting 4-6 named entities (people, places, works, events) from claims
- **HoverMultiHop**: The original DSPy program implementing the 3-hop retrieval logic with summarization (kept for reference but not used in current implementation)
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Entity Extraction: Extract 4-6 named entities from the claim using EntityExtractionSignature
3. Query Generation: Use the first 3 entities as simple direct queries (entity name or 2-3 word phrases)
4. Retrieval: Retrieve k=100 documents per entity query (maximum 3 queries total)
5. Deduplication: Remove duplicate documents from the combined retrieval results
6. Lexical Reranking: Score all unique documents using n-gram matching (unigrams, bigrams, trigrams) with term rarity weighting (IDF-like scoring)
7. Selection: Return top 21 documents with highest lexical overlap scores as retrieved_docs

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

