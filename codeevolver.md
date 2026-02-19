PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims from the HoVer dataset. The system uses entity extraction and targeted query generation to find relevant supporting documents, employing a ColBERTv2 retriever with a two-stage reranking pipeline to prioritize bridge documents.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level pipeline implementing entity-aware retrieval with named entity extraction, multi-query retrieval, and two-stage reranking
- **WikipediaTitleExtractor**: DSPy signature for extracting potential Wikipedia article titles from claims
- **Query Generators**: Three specialized signatures (EntitySearchQuery, RelationshipQuery, AttributeQuery) for generating targeted queries
- **RelevanceScorer**: DSPy signature for LLM-based relevance scoring in the reranking stage
- **HoverMultiHop**: The original 3-hop retrieval program (now a fallback component)
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Entity Extraction: Use WikipediaTitleExtractor to identify named entities from the claim
3. Query Generation: Generate 3 targeted queries:
   - Query 1: Direct entity search using extracted titles
   - Query 2: Relationship query connecting the entities
   - Query 3: Attribute/property query for descriptive facts
4. Document Retrieval: Retrieve k=50 documents per query (total ~150 documents after deduplication)
5. Two-Stage Reranking:
   - Stage 1: Boost documents with titles matching extracted entities using fuzzy string matching (threshold 0.85)
   - Stage 2: Apply LLM-based relevance scoring on top 50 candidates
6. Return top 21 documents as retrieved_docs

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

