PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a Named Entity-First Retrieval system with keyword fallback for multi-hop document retrieval on the HoVer (Hover-nlp) dataset. The system extracts named entities directly from claims, performs targeted entity-based retrieval, then adds contextual queries for bridging documents, followed by listwise LLM reranking to find the most relevant supporting documents for fact-checking claims.

**Key Modules**:
- **HoverMultiHopPipeline** (hover_pipeline.py): Top-level pipeline implementing the Named Entity-First two-stage retrieval strategy with direct entity retrieval, contextual query generation, and listwise reranking
- **ExtractNamedEntities** (hover_pipeline.py): DSPy Signature for extracting proper nouns/named entities (people, titles, places, organizations) and key descriptive phrases directly from the claim itself
- **ContextualQueryGenerator** (hover_pipeline.py): DSPy Signature for generating 1-2 contextual queries that combine entities with key relationships from the claim to find bridging documents
- **ListwiseReranker** (hover_pipeline.py): DSPy Signature for scoring and ranking all retrieved documents based on multi-hop reasoning chain relevance
- **EntityExtraction** (hover_pipeline.py): Legacy DSPy Signature for extracting entities from retrieved documents (not currently used)
- **EntityQueryGenerator** (hover_pipeline.py): Legacy DSPy Signature for generating entity queries (not currently used)
- **HoverMultiHop** (hover_program.py): Legacy DSPy module implementing 3-hop retrieval logic (not currently used)
- **hoverBench** (hover_data.py): Dataset handler that loads and filters HoVer dataset to 3-hop examples, creating train/test splits
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting document titles are retrieved (maximum 21 documents)

**Data Flow**:
1. **Named Entity Extraction**: Extract proper nouns/named entities (people, titles, places, organizations) and descriptive phrases directly from the claim using ExtractNamedEntities module
2. **Stage 1 - Entity Retrieval**: For each extracted named entity and descriptive phrase, perform direct retrieval with k=30 using the entity name itself as the query (no query generation). This ensures entities like "Josh Flitter", "The Broken Tower", "Hart Crane" are directly targeted.
3. **Stage 2 - Contextual Retrieval**: Generate 1-2 contextual queries combining entities with key relationships from the claim using ContextualQueryGenerator, then retrieve k=20 documents per query to find bridging documents for multi-hop reasoning
4. **Document Merging**: Combine all retrieved documents from both stages (up to ~80 total: multiple entities × 30 + 2 queries × 20) and deduplicate by normalized document title
5. **Listwise Reranking**: Apply ListwiseReranker using ChainOfThought reasoning to score all unique documents based on multi-hop reasoning chain support, selecting top 21 most relevant documents
6. Output: Final 21 highest-ranked documents as retrieved_docs prediction

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

