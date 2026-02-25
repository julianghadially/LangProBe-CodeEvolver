PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is an entity-focused document retrieval system for fact-checking claims from the HoVer (Fact Extraction and VERification over Unstructured and Structured information) dataset. The system uses a three-stage retrieval approach that focuses on extracting named entities from claims and retrieving biographical/entity-specific documents that serve as critical supporting facts.

**Key Modules**:
- **HoverMultiHopPipeline** (hover_pipeline.py): Top-level pipeline that implements entity-focused retrieval with five stages: (1) entity extraction, (2) entity-based retrieval, (3) claim-based retrieval, (4) deduplication, and (5) BM25 re-ranking. Inherits from LangProBeDSPyMetaProgram and dspy.Module.
- **EntityExtraction** (hover_pipeline.py): DSPy Signature that extracts 2-4 key named entities (people, organizations, places) from the claim using LLM predictions.
- **hoverBench** (hover_data.py): Dataset loader that filters HoVer dataset examples to only include 3-hop cases, formats them as DSPy examples with claims and supporting facts.
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting document titles are present in the retrieved documents (max 21 documents).

**Data Flow**:
1. Extract 2-4 named entities from the claim using EntityExtraction signature (limited to 3 entities to stay within search limits)
2. For each entity, retrieve k=10 documents using ColBERTv2 retrieval (up to 30 documents total from entities)
3. Retrieve k=30 documents for the full claim using ColBERTv2 retrieval
4. Combine all retrieved passages and deduplicate
5. Re-rank deduplicated passages using BM25 scoring algorithm to select the top 21 most relevant documents
6. Return final 21 documents as retrieved_docs

**Metric**: discrete_retrieval_eval evaluates whether all gold supporting fact documents are subset of retrieved documents, enforcing a maximum of 21 retrieved documents.

**Key Design Decision**: The entity-focused approach ensures the system retrieves specific biographical and entity-related documents that are often critical supporting facts, rather than only topically related documents. The BM25 re-ranking ensures the final 21 documents are the most relevant based on term matching with the original claim.

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

