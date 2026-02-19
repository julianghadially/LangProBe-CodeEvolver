PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a two-stage retrieval system with LLM-based pointwise reranking for fact-checking claims from the HoVer dataset. The system generates diverse search queries, retrieves candidate documents, and uses LLM-based relevance scoring to select the most relevant documents for claim verification.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level module implementing a two-stage retrieval pipeline with LLM reranking
- **ClaimQueryGenerator**: DSPy signature that generates two diverse queries (entity-focused and relation-focused) from a claim
- **DocumentRelevanceScorer**: DSPy signature that scores individual documents for relevance to the claim (0-10 scale)
- **HoverMultiHop**: The legacy core DSPy program (no longer used in the current pipeline)
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Query Generation: Use ClaimQueryGenerator to create two diverse queries (entity-focused and relation-focused)
3. Initial Retrieval: Retrieve k=50 documents per query using ColBERTv2 (100 total documents)
4. Deduplication: Remove duplicate documents by title, keeping first occurrence
5. Relevance Scoring: Score each unique document using DocumentRelevanceScorer with ChainOfThought
6. Reranking: Sort documents by relevance score (descending) and return top 21

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

