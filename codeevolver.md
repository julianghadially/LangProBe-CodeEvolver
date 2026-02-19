PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims from the HoVer dataset. The system performs three iterative retrieval hops with deduplication and coverage-based reranking to find relevant supporting documents for a given claim, using a ColBERTv2 retriever and targeted query generation strategies.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level pipeline that implements a three-hop retrieval strategy with deduplication and coverage-based reranking. Contains three DSPy Signature classes (EntityQueryGenerator, GapAnalysis, KeyTermExtractor) as sub-modules used in the forward() method.
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. **Hop 1**: Generate 2-3 entity-focused queries from the claim using EntityQueryGenerator, retrieve k=50 documents per query, deduplicate by title
3. **Hop 2**: Analyze retrieved documents using GapAnalysis to identify missing entities/concepts mentioned in the claim, generate 1-2 gap-filling queries, retrieve k=30 documents per query, deduplicate by title
4. **Hop 3**: Extract still-missing key terms using KeyTermExtractor, perform final targeted retrieval with k=20 documents, deduplicate by title
5. **Coverage-based reranking**: Score each unique document by counting how many claim entities/concepts it mentions, select top 21 documents by coverage score
6. Return final 21 documents as retrieved_docs

**Key Features**:
- Entity extraction and coverage scoring to prioritize documents mentioning multiple claim concepts
- Iterative gap-filling strategy to ensure comprehensive coverage
- Title-based deduplication after each hop to eliminate redundancy
- Progressive retrieval with decreasing k values (50→30→20) for efficiency

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

