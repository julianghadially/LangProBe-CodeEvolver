PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a Query Perspective Ensemble document retrieval system for fact-checking claims from the HoVer dataset. The system uses three parallel retrieval perspectives with different reasoning strategies and adaptive k-values to find relevant supporting documents, followed by LLM-based relevance scoring to select the best 21 documents.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level module implementing Query Perspective Ensemble with Adaptive k-distribution. Contains four DSPy signature sub-modules for query generation and document scoring.
- **DirectEntityExtractor**: DSPy Signature using `dspy.Predict` to extract key entities/terms for direct factual retrieval (k=10)
- **RelationalQueryGenerator**: DSPy Signature using `dspy.ChainOfThought` to identify relationships/connections in claims (k=8)
- **ContradictionQueryGenerator**: DSPy Signature using `dspy.ChainOfThought` to generate queries finding contradictory evidence (k=5)
- **RelevanceScorer**: DSPy Signature using `dspy.Predict` to score document relevance to the claim (0-10 scale)
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Three query perspectives are generated IN PARALLEL (not sequentially):
   - DirectEntityExtractor extracts entities → retrieve k=10 docs
   - RelationalQueryGenerator identifies relationships → retrieve k=8 docs
   - ContradictionQueryGenerator generates contradictory queries → retrieve k=5 docs
3. All 23 documents (10+8+5) are collected from parallel retrievals
4. RelevanceScorer scores each document's relevance to the original claim
5. Top 21 documents by relevance score are selected and returned as retrieved_docs

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

