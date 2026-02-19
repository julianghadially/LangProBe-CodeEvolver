PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims from the HoVer dataset. The system uses a parallel query ensemble architecture with confidence-weighted fusion to find relevant supporting documents for a given claim, using a ColBERTv2 retriever and LM-based scoring to optimize document relevance.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level wrapper implementing parallel query ensemble architecture with three specialized query generators (EntityQueryGenerator, RelationQueryGenerator, TemporalQueryGenerator), retrieving 25 documents per query type (75 total), scoring all documents with DocumentConfidenceScorer, and returning top 21 deduplicated documents sorted by confidence
- **HoverMultiHop**: Legacy core DSPy program implementing the 3-hop retrieval logic with summarization (no longer actively used)
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Three query types generated in parallel using ChainOfThought: entity-focused, relation-focused, and temporal-focused queries
3. Each query retrieves k=25 documents from ColBERTv2 (75 documents total across 3 parallel queries)
4. DocumentConfidenceScorer uses dspy.Predict to score each of the 75 documents for relevance (0.0-1.0 confidence)
5. Documents deduplicated by title, keeping highest confidence score for each unique document
6. Sorted by confidence descending and top 21 unique documents returned as retrieved_docs

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

