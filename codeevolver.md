PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a parallel entity-specific document retrieval system for fact-checking claims from the HoVer dataset. The system identifies named entities in claims and performs parallel retrieval for each entity, then uses relevance scoring to select the most valuable documents. This architecture eliminates the summarization bottleneck that loses secondary entity information and directly targets all entities in multi-entity comparison claims.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level pipeline implementing parallel entity-specific retrieval with relevance scoring
- **EntityIdentificationSignature**: DSPy signature that extracts 2-4 distinct named entities from claims
- **DocumentRelevanceScorer**: DSPy module that scores documents (0-10) based on relevance for claim verification
- **HoverMultiHop**: (Legacy) The original 3-hop retrieval program with summarization
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Extract 2-4 named entities (people, places, organizations, works) from the claim using EntityIdentificationSignature
3. Perform parallel entity-specific retrieval: retrieve k=10 documents for each entity (up to 3 entities max, respecting 3-search constraint)
4. Score each retrieved document (0-10) using DocumentRelevanceScorer based on claim verification relevance
5. Sort documents by relevance score in descending order
6. Select top 21 documents with diversity constraint: maximum 8 documents per entity to ensure balanced coverage
7. Return selected 21 documents as retrieved_docs

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

