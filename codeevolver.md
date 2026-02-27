PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

This program implements a multi-hop document retrieval system for the HoVer (Hover-nlp) fact verification task using DSPy. The system retrieves supporting documents for a given claim through an entity-focused multi-query strategy with relevance-based reranking.

**Key Modules:**

1. **HoverMultiHopPipeline** (`hover_pipeline.py`): The top-level pipeline wrapper that implements an entity-focused multi-query retrieval strategy:
   - **ExtractClaimEntities**: DSPy signature that identifies 2-3 key entities/topics from the claim
   - **EntityQueryGenerator**: DSPy signature that generates focused retrieval queries for each entity
   - **DocumentRelevanceScorer**: DSPy signature that scores document relevance to the claim (0-10 scale)
   - Retrieves k=25 documents per entity query (up to 75 total documents)
   - Applies relevance-based reranking to consolidate to top 21 unique documents
   - Initializes ColBERTv2 retriever with a remote API endpoint

2. **HoverMultiHop** (`hover_program.py`): Legacy multi-hop retrieval program (no longer used in current pipeline):
   - Previously performed three sequential retrieval hops with summarization
   - Kept for backward compatibility

3. **hover_data.py**: Manages dataset loading from the HoVer benchmark, filtering for 3-hop examples and creating DSPy Example objects.

4. **hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` which checks if all gold supporting document titles are present in the retrieved documents (subset match).

**Data Flow:**
Claim → Extract Entities (2-3) → Generate Entity-Focused Queries (max 3) → Retrieve Documents (k=25 per query, up to 75 total) → Deduplicate → Score Relevance (0-10) → Rerank and Select Top 21 Documents

**Metric:** The `discrete_retrieval_eval` metric returns True if all gold standard supporting document titles are found within the predicted retrieved documents (max 21), using normalized text matching.

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

