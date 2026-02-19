PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a self-critique iterative retrieval system for fact-checking claims from the HoVer dataset. The system uses an adaptive retrieval approach with verification and refinement to find relevant supporting documents, inspired by FIRE and SELF-RAG research. It employs a ColBERTv2 retriever with chain-of-thought reasoning for document relevance verification and ranking.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level pipeline implementing self-critique iterative retrieval architecture with document verification and ranking
- **DocumentRelevanceVerifier**: DSPy signature that evaluates retrieved documents for relevance and sufficiency, providing refinement suggestions
- **DocumentRanker**: DSPy signature that ranks all retrieved documents by relevance to the claim
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Initial retrieval: Fetch k=10 documents using the claim as query
3. Self-critique loop (max 2 iterations, staying within 3 total retrievals):
   a. DocumentRelevanceVerifier evaluates current documents for relevance_score, is_sufficient, and suggested_refinement
   b. If is_sufficient=False and under retrieval limit, generate refined query from suggested_refinement
   c. Retrieve k=10 more documents with refined query
   d. Repeat until sufficient or max iterations reached
4. Deduplicate all retrieved documents
5. If >21 documents, use DocumentRanker to rank by relevance and select top 21
6. Return final ranked documents as retrieved_docs

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

