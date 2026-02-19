PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims from the HoVer dataset. The system performs three iterative retrieval hops using Pseudo-Relevance Feedback with LLM-Enhanced Term Extraction to find relevant supporting documents for a given claim, using a ColBERTv2 retriever with progressive query refinement based on relevance scoring and key term extraction.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level wrapper that implements the Pseudo-Relevance Feedback architecture with three custom DSPy signatures for relevance scoring, term extraction, and query expansion
- **DocumentRelevanceScorer**: DSPy signature that scores document relevance to a claim (1-5 scale) with reasoning
- **KeyTermExtractor**: DSPy signature that extracts key entities and phrases from relevant documents
- **ExpandedQueryGenerator**: DSPy signature that generates expanded search queries from key terms
- **HoverMultiHop**: Legacy core DSPy program (kept for backward compatibility but not used in new implementation)
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Hop 1: Retrieve k=15 documents from claim, score each with DocumentRelevanceScorer, keep top 5
3. Extract key entities/phrases from top 5 using KeyTermExtractor
4. Generate expanded query using ExpandedQueryGenerator with extracted terms
5. Hop 2: Retrieve k=15 documents using expanded query, score each, keep top 5
6. Extract new key entities/phrases from Hop 2 top 5 documents
7. Generate refined query for Hop 3 using newly extracted terms
8. Hop 3: Retrieve k=15 documents using refined query, score each, keep top 5
9. Apply set deduplication across all hops (maximum 15 unique top docs)
10. Score and rank remaining documents from all three hops
11. Pad with highest-scored unique documents to reach exactly 21 total documents
12. Return final 21 documents as retrieved_docs

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

