PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims using the HoVer (Hop Verification) dataset. The system performs iterative retrieval and summarization to find supporting documents relevant to a given claim through a 3-hop reasoning process with two-stage retrieval architecture (over-retrieval + LLM-based re-ranking).

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level wrapper that initializes the ColBERTv2 retriever and orchestrates the retrieval pipeline
- `HoverMultiHop`: Core program implementing 3-hop retrieval logic with query generation, summarization, and two-stage retrieval
- `DocumentReranker`: LLM-based re-ranking module that scores documents 0-10 for relevance to the claim using dspy.ChainOfThought
- `hover_data.py`: Data loader that filters HoVer dataset to 3-hop examples (claim-fact pairs requiring 3 documents)
- `hover_utils.py`: Evaluation utilities including the `discrete_retrieval_eval` metric

**Data Flow**:
1. Input claim is used to retrieve k=30 documents (Hop 1 over-retrieval)
2. Documents are summarized, then used to generate a refined query for Hop 2
3. Hop 2 retrieves k=30 more documents, summarizes with context from Hop 1
4. Summaries from Hops 1-2 inform the query for Hop 3, retrieving k=30 final documents
5. All 90 documents (30×3 hops) are passed to the DocumentReranker
6. DocumentReranker uses LLM-based scoring (with justification) to score each document 0-10 for relevance
7. Top 21 highest-scored documents are selected and returned as `retrieved_docs`

**Two-Stage Retrieval**: The system casts a wider net by over-retrieving (k=30 per hop, 90 total), then intelligently filters using LLM-based re-ranking to select the most relevant 21 documents. This addresses issues where critical supporting documents are missed with narrow retrieval.

**Metric**: `discrete_retrieval_eval` checks if all gold supporting document titles (from `supporting_facts`) are present in the retrieved documents (max 21). Returns True if gold titles are a subset of retrieved titles after normalization.

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

