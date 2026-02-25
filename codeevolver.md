PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements a two-phase "retrieve-then-score" multi-hop document retrieval system for the HoVer (Hover-nlp) claim verification benchmark using DSPy. The system performs iterative retrieval across three hops to maximize recall, then uses LLM-based relevance scoring to identify the most relevant supporting documents for fact-checking claims that require evidence from multiple sources.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level pipeline wrapper that implements the two-phase architecture. Phase 1: Initializes ColBERTv2 retrieval and calls HoverMultiHop to retrieve ~36 documents (k=12 per hop × 3 hops). Phase 2: Deduplicates documents, scores each using DocumentRelevanceScorer with chain-of-thought reasoning, and returns the top 21 highest-scored documents. Serves as the entry point for the evaluation framework.

**DocumentRelevanceScorer** (`hover_pipeline.py`): DSPy ChainOfThought module that evaluates document relevance by taking a claim and document as input, outputting reasoning and a relevance score (1-10). This LLM-based scoring replaces sole reliance on ColBERT ranking to better identify supporting evidence.

**HoverMultiHop** (`hover_program.py`): Core retrieval logic implementing a 3-hop iterative retrieval strategy. Each hop retrieves k=12 documents (configurable), uses Chain-of-Thought prompting to summarize findings, and generates refined queries for subsequent hops. Returns all retrieved documents for downstream scoring.

**hover_utils**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting documents are found within the top 21 retrieved documents.

**hover_data**: Loads and preprocesses the HoVer dataset, filtering for 3-hop examples and formatting them for DSPy evaluation.

## Data Flow
1. Input claim enters via `HoverMultiHopPipeline.forward()`
2. **Phase 1 - Retrieval (maximizing recall)**:
   - Hop 1: Retrieve k=12 documents directly from claim, generate summary
   - Hop 2: Create refined query from claim+summary_1, retrieve k=12 more documents, summarize
   - Hop 3: Create query from claim+both summaries, retrieve k=12 final documents
   - Total: ~36 documents retrieved
3. **Phase 2 - Scoring and Selection (maximizing precision)**:
   - Deduplicate documents by title to get unique set
   - Score each unique document using DocumentRelevanceScorer (LLM evaluates relevance with reasoning)
   - Sort by relevance score (1-10) descending
   - Return top 21 highest-scored documents as `retrieved_docs`

## Metric
The `discrete_retrieval_eval` metric computes recall@21: whether all gold supporting document titles from `supporting_facts` are present in the retrieved set. Success requires the retrieval pipeline to discover all necessary evidence documents within the 21-document budget. The two-phase architecture maximizes recall through over-retrieval (k=12), then uses LLM reasoning to select the most relevant subset.

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

