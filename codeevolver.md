PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
The HoVer (fact verification) system is a multi-hop document retrieval pipeline with LLM-powered listwise reranking, designed to identify supporting documents for fact-checking claims. It uses an iterative retrieval approach with three sequential "hops" to progressively discover relevant documents, followed by a sophisticated sliding-window reranking phase that uses chain-of-thought reasoning to select the most relevant documents.

## Key Modules
- **HoverMultiHopPipeline**: Top-level wrapper that initializes the ColBERTv2 retrieval model, executes the multi-hop program, and applies listwise reranking
- **ListwiseReranker**: DSPy Signature that uses chain-of-thought reasoning to evaluate and rank documents by relevance to the claim
- **HoverMultiHop**: Core program implementing the 3-hop retrieval strategy with summarization at each hop (configurable k parameter)
- **hoverBench**: Dataset loader filtering HoVer dataset examples to 3-hop cases (26K+ training examples from hover-nlp/hover)
- **hover_utils**: Contains the evaluation metric and document counting utilities

## Data Flow
1. Initial claim is used to retrieve k=21 documents (hop 1)
2. Hop 1 docs are summarized using ChainOfThought
3. Summary generates a refined query for hop 2, retrieving k=21 more documents
4. Both summaries inform hop 3 query generation for final k=21 documents
5. All retrieved documents (63 total) are collected
6. Sliding-window listwise reranking is applied:
   - Documents are split into overlapping windows of 30 documents each (stride=15)
   - Each window is reranked using ListwiseReranker with ChainOfThought reasoning
   - Scores are aggregated across windows using reciprocal rank scoring
   - Top 21 documents are selected based on averaged scores and returned

## Optimization Metric
`discrete_retrieval_eval` checks if all gold supporting documents (from supporting_facts) are present in the retrieved set (max 21 docs). Returns binary score: True if gold_titles ⊆ found_titles, False otherwise. Document titles are normalized and matched using the first segment before " | " delimiter.

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

