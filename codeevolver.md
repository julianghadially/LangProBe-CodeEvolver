PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
The HoVer (fact verification) system is an iterative multi-round document retrieval pipeline with confidence-based self-verification loops, designed to identify supporting documents for fact-checking claims. It uses an entity-aware retrieval approach with parallel queries targeting different entity chains, confidence evaluation to identify information gaps, conditional targeted follow-up retrieval, and intelligent score-based reranking to select the most relevant documents.

## Key Modules
- **HoverMultiHopPipeline**: Top-level wrapper that initializes the ColBERTv2 retrieval model and executes the program
- **HoverMultiHop**: Core program implementing 2-round iterative retrieval with confidence-based self-verification loops
- **EntityAndGapAnalyzer**: Signature that extracts entity chains from claims and generates 2-3 parallel search queries
- **ConfidenceEvaluator**: Signature that assesses whether retrieved documents provide sufficient evidence to verify the claim, outputs confidence score (0-100) and identifies missing information gaps
- **TargetedQueryGenerator**: Signature that generates 1-2 highly targeted follow-up queries to address identified information gaps
- **ListwiseDocumentReranker**: Signature that evaluates all retrieved documents together to identify multi-hop relationships and document interdependencies, outputs ranked list of document indices
- **hoverBench**: Dataset loader filtering HoVer dataset examples to 3-hop cases (26K+ training examples from hover-nlp/hover)
- **hover_utils**: Contains the evaluation metric and document counting utilities

## Data Flow
1. **Round 1**: EntityAndGapAnalyzer extracts entity chains and generates 2-3 parallel queries targeting different aspects of the claim
2. Each query retrieves k=23 documents using ColBERTv2 (total: 46-69 documents across 2-3 queries)
3. Documents are deduplicated while preserving order
4. **Confidence Evaluation**: ConfidenceEvaluator assesses coverage and identifies missing entity chains, bridging entities, or factual gaps
5. **Round 2 (conditional)**: If confidence < 80, TargetedQueryGenerator creates 1-2 targeted follow-up queries addressing the gaps, each retrieving k=15 additional documents
6. All documents from both rounds are deduplicated
7. **Listwise Reranking**: ListwiseDocumentReranker evaluates all documents together (concatenated with separators) to identify multi-hop relationships and document interdependencies, outputting a ranked list of document indices
8. Documents are reordered based on the ranked indices and top 21 are returned

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

