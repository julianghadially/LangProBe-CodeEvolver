PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
The HoVer (fact verification) system is a confidence-weighted iterative retrieval pipeline designed to identify supporting documents for fact-checking claims. It uses entity extraction, coverage analysis, and adaptive retrieval across three hops with document deduplication and reranking to discover the most relevant documents.

## Key Modules
- **HoverMultiHopPipeline**: Top-level wrapper that initializes the ColBERTv2 retrieval model and executes the program
- **HoverMultiHop**: Core program implementing confidence-weighted 3-hop retrieval with entity tracking, coverage analysis, deduplication, and relevance-based reranking
- **EntityExtractor**: Signature that extracts 2-3 key entities/topics from the claim
- **CoverageAnalyzer**: Signature that analyzes which entities are well-covered vs under-covered and provides confidence scores
- **RelevanceScorer**: Signature that scores each document's relevance to the claim (0-10 scale)
- **hoverBench**: Dataset loader filtering HoVer dataset examples to 3-hop cases (26K+ training examples from hover-nlp/hover)
- **hover_utils**: Contains the evaluation metric and document counting utilities

## Data Flow
1. Extract 2-3 key entities from the claim using EntityExtractor
2. Initialize empty deduplication set to track document titles
3. For each of 3 hops:
   - Use CoverageAnalyzer to identify which entities lack coverage
   - Generate focused queries for under-covered entities (max 3 queries/hop)
   - Retrieve k=25 documents per query using ColBERTv2
   - Deduplicate by filtering out documents with titles already seen
   - Add unique documents to collection
4. After all hops, use RelevanceScorer to score each unique document (0-10)
5. Rerank and return top 21 highest-scoring documents

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

