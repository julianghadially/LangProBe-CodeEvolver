PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
The HoVer (fact verification) system is a sequential multi-hop chain reasoning architecture designed to identify supporting documents for fact-checking claims. It explicitly models the multi-hop reasoning structure by orchestrating hop-by-hop retrieval, starting with concrete entities and progressively retrieving bridging documents through a logical chain. The system uses hop chain extraction, completeness evaluation, targeted hop-specific queries, and position-aware reranking that prioritizes documents covering multiple hops in the reasoning chain.

## Key Modules
- **HoverMultiHopPipeline**: Top-level wrapper that initializes the ColBERTv2 retrieval model and executes the program
- **HoverMultiHop**: Core program implementing sequential hop-by-hop retrieval with chain completeness evaluation
- **HopChainExtractor**: Signature that analyzes the claim to identify the logical hop structure and entity bridges (e.g., "Hop 1: author of book → Hop 2: birth details of author")
- **HopTargetedQuery**: Signature that generates a single highly-focused query for a specific missing hop/bridge using previously retrieved documents as context
- **ChainCompletenessEvaluator**: Signature that evaluates whether all hops in the reasoning chain have bridging documents retrieved
- **DocumentRelevanceScorer**: Signature that scores each document's relevance (0-100) to the claim and entity chains
- **hoverBench**: Dataset loader filtering HoVer dataset examples to 3-hop cases (26K+ training examples from hover-nlp/hover)
- **hover_utils**: Contains the evaluation metric and document counting utilities

## Data Flow
1. **Hop Chain Extraction**: HopChainExtractor analyzes the claim to identify the logical hop structure and concrete entities that should be retrieved first
2. **Hop 1 Retrieval**: Retrieve k=25 documents for the most concrete entities mentioned in the claim using ColBERTv2
3. **Iterative Hop-by-Hop Retrieval** (max 2 additional hops for total of 3 queries):
   - ChainCompletenessEvaluator assesses whether all hops in the chain have bridging documents
   - If incomplete, HopTargetedQuery generates a single highly-focused query for the missing hop using previously retrieved documents as context
   - Retrieve k=20 documents for the targeted hop
   - Track which documents appear in multiple hops (bridge documents)
   - Repeat until chain is complete or max hops reached (3 total queries)
4. **Position-Aware Reranking**: Documents are scored using position (earlier = higher), multi-hop coverage bonus (documents covering multiple hops get +100 per hop), and early hop bonus (+50 for hop 1 documents with concrete entities)
5. Top 21 documents are returned based on combined scores

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

