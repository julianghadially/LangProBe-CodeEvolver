PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a coverage-tracking multi-hop document retrieval system for fact-checking claims from the HoVer dataset. The system performs three iterative retrieval hops to find relevant supporting documents for a given claim, using a ColBERTv2 retriever with explicit coverage analysis at each step to identify and target gaps in evidence rather than following associative links.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level pipeline that implements the complete coverage-tracking retrieval architecture with integrated coverage analysis, targeted query generation, and coverage-based reranking
- **CoverageAnalysisSignature**: DSPy signature that analyzes retrieved passages to identify which claim aspects are covered and which remain uncovered
- **TargetedQuerySignature**: DSPy signature that generates search queries explicitly focused on uncovered aspects
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Hop 1: Retrieve k=10 documents directly from claim using ColBERTv2
3. Coverage Analysis 1: Identify covered entities and uncovered aspects from the 10 documents
4. Hop 2: Generate targeted query focusing on uncovered_aspects, retrieve k=10 more documents
5. Coverage Analysis 2: Analyze all 20 documents to identify remaining uncovered aspects
6. Hop 3: Generate highly specific query targeting remaining gaps, retrieve k=10 documents (30 total)
7. Coverage-based reranking: Score all 30 documents by how well they address uncovered aspects using keyword-based heuristics
8. Deduplication: Remove duplicate documents by exact title match, keeping first occurrence
9. Return top 21 unique, reranked documents as retrieved_docs

**Coverage-Tracking Architecture**: The system explicitly tracks what aspects of the claim are covered vs uncovered at each hop. Each subsequent retrieval hop generates queries that target the specific gaps identified by coverage analysis, ensuring comprehensive evidence gathering rather than redundant retrieval. The final reranking step prioritizes documents that address previously uncovered aspects, maximizing coverage in the final 21-document output.

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

