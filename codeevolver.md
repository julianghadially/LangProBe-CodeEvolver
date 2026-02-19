PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a gap-aware multi-hop document retrieval system for fact-checking claims from the HoVer dataset. The system performs three iterative retrieval hops with self-critique and gap analysis to systematically identify missing information and target retrieval to fill specific gaps, using a ColBERTv2 retriever with relevance scoring and reranking.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level pipeline implementing gap-aware retrieval with three DSPy signature classes (GapAnalysisSignature, GapAwareQueryGenerator, RelevanceScorer) for identifying missing entities and scoring document relevance
- **HoverMultiHop**: The original core DSPy program implementing the 3-hop retrieval logic (currently not used by HoverMultiHopPipeline)
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Hop 1: Retrieve k=30 documents from claim, score and rerank to top 7, summarize results
3. Gap Analysis 1: Use GapAnalysisSignature to identify missing entities (people, places, events, dates) not covered by hop 1 docs
4. Hop 2: Generate gap-aware query using GapAwareQueryGenerator targeting missing info, retrieve k=30, rerank to top 7
5. Gap Analysis 2: Identify remaining gaps after combining hop 1+2 documents
6. Hop 3: Generate final gap-aware query for remaining gaps, retrieve k=30, rerank to top 7
7. Return all 21 documents (3 hops × 7 docs each) as retrieved_docs

**Gap-Aware Components**:
- **GapAnalysisSignature**: DSPy signature for analyzing claim vs retrieved docs to identify specific missing information entities
- **GapAwareQueryGenerator**: DSPy signature for generating targeted queries to fill identified information gaps
- **RelevanceScorer**: DSPy signature for scoring document relevance (0-10 scale) to the claim for reranking

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

