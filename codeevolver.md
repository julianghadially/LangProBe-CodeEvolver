PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements an entity-focused retrieval system for the HoVer (Hover-nlp) claim verification benchmark using DSPy. The system extracts key named entities from claims, performs targeted retrieval queries for each entity to find specific Wikipedia articles, then uses a multi-faceted scoring mechanism to identify the most relevant supporting documents for fact-checking claims that require evidence from multiple sources.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level pipeline implementing entity-focused retrieval. First, extracts 3-5 key named entities (people, places, works, events) from the claim using the ExtractKeyEntities module. Then performs up to 3 entity-based retrieval queries (k=35 documents each) to target specific Wikipedia articles. After deduplication, employs a three-tier scoring mechanism that prioritizes: (1) exact entity name matches in document titles (100 points per match), (2) claim keyword overlap with titles (10 points) and content (2 points), and (3) LLM-based relevance scoring (1-10 points). Returns exactly 21 top-scored documents. Serves as the entry point for the evaluation framework.

**ExtractKeyEntities** (`hover_pipeline.py`): DSPy Signature class that takes a claim as input and outputs a list of 3-5 key named entities (people, places, works, events) that are most helpful for finding supporting documents. Used to guide the entity-focused retrieval strategy.

**DocumentRelevanceScorer** (`hover_pipeline.py`): DSPy ChainOfThought module that evaluates document relevance by taking a claim and document as input, outputting reasoning and a relevance score (1-10). This LLM-based scoring serves as a baseline component of the multi-faceted scoring mechanism.

**HoverMultiHop** (`hover_program.py`): Legacy core retrieval logic implementing a 3-hop iterative retrieval strategy. Currently not used by HoverMultiHopPipeline but preserved in codebase. Each hop retrieves k=12 documents (configurable), uses Chain-of-Thought prompting to summarize findings, and generates refined queries for subsequent hops.

**hover_utils**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting documents are found within the top 21 retrieved documents.

**hover_data**: Loads and preprocesses the HoVer dataset, filtering for 3-hop examples and formatting them for DSPy evaluation.

## Data Flow
1. Input claim enters via `HoverMultiHopPipeline.forward()`
2. **Entity Extraction**:
   - ExtractKeyEntities module analyzes the claim and identifies 3-5 key named entities (people, places, works, events)
   - Entities are limited to the first 3 for retrieval queries (constraint compliance)
3. **Entity-Based Retrieval**:
   - For each of the 3 entities, perform a separate retrieval query using the entity name directly
   - Retrieve k=35 documents per entity query
   - Total: up to 105 documents retrieved (3 queries × 35 documents)
4. **Deduplication and Reranking**:
   - Deduplicate documents by title to get unique set
   - Score each document using three-tier mechanism:
     * Tier 1: Exact entity name matches in title (+100 per match)
     * Tier 2: Claim keyword overlap (title: +10 per word, content: +2 per word)
     * Tier 3: LLM-based relevance score (+1 to +10)
   - Sort by combined score descending
   - Return top 21 highest-scored documents as `retrieved_docs`

## Metric
The `discrete_retrieval_eval` metric computes recall@21: whether all gold supporting document titles from `supporting_facts` are present in the retrieved set. Success requires the retrieval pipeline to discover all necessary evidence documents within the 21-document budget. The entity-focused strategy maximizes recall by targeting specific Wikipedia articles for key entities (up to 105 documents with k=35 × 3 queries), then uses a multi-faceted scoring mechanism to prioritize documents with exact entity matches and high keyword overlap, ensuring the most relevant 21 documents are selected.

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

