PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims using the HoVer (Hop Verification) dataset. The system uses claim decomposition strategy to extract specific entities and relationships, then performs targeted retrieval with score-based reranking to find the most relevant supporting documents.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level wrapper implementing claim decomposition strategy with targeted retrieval and reranking
- `ClaimDecomposition`: DSPy signature that extracts 2-3 highly specific search phrases from a claim, focusing on proper nouns, entities, unique descriptive phrases, and relationship keywords
- `DocumentReranker`: Score-based reranking module that computes relevance scores using exact phrase matching and entity overlap with the original claim
- `HoverMultiHop`: Legacy core program implementing 3-hop retrieval logic (not used in new pipeline)
- `hover_data.py`: Data loader that filters HoVer dataset to 3-hop examples (claim-fact pairs requiring 3 documents)
- `hover_utils.py`: Evaluation utilities including the `discrete_retrieval_eval` metric

**Data Flow**:
1. Input claim is decomposed into 2-3 specific search phrases targeting: (1) proper nouns/named entities, (2) unique descriptive phrases, (3) relationship keywords
2. For each search phrase, the system over-retrieves k=25 documents using ColBERTv2
3. DocumentReranker computes relevance scores for each document based on entity overlap (30%), proper noun overlap (40%), and exact phrase matching (30%)
4. Top 7 documents are selected per query after reranking
5. All 21 documents (3 queries × 7 docs each) are returned as `retrieved_docs`

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

