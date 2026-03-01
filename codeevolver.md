PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
The HoVer (fact verification) system is a multi-hop document retrieval pipeline designed to identify supporting documents for fact-checking claims. It uses a query decomposition strategy that extracts distinct entities from the claim and performs dedicated retrieval for each entity, followed by intelligent re-ranking to select the most relevant documents.

## Key Modules
- **HoverMultiHopPipeline**: Top-level pipeline implementing query decomposition with entity extraction, focused retrieval, and document re-ranking
- **EntityExtractor**: DSPy Signature that extracts 2-3 distinct entities or key topics from claims
- **QueryGenerator**: DSPy Signature that generates focused search queries for each entity
- **HoverMultiHop**: Legacy core program implementing the 3-hop retrieval strategy (not currently used)
- **hoverBench**: Dataset loader filtering HoVer dataset examples to 3-hop cases (26K+ training examples from hover-nlp/hover)
- **hover_utils**: Contains the evaluation metric and document counting utilities

## Data Flow
1. **Entity Extraction**: Extract 2-3 distinct entities or key topics from the claim using EntityExtractor signature
2. **Query Generation**: Generate one focused query per entity (max 3 queries) using QueryGenerator signature
3. **Retrieval**: Retrieve 22 documents per query (66 total documents across 3 queries)
4. **Re-ranking**: Score and re-rank documents based on:
   - Number of extracted entities mentioned (weighted 3x per entity)
   - Term overlap with claim (Jaccard similarity)
   - Document uniqueness (deduplication by title)
5. **Selection**: Return top 21 unique, highest-scoring documents

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

