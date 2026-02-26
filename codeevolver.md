PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements Query Decomposition with Early Gap Analysis and Iterative Entity Discovery for multi-hop document retrieval on the HoVer claim verification benchmark using DSPy. The system transforms retrieval from similarity search to structured reasoning by decomposing claims into sub-questions, performing early gap analysis before query generation, executing dual-phase targeted retrieval (k=18 each), discovering entities/relationships iteratively, identifying bridging entities for dedicated retrieval, and using entity-based scoring for reranking. It retrieves ~36 documents in initial dual-phase plus additional documents across later iterations, then reranks to return exactly 21 most relevant documents using entity matching from both claim and gap analysis.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Main pipeline implementing early gap analysis with dual-phase retrieval and iterative entity discovery. Decomposes claims into sub-questions, performs initial retrieval (k=18), invokes gap analyzer to identify missing information, executes second targeted retrieval (k=18) based on gap analysis output. After initial dual-phase retrieval (~36 docs), identifies bridging entities (people, organizations, events) in retrieved documents that need dedicated retrieval, retrieves k=5 docs per bridging entity. Performs additional gap analysis iterations to identify missing info and generate targeted queries. Deduplicates all documents, reranks using entity-based scoring mechanism that prioritizes documents matching entities from both claim and gap analysis output, returns top 21. Entry point for evaluation.

**ClaimDecomposition** (`hover_pipeline.py`): Signature decomposing claims into 2-3 answerable sub-questions.

**GapAnalyzer** (`hover_pipeline.py`): Early gap analysis signature taking only a claim as input and identifying what facts need verification. Used before query generation in dual-phase retrieval.

**EntityExtractor** (`hover_pipeline.py`): Signature extracting entities/relationships from documents for structured knowledge.

**GapAnalysis** (`hover_pipeline.py`): Advanced gap analysis signature analyzing missing information given entities/relationships/documents found so far, generating targeted queries for later iterations.

**BridgingEntityIdentifier** (`hover_pipeline.py`): Signature identifying 3-5 specific bridging entities (people, organizations, events) in retrieved documents that appear as important intermediate connections but need standalone retrieval.

**DocumentRelevanceScorer** (`hover_pipeline.py`): ChainOfThought module scoring document relevance (1-10). Note: Currently not used in main pipeline; replaced by entity-based scoring mechanism.

**hover_utils**: Contains `discrete_retrieval_eval` metric for recall@21 evaluation.

**hover_data**: Loads HoVer dataset with 3-hop examples.

## Data Flow
1. Input claim enters `HoverMultiHopPipeline.forward()`
2. **Dual-Phase Initial Retrieval**:
   - Decompose claim → 2-3 sub-questions → use first sub-question for initial retrieval (k=18)
   - Invoke GapAnalyzer on claim → identify missing information → generate targeted query → second retrieval (k=18)
   - Combine both retrievals (~36 docs total)
3. **Entity Extraction**: Extract entities/relationships from initial documents
4. **Bridging Entity Discovery**: Identify 3-5 bridging entities from iteration 1 docs → retrieve k=5 docs per entity using entity name as direct query (e.g., 'Lisa Raymond', 'Ellis Ferreira') → add to all_retrieved_docs
5. **Iteration 2**: Advanced GapAnalysis identifies missing info based on entities/relationships found → generate 3 targeted queries → retrieve k=5 docs per query → update entities/relationships
6. **Iteration 3**: Final GapAnalysis → generate 3 queries for remaining gaps → retrieve k=5 docs per query (total ~50-60 docs)
7. **Post-Iteration**: Deduplicate by title → entity-based scoring (extract entities from claim + gap analysis output, score documents by entity matches with bonus for matching both) → sort by score → return top 21 documents

## Metric
The `discrete_retrieval_eval` metric computes recall@21: whether all gold supporting document titles are in the retrieved set. The dual-phase retrieval architecture with early gap analysis and entity-based reranking maximizes recall through claim decomposition, early gap analysis to identify missing information before query generation, dual-phase targeted retrieval (k=18 each), structured entity/relationship tracking, bridging entity identification for dedicated retrieval of implicit entities discovered through initial docs, iterative gap analysis for missing information in later phases, and entity-based scoring mechanism that prioritizes documents matching entities from both claim and gap analysis output to select the most relevant 21 documents.

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

