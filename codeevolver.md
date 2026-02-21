PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is an iterative query decomposition multi-hop document retrieval system for fact-checking claims using the HoVer (Hover-nlp) dataset. The system uses bridge entity recognition with sequential hop-based retrieval and cross-hop connectivity ranking inspired by the DeCoR framework to find the most relevant supporting documents for veracity assessment of claims.

**Key Modules**:
- **HoverMultiHopPipeline** (hover_pipeline.py): Top-level pipeline implementing the complete four-stage iterative query decomposition with bridge entity recognition architecture
- **ClaimDecomposer** (hover_pipeline.py): DSPy Signature for analyzing claims and decomposing them into 2-3 sequential sub-questions with explicit bridge entity identification for multi-hop reasoning
- **BridgeEntityExtractor** (hover_pipeline.py): DSPy Signature for extracting specific entities that serve as bridges to connect reasoning hops from retrieved documents
- **MultiHopRelevanceScorer** (hover_pipeline.py): DSPy Signature for scoring documents using intermediate layer representations based on sub-question relevance, bridge entity presence, and cross-hop connectivity
- **CrossHopConnectivityRanker** (hover_pipeline.py): DSPy Signature for final fusion ranking that scores documents based on how well they form a complete reasoning chain across all hops
- **EntityExtraction** (hover_pipeline.py): Legacy DSPy Signature for extracting key entities (kept for compatibility)
- **EntityQueryGenerator** (hover_pipeline.py): Legacy DSPy Signature for generating focused search queries for specific entities (kept for compatibility)
- **ListwiseReranker** (hover_pipeline.py): Legacy DSPy Signature for scoring and ranking documents (kept for compatibility)
- **hoverBench** (hover_data.py): Dataset handler that loads and filters HoVer dataset to 3-hop examples, creating train/test splits
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting document titles are retrieved (maximum 21 documents)

**Data Flow**:
1. **Stage 1 - Claim Decomposition**: Use ClaimDecomposer with ChainOfThought to analyze the claim and break it into 2-3 sequential sub-questions representing logical hops, explicitly identifying expected bridge entity types (e.g., actor name, location, organization) for each hop
2. **Stage 2 - Sequential Bridge Retrieval**: For each sub-question (2-3 total retrieval searches adhering to constraint):
   - Retrieve k=50 documents using the sub-question as query (enhanced with bridge entity from previous hop if available)
   - Store all documents organized by hop index
   - Extract bridge entity from top 20 retrieved documents using BridgeEntityExtractor (except for final hop)
   - Use extracted bridge entity to formulate the next sub-question's query, creating a sequential reasoning chain
3. **Stage 3 - Intermediate Representation Reranking**: For each hop's documents (2-3 hops):
   - Apply MultiHopRelevanceScorer using ChainOfThought to score documents based on: (a) direct relevance to sub-question, (b) presence of bridge entities identified so far, and (c) how documents connect across reasoning hops
   - Select top 7-10 documents per hop based on multi-hop relevance scores
   - Aggregate selected documents from all hops (14-30 documents total)
4. **Stage 4 - Final Fusion**: Combine all selected documents from all hops (total retrieval budget: 2-3 searches as per constraint):
   - Deduplicate documents by normalized title while preserving content
   - Apply CrossHopConnectivityRanker using ChainOfThought that scores all unique documents based on how well they form a complete reasoning chain connecting all sub-questions and bridge entities
   - Select final 21 documents that best support the complete multi-hop verification chain
5. Output: Final 21 highest-ranked documents as retrieved_docs prediction

**Metric**: discrete_retrieval_eval compares normalized gold document titles against retrieved document titles, returning True if all gold titles are found within the retrieved set (subset check).

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

