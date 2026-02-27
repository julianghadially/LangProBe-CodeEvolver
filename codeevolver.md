PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

This program implements a diversity-aware multi-hop document retrieval system for the HoVer (Hover-nlp) fact verification task using DSPy. The system retrieves supporting documents for a given claim through iterative retrieval with explicit diversity optimization to reduce redundancy and improve coverage of multi-hop reasoning chains.

**Key Modules:**

1. **HoverMultiHopPipeline** (`hover_pipeline.py`): The top-level pipeline that implements diversity-aware iterative retrieval:
   - Initializes ColBERTv2 retriever with a remote API endpoint
   - Contains DiversityScorer signature and DiversityReranker module for scoring document diversity
   - Orchestrates 3-hop retrieval where each hop retrieves k=21 candidates, reranks by diversity, and selects top k=7
   - Maintains a set of retrieved_titles to prevent duplicates across hops
   - Returns exactly 21 diverse documents (3 hops × 7 documents each)

2. **DiversityScorer** (Signature in `hover_pipeline.py`): DSPy signature that scores how different a candidate document is from already retrieved documents (0.0-1.0, where 1.0 means highly diverse). Takes claim, already_retrieved_titles (comma-separated), and candidate_passage as inputs.

3. **DiversityReranker** (Module in `hover_pipeline.py`): DSPy module that reranks candidate passages based on diversity scores. Uses ChainOfThought(DiversityScorer) to score each candidate and returns passages sorted by diversity (highest first). Automatically assigns diversity_score=0.0 to duplicate titles.

4. **HoverMultiHop** (`hover_program.py`): The original multi-hop retrieval program (now unused, replaced by logic in HoverMultiHopPipeline.forward()).

5. **hover_data.py**: Manages dataset loading from the HoVer benchmark, filtering for 3-hop examples and creating DSPy Example objects.

6. **hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` which checks if all gold supporting document titles are present in the retrieved documents (subset match).

**Data Flow:**
Claim → Hop1: Retrieve 21 candidates → DiversityRerank → Select top 7 diverse → Summarize → Hop2: Query Generation → Retrieve 21 candidates → DiversityRerank (excluding already retrieved) → Select top 7 diverse → Summarize → Hop3: Query Generation → Retrieve 21 candidates → DiversityRerank (excluding already retrieved) → Select top 7 diverse → Return 21 unique documents

**Metric:** The `discrete_retrieval_eval` metric returns True if all gold standard supporting document titles are found within the predicted retrieved documents (max 21), using normalized text matching.

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

