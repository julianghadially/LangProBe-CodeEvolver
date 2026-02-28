import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class DiverseQueryGeneration(dspy.Signature):
    """Generate 3 diverse search queries that each focus on different entities or aspects mentioned in this claim."""

    claim: str = dspy.InputField()
    query1: str = dspy.OutputField(desc="First search query targeting a specific entity or aspect")
    query2: str = dspy.OutputField(desc="Second search query targeting a different entity or aspect")
    query3: str = dspy.OutputField(desc="Third search query targeting yet another entity or aspect")


class CoverageScoring(dspy.Signature):
    """Score 0-10 based on unique entity/fact coverage, prioritizing documents that cover different entities than already selected docs."""

    claim: str = dspy.InputField()
    document: str = dspy.InputField()
    covered_entities: str = dspy.InputField(desc="Entities and facts already covered by previously selected documents")
    score: int = dspy.OutputField(desc="Score 0-10 based on how well this document covers entities/facts not yet covered")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

        # Initialize modules for parallel multi-query retrieval with coverage-based reranking
        self.query_generator = dspy.Predict(DiverseQueryGeneration)
        self.retrieve_k35 = dspy.Retrieve(k=35)
        self.coverage_scorer = dspy.Predict(CoverageScoring)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Generate 3 diverse queries
            query_result = self.query_generator(claim=claim)
            queries = [query_result.query1, query_result.query2, query_result.query3]

            # Step 2: Retrieve k=35 documents per query (total 105 documents)
            all_docs = []
            for query in queries:
                retrieved = self.retrieve_k35(query)
                all_docs.extend(retrieved.passages)

            # Remove duplicates while preserving order
            seen = set()
            unique_docs = []
            for doc in all_docs:
                if doc not in seen:
                    seen.add(doc)
                    unique_docs.append(doc)

            # Step 3: Iterative reranking with coverage-based scoring
            selected_docs = []
            remaining_docs = unique_docs.copy()
            covered_entities = ""

            # Select 21 documents iteratively based on entity coverage
            for iteration in range(21):
                if not remaining_docs:
                    break

                # Score all remaining documents
                best_doc = None
                best_score = -1

                for doc in remaining_docs:
                    try:
                        score_result = self.coverage_scorer(
                            claim=claim,
                            document=doc,
                            covered_entities=covered_entities
                        )
                        score = int(score_result.score)
                    except (ValueError, AttributeError):
                        # If scoring fails, assign a default score
                        score = 0

                    if score > best_score:
                        best_score = score
                        best_doc = doc

                # Add the best scoring document
                if best_doc:
                    selected_docs.append(best_doc)
                    remaining_docs.remove(best_doc)

                    # Update covered entities summary
                    # Extract key entities from the selected document (simplified approach)
                    doc_summary = best_doc[:200] if len(best_doc) > 200 else best_doc
                    covered_entities += f" {doc_summary}"

            # Return exactly 21 documents
            return dspy.Prediction(retrieved_docs=selected_docs[:21])
