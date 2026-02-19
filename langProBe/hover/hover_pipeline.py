import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


# DSPy Signature classes for query perspective ensemble
class DirectEntityExtractor(dspy.Signature):
    """Extract key entities, terms, and concepts from a claim for direct factual retrieval."""
    claim: str = dspy.InputField(desc="The claim to extract entities from")
    query: str = dspy.OutputField(desc="A search query containing key entities and terms from the claim")


class RelationalQueryGenerator(dspy.Signature):
    """Identify relationships, connections, and contextual information needed to verify a claim."""
    claim: str = dspy.InputField(desc="The claim to analyze for relationships")
    query: str = dspy.OutputField(desc="A search query focused on relationships and connections in the claim")


class ContradictionQueryGenerator(dspy.Signature):
    """Generate a query that would find evidence contradicting or challenging the claim for comprehensive coverage."""
    claim: str = dspy.InputField(desc="The claim to challenge")
    query: str = dspy.OutputField(desc="A search query to find contradictory or challenging evidence")


class RelevanceScorer(dspy.Signature):
    """Score the relevance of a document passage to a claim on a scale from 0 to 10."""
    claim: str = dspy.InputField(desc="The original claim to verify")
    passage: str = dspy.InputField(desc="The document passage to score")
    score: float = dspy.OutputField(desc="Relevance score from 0 (not relevant) to 10 (highly relevant)")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using Query Perspective Ensemble.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize the three query perspective modules
        self.entity_extractor = dspy.Predict(DirectEntityExtractor)
        self.relational_generator = dspy.ChainOfThought(RelationalQueryGenerator)
        self.contradiction_generator = dspy.ChainOfThought(ContradictionQueryGenerator)

        # Initialize retrievers with adaptive k values
        self.retrieve_k10 = dspy.Retrieve(k=10)
        self.retrieve_k8 = dspy.Retrieve(k=8)
        self.retrieve_k5 = dspy.Retrieve(k=5)

        # Initialize relevance scorer
        self.relevance_scorer = dspy.Predict(RelevanceScorer)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Generate queries from three different perspectives in parallel
            entity_result = self.entity_extractor(claim=claim)
            relational_result = self.relational_generator(claim=claim)
            contradiction_result = self.contradiction_generator(claim=claim)

            entity_query = entity_result.query
            relational_query = relational_result.query
            contradiction_query = contradiction_result.query

            # Retrieve documents in parallel with adaptive k-values
            entity_docs = self.retrieve_k10(entity_query).passages  # k=10
            relational_docs = self.retrieve_k8(relational_query).passages  # k=8
            contradiction_docs = self.retrieve_k5(contradiction_query).passages  # k=5

            # Combine all 23 documents (10 + 8 + 5)
            all_docs = entity_docs + relational_docs + contradiction_docs

            # Score each document for relevance to the original claim
            scored_docs = []
            for doc in all_docs:
                try:
                    score_result = self.relevance_scorer(claim=claim, passage=doc)
                    score = float(score_result.score)
                except (ValueError, AttributeError):
                    # If scoring fails, assign a default low score
                    score = 0.0
                scored_docs.append((doc, score))

            # Sort by relevance score (descending) and select top 21
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_21_docs = [doc for doc, score in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_21_docs)
