import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ClaimQueryGenerator(dspy.Signature):
    """Generate two diverse search queries from a claim for document retrieval:
    one focused on key entities/nouns, another focused on relations/actions/verbs."""

    claim: str = dspy.InputField(desc="The claim to generate queries for")
    entity_query: str = dspy.OutputField(desc="A query focused on key entities and nouns mentioned in the claim")
    relation_query: str = dspy.OutputField(desc="A query focused on relations, actions, and verbs in the claim")


class DocumentRelevanceScorer(dspy.Signature):
    """Assess how relevant a document is to verifying a given claim.
    Consider whether the document contains information that would help verify or refute the claim."""

    claim: str = dspy.InputField(desc="The claim to be verified")
    document: str = dspy.InputField(desc="The document to assess for relevance")
    score: int = dspy.OutputField(desc="Relevance score from 0 (irrelevant) to 10 (highly relevant)")
    justification: str = dspy.OutputField(desc="Brief explanation of why this score was assigned")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.query_generator = dspy.ChainOfThought(ClaimQueryGenerator)
        self.retrieve_k = dspy.Retrieve(k=50)
        self.relevance_scorer = dspy.ChainOfThought(DocumentRelevanceScorer)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Generate two diverse queries
            query_result = self.query_generator(claim=claim)
            entity_query = query_result.entity_query
            relation_query = query_result.relation_query

            # Step 2: Retrieve k=50 documents per query (100 total)
            entity_docs = self.retrieve_k(entity_query).passages
            relation_docs = self.retrieve_k(relation_query).passages

            # Step 3: Deduplicate by title
            all_docs = entity_docs + relation_docs
            seen_titles = set()
            unique_docs = []
            for doc in all_docs:
                # Extract title (before " | " separator)
                title = doc.split(" | ")[0]
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_docs.append(doc)

            # Step 4 & 5: Score each document individually using LLM
            scored_docs = []
            for doc in unique_docs:
                try:
                    score_result = self.relevance_scorer(claim=claim, document=doc)
                    score = int(score_result.score)
                except (ValueError, AttributeError):
                    # If scoring fails, assign a default low score
                    score = 0
                scored_docs.append((score, doc))

            # Step 6: Sort by score descending and return top 21
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_docs = [doc for score, doc in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_docs)
