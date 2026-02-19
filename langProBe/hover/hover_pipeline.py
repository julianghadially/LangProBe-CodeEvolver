import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


# DSPy Signatures for Pseudo-Relevance Feedback
class DocumentRelevanceScorer(dspy.Signature):
    """Score the relevance of a document to a claim on a scale of 1-5."""
    claim: str = dspy.InputField()
    document: str = dspy.InputField()
    relevance_score: int = dspy.OutputField(desc="relevance score from 1 (not relevant) to 5 (highly relevant)")
    reasoning: str = dspy.OutputField(desc="brief explanation of the relevance score")


class KeyTermExtractor(dspy.Signature):
    """Extract key entities and phrases from relevant documents that are important for the claim."""
    claim: str = dspy.InputField()
    relevant_documents: str = dspy.InputField(desc="top relevant documents from previous retrieval")
    key_entities: str = dspy.OutputField(desc="comma-separated list of key entities")
    key_phrases: str = dspy.OutputField(desc="comma-separated list of key phrases")


class ExpandedQueryGenerator(dspy.Signature):
    """Generate an expanded search query incorporating key entities and phrases."""
    claim: str = dspy.InputField()
    key_entities: str = dspy.InputField(desc="comma-separated key entities")
    key_phrases: str = dspy.InputField(desc="comma-separated key phrases")
    expanded_query: str = dspy.OutputField(desc="expanded search query")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

        # Initialize modules for pseudo-relevance feedback
        self.relevance_scorer = dspy.ChainOfThought(DocumentRelevanceScorer)
        self.term_extractor = dspy.Predict(KeyTermExtractor)
        self.query_expander = dspy.Predict(ExpandedQueryGenerator)
        self.retrieve_k15 = dspy.Retrieve(k=15)

    def _score_and_select_top_docs(self, claim, docs, top_k=5):
        """Score documents and return top k with their scores."""
        scored_docs = []
        for doc in docs:
            try:
                result = self.relevance_scorer(claim=claim, document=doc)
                score = int(result.relevance_score)
            except:
                score = 3  # default middle score if parsing fails
            scored_docs.append((doc, score))

        # Sort by score descending and take top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            all_scored_docs = []  # Track all docs with scores
            seen_docs = set()  # Track unique documents

            # HOP 1: Initial retrieval from claim
            hop1_docs = self.retrieve_k15(claim).passages
            hop1_scored = self._score_and_select_top_docs(claim, hop1_docs, top_k=5)

            # Store and deduplicate
            for doc, score in hop1_scored:
                if doc not in seen_docs:
                    all_scored_docs.append((doc, score))
                    seen_docs.add(doc)

            # Extract terms from top 5 docs
            top5_docs_text = "\n\n".join([doc for doc, _ in hop1_scored])
            terms = self.term_extractor(claim=claim, relevant_documents=top5_docs_text)

            # Generate expanded query for Hop 2
            hop2_query = self.query_expander(
                claim=claim,
                key_entities=terms.key_entities,
                key_phrases=terms.key_phrases
            ).expanded_query

            # HOP 2: Retrieval with expanded query
            hop2_docs = self.retrieve_k15(hop2_query).passages
            hop2_scored = self._score_and_select_top_docs(claim, hop2_docs, top_k=5)

            # Store and deduplicate
            for doc, score in hop2_scored:
                if doc not in seen_docs:
                    all_scored_docs.append((doc, score))
                    seen_docs.add(doc)

            # Extract terms from Hop 2 top 5 docs
            hop2_top5_text = "\n\n".join([doc for doc, _ in hop2_scored])
            terms2 = self.term_extractor(claim=claim, relevant_documents=hop2_top5_text)

            # Generate expanded query for Hop 3
            hop3_query = self.query_expander(
                claim=claim,
                key_entities=terms2.key_entities,
                key_phrases=terms2.key_phrases
            ).expanded_query

            # HOP 3: Final retrieval with refined query
            hop3_docs = self.retrieve_k15(hop3_query).passages
            hop3_scored = self._score_and_select_top_docs(claim, hop3_docs, top_k=5)

            # Store and deduplicate
            for doc, score in hop3_scored:
                if doc not in seen_docs:
                    all_scored_docs.append((doc, score))
                    seen_docs.add(doc)

            # Now we need to pad to exactly 21 documents
            # We have at most 15 unique docs from top 5 of each hop
            # Need to add more from the retrieved docs to reach 21

            # Collect all remaining docs with scores
            remaining_docs = []
            for hop_docs in [hop1_docs, hop2_docs, hop3_docs]:
                for doc in hop_docs:
                    if doc not in seen_docs:
                        # Score remaining docs
                        try:
                            result = self.relevance_scorer(claim=claim, document=doc)
                            score = int(result.relevance_score)
                        except:
                            score = 2
                        remaining_docs.append((doc, score))
                        seen_docs.add(doc)

            # Sort remaining by score
            remaining_docs.sort(key=lambda x: x[1], reverse=True)

            # Combine: top docs from hops + highest scored remaining
            all_scored_docs.extend(remaining_docs)

            # Take exactly 21 documents
            final_docs = [doc for doc, _ in all_scored_docs[:21]]

            # Pad with empty strings if needed (shouldn't happen with k=15 per hop)
            while len(final_docs) < 21:
                final_docs.append("")

            return dspy.Prediction(retrieved_docs=final_docs[:21])
