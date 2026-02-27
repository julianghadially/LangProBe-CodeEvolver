import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class DiversityScorer(dspy.Signature):
    """Score how different this document is from already retrieved documents based on the claim."""

    claim: str = dspy.InputField()
    already_retrieved_titles: str = dspy.InputField(desc="comma-separated list of already retrieved document titles")
    candidate_passage: str = dspy.InputField()
    diversity_score: float = dspy.OutputField(desc="0.0-1.0, where 1.0 means highly diverse/different from already retrieved documents")


class DiversityReranker(dspy.Module):
    """Reranks candidate passages based on diversity from already retrieved documents."""

    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(DiversityScorer)

    def forward(self, claim, candidate_passages, already_retrieved_titles):
        """
        Args:
            claim: The claim to verify
            candidate_passages: List of candidate passage strings (format: "title | content")
            already_retrieved_titles: Set of already retrieved document titles

        Returns:
            List of passages sorted by diversity score (highest first)
        """
        scored_passages = []

        # Convert already_retrieved_titles set to comma-separated string
        titles_str = ", ".join(sorted(already_retrieved_titles)) if already_retrieved_titles else "none"

        for passage in candidate_passages:
            # Extract title from passage (format: "title | content")
            title = passage.split(" | ")[0] if " | " in passage else passage.split("\n")[0]

            # If this title is already retrieved, give it diversity_score=0.0
            if title in already_retrieved_titles:
                scored_passages.append((passage, 0.0))
                continue

            # Score diversity using the DSPy module
            try:
                result = self.scorer(
                    claim=claim,
                    already_retrieved_titles=titles_str,
                    candidate_passage=passage
                )
                # Extract diversity score, ensuring it's a float between 0 and 1
                score = float(result.diversity_score)
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except (ValueError, AttributeError, TypeError):
                # If scoring fails, give a neutral score
                score = 0.5

            scored_passages.append((passage, score))

        # Sort by diversity score (highest first)
        scored_passages.sort(key=lambda x: x[1], reverse=True)

        # Return just the passages (without scores)
        return [passage for passage, score in scored_passages]


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.diversity_reranker = DiversityReranker()
        # Query generation modules for hops 2 and 3
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        # Summarization modules
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Initialize tracking structures
            retrieved_titles = set()
            retrieved_docs = []

            # HOP 1: Retrieve initial documents and select most diverse k=7
            hop1_candidates = self.rm(claim, k=21).passages
            hop1_reranked = self.diversity_reranker(
                claim=claim,
                candidate_passages=hop1_candidates,
                already_retrieved_titles=retrieved_titles
            )
            hop1_docs = hop1_reranked[:7]

            # Add to retrieved set
            for doc in hop1_docs:
                title = doc.split(" | ")[0] if " | " in doc else doc.split("\n")[0]
                retrieved_titles.add(title)
                retrieved_docs.append(doc)

            # Summarize hop 1 documents
            summary_1 = self.summarize1(claim=claim, passages=hop1_docs).summary

            # HOP 2: Generate query, retrieve, and select diverse k=7
            hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
            hop2_candidates = self.rm(hop2_query, k=21).passages
            hop2_reranked = self.diversity_reranker(
                claim=claim,
                candidate_passages=hop2_candidates,
                already_retrieved_titles=retrieved_titles
            )
            hop2_docs = hop2_reranked[:7]

            # Add to retrieved set
            for doc in hop2_docs:
                title = doc.split(" | ")[0] if " | " in doc else doc.split("\n")[0]
                retrieved_titles.add(title)
                retrieved_docs.append(doc)

            # Summarize hop 2 documents
            summary_2 = self.summarize2(
                claim=claim, context=summary_1, passages=hop2_docs
            ).summary

            # HOP 3: Generate query, retrieve, and select diverse k=7
            hop3_query = self.create_query_hop3(
                claim=claim, summary_1=summary_1, summary_2=summary_2
            ).query
            hop3_candidates = self.rm(hop3_query, k=21).passages
            hop3_reranked = self.diversity_reranker(
                claim=claim,
                candidate_passages=hop3_candidates,
                already_retrieved_titles=retrieved_titles
            )
            hop3_docs = hop3_reranked[:7]

            # Add to retrieved set
            for doc in hop3_docs:
                title = doc.split(" | ")[0] if " | " in doc else doc.split("\n")[0]
                retrieved_titles.add(title)
                retrieved_docs.append(doc)

            # Return exactly 21 documents with no duplicates
            return dspy.Prediction(retrieved_docs=retrieved_docs)
