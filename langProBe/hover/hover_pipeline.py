import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class DocumentRelevanceSignature(dspy.Signature):
    """Evaluate the relevance of a document to a claim. Score from 1-10 where 10 is highly relevant and provides critical evidence, and 1 is completely irrelevant."""

    claim: str = dspy.InputField(desc="the claim to verify")
    document: str = dspy.InputField(desc="the document to evaluate")
    reasoning: str = dspy.OutputField(desc="explanation of why this document is relevant or not relevant to the claim")
    score: int = dspy.OutputField(desc="relevance score from 1 (irrelevant) to 10 (highly relevant)")


class DocumentRelevanceScorer(dspy.Module):
    """Module that scores document relevance using chain-of-thought reasoning."""

    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(DocumentRelevanceSignature)

    def forward(self, claim, document):
        return self.scorer(claim=claim, document=document)


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop(k=12)  # Increased k to retrieve more documents per hop
        self.scorer = DocumentRelevanceScorer()

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Phase 1: Retrieve documents (k=12 per hop, ~36 total)
            result = self.program(claim=claim)
            all_docs = result.retrieved_docs

            # Deduplicate documents based on title (before " | ")
            unique_docs = []
            seen_titles = set()
            for doc in all_docs:
                title = doc.split(" | ")[0]
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_docs.append(doc)

            # Phase 2: Score each unique document
            scored_docs = []
            for doc in unique_docs:
                try:
                    score_result = self.scorer(claim=claim, document=doc)
                    # Parse score as integer, default to 5 if parsing fails
                    try:
                        score = int(score_result.score)
                    except (ValueError, TypeError):
                        score = 5
                    scored_docs.append((doc, score))
                except Exception:
                    # If scoring fails, assign neutral score
                    scored_docs.append((doc, 5))

            # Sort by score descending and take top 21
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_docs)
