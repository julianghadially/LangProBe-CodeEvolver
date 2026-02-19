import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class QuestionDecomposer(dspy.Signature):
    """Decompose a claim into atomic sub-questions that must be answered to verify it.
    Generate up to 3 specific, targeted questions about entities, events, relationships, or facts mentioned in the claim.
    Each sub-question should focus on a single verifiable aspect (e.g., 'Who is X?', 'What is Y?', 'Where did Z happen?', 'When did event occur?')."""

    claim: str = dspy.InputField()
    sub_questions: list[str] = dspy.OutputField(desc="up to 3 atomic sub-questions that need to be answered to verify the claim")


class DocumentScorer(dspy.Signature):
    """Evaluate how relevant a document is for answering ALL the given sub-questions.
    Consider whether the document contains information that helps answer multiple sub-questions.
    Output a relevance score between 0.0 and 1.0, where 1.0 means highly relevant to all sub-questions."""

    sub_questions: list[str] = dspy.InputField(desc="the sub-questions that need to be answered")
    document: str = dspy.InputField(desc="the document to evaluate")
    relevance_score: float = dspy.OutputField(desc="relevance score between 0.0 and 1.0 indicating how well the document answers all sub-questions")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using question decomposition and coverage-based selection.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize question decomposition module
        self.decomposer = dspy.ChainOfThought(QuestionDecomposer)

        # Initialize document scorer module
        self.scorer = dspy.ChainOfThought(DocumentScorer)

        # Initialize retriever with k=30 for each sub-question
        self.retrieve_k = dspy.Retrieve(k=30)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Decompose claim into sub-questions (up to 3)
            decomposition_result = self.decomposer(claim=claim)
            sub_questions = decomposition_result.sub_questions

            # Ensure we have at most 3 sub-questions
            if len(sub_questions) > 3:
                sub_questions = sub_questions[:3]

            # Handle edge case: if no sub-questions generated, use claim directly
            if not sub_questions:
                sub_questions = [claim]

            # Step 2: Retrieve k=30 documents per sub-question
            all_docs = []
            for sub_question in sub_questions:
                retrieved = self.retrieve_k(sub_question).passages
                all_docs.extend(retrieved)

            # Step 3: Deduplicate documents
            unique_docs = deduplicate(all_docs)

            # Step 4: Score each unique document for coverage of all sub-questions
            doc_scores = []
            for doc in unique_docs:
                try:
                    score_result = self.scorer(
                        sub_questions=sub_questions,
                        document=doc
                    )
                    # Extract relevance score, handle both float and string formats
                    relevance = score_result.relevance_score
                    if isinstance(relevance, str):
                        # Try to extract numeric value from string
                        import re
                        match = re.search(r'(\d+\.?\d*)', relevance)
                        if match:
                            relevance = float(match.group(1))
                        else:
                            relevance = 0.5  # default fallback
                    else:
                        relevance = float(relevance)

                    # Ensure score is within valid range
                    relevance = max(0.0, min(1.0, relevance))
                    doc_scores.append((doc, relevance))
                except Exception:
                    # If scoring fails, assign neutral score
                    doc_scores.append((doc, 0.5))

            # Step 5: Sort by relevance score (descending) and select top 21
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in doc_scores[:21]]

            return dspy.Prediction(retrieved_docs=top_docs)
