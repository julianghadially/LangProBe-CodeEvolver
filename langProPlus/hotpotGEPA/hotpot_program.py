import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractAnswer(dspy.Signature):
    """Extract the exact answer from retrieved passages."""

    question = dspy.InputField()
    passages = dspy.InputField()
    answer = dspy.OutputField(desc="extract the exact short factoid answer from the passages - be precise and concise, no elaboration")


class VerifyAndRefineAnswer(dspy.Signature):
    """Verify and refine the initial answer to match the exact specificity required by the question.

    Consider:
    - Does the question ask for a full name including middle name? (e.g., "What is X's full name?")
    - Does the question require specific qualifiers? (e.g., "flowering plants" vs just "plants")
    - Is the answer too general or too specific for what's being asked?
    - Does the answer match the exact format implied by the question?
    """

    question = dspy.InputField()
    all_passages = dspy.InputField(desc="all retrieved passages that may contain the complete answer")
    initial_answer = dspy.InputField(desc="the initial answer that may need refinement")
    refined_answer = dspy.OutputField(desc="the final answer with exact specificity matching the question - include full names with middle names when asked, use appropriate qualifiers, and match the precision level required")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Direct passage-to-answer extraction with two-hop retrieval and answer refinement."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("question,hop1_passages->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.extract_answer = dspy.ChainOfThought(ExtractAnswer)
        self.verify_and_refine = dspy.ChainOfThought(VerifyAndRefineAnswer)

    def forward(self, question):
        # HOP 1: Initial retrieval using the question
        hop1_docs = self.retrieve_k(question).passages

        # HOP 2: Refined retrieval using hop1 context
        hop2_query = self.create_query_hop2(question=question, hop1_passages=hop1_docs).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # Concatenate all retrieved passages from both hops
        all_passages = hop1_docs + hop2_docs

        # Direct extraction: passages -> answer (no intermediate summarization)
        initial_answer = self.extract_answer(
            question=question, passages=all_passages
        ).answer

        # Self-verification and answer refinement
        # This step ensures the answer matches the exact specificity required by the question
        # (e.g., full names with middle names, appropriate qualifiers, correct precision level)
        refined_answer = self.verify_and_refine(
            question=question,
            all_passages=all_passages,
            initial_answer=initial_answer
        ).refined_answer

        return dspy.Prediction(answer=refined_answer)
