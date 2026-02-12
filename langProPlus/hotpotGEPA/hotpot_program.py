import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer using exact facts extracted from passages.

    Preserve exact names, dates, and locations without adding or removing details.
    """

    question = dspy.InputField()
    facts_1 = dspy.InputField(desc="Exact key facts extracted verbatim from first hop passages")
    facts_2 = dspy.InputField(desc="Exact key facts extracted verbatim from second hop passages")
    answer = dspy.OutputField(desc="The answer itself and nothing else")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant with relevance-filtering instead of summarization."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.extract_relevant_facts = dspy.ChainOfThought('question,passages->relevant_facts')
        self.create_query_hop2 = dspy.Predict("question,facts_1->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.generate_answer = dspy.Predict(GenerateAnswer)

    def forward(self, question):
        # HOP 1: Retrieve and extract relevant facts verbatim
        hop1_docs = self.retrieve_k(question).passages
        facts_1 = self.extract_relevant_facts(
            question=question, passages=hop1_docs
        ).relevant_facts

        # HOP 2: Create query using extracted facts, retrieve, and extract relevant facts
        hop2_query = self.create_query_hop2(question=question, facts_1=facts_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        facts_2 = self.extract_relevant_facts(
            question=question, passages=hop2_docs
        ).relevant_facts

        # HOP 3: Generate answer using exact facts from both hops
        answer = self.generate_answer(
            question=question, facts_1=facts_1, facts_2=facts_2
        ).answer

        return dspy.Prediction(answer=answer)
