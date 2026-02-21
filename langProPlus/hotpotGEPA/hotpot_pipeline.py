import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ExtractShortAnswer(dspy.Signature):
    """Extract the shortest exact factoid answer from passages."""

    question = dspy.InputField()
    passages = dspy.InputField()
    answer = dspy.OutputField(desc="Extract the shortest exact factoid answer from the passages - just the answer itself, no extra words or explanation")


class HotpotMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """2-hop retrieve-then-extract architecture for multi-hop question answering."""

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.retrieve_k = dspy.Retrieve(k=10)
        self.generate_hop2_query = dspy.ChainOfThought("question,passages->query")
        self.extract_answer = dspy.Predict(ExtractShortAnswer)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            # Hop 1: Retrieve k=10 passages
            hop1_docs = self.retrieve_k(question).passages

            # Hop 2: Generate focused query from hop 1 passages
            hop2_query = self.generate_hop2_query(
                question=question,
                passages=hop1_docs
            ).query

            # Hop 2: Retrieve k=10 passages
            hop2_docs = self.retrieve_k(hop2_query).passages

            # Concatenate all passages from both hops
            all_passages = hop1_docs + hop2_docs

            # Extract the minimal answer directly from passages
            answer = self.extract_answer(
                question=question,
                passages=all_passages
            ).answer

            return dspy.Prediction(answer=answer)

