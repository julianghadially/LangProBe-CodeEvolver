import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class KeyEntityExtractor(dspy.Signature):
    """Extract 2-3 critical named entities or topics from a claim for targeted document retrieval.
    Focus on people, places, organizations, works, or key concepts that require dedicated coverage."""

    claim: str = dspy.InputField(desc="The claim to analyze")
    entities: list[str] = dspy.OutputField(
        desc="A list of 2-3 critical named entities or key topics from the claim (people, places, works, organizations)"
    )


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()
        self.entity_extractor = dspy.Predict(KeyEntityExtractor)
        self.entity_retriever = dspy.Retrieve(k=15)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Extract 2-3 key entities from the claim
            extraction_result = self.entity_extractor(claim=claim)
            entities = extraction_result.entities

            # Limit to max 3 entities to respect query limit
            entities = entities[:3]

            # Perform dedicated retrieval for each entity
            all_docs = []
            doc_frequency = {}  # Track how many times each document appears

            for entity in entities:
                # Retrieve k=15 documents for this entity
                docs = self.entity_retriever(entity).passages

                for doc in docs:
                    # Extract title (before " | ")
                    title = doc.split(" | ")[0] if " | " in doc else doc

                    # Track frequency
                    if title not in doc_frequency:
                        doc_frequency[title] = {"count": 0, "doc": doc}
                    doc_frequency[title]["count"] += 1

            # Convert to list and sort by frequency (descending)
            sorted_docs = sorted(
                doc_frequency.values(),
                key=lambda x: x["count"],
                reverse=True
            )

            # Extract the actual documents and truncate to exactly 21
            retrieved_docs = [item["doc"] for item in sorted_docs][:21]

            return dspy.Prediction(retrieved_docs=retrieved_docs)
