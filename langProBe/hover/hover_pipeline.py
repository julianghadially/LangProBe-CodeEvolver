import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ExtractKeyEntities(dspy.Signature):
    """Extract specific named entities, dates, locations, organizations, and person names from documents.
    Focus on preserving exact names, dates, and specific identifiers that are critical for finding supporting documents."""

    claim: str = dspy.InputField()
    passages: list[str] = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="list of specific named entities, dates, locations, organizations, and person names from the documents")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.k = 7

        # Abstract Track: summarization modules
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

        # Entity Track: entity extraction module
        self.extract_entities = dspy.Predict(ExtractKeyEntities)

        # Query generation modules with dual-track inputs
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1,entities_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2,entities_1,entities_2->query")

        # Retrieval module
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # HOP 1: Initial retrieval
            hop1_docs = self.retrieve_k(claim).passages

            # Abstract Track: Summarize hop 1 documents
            summary_1 = self.summarize1(
                claim=claim, passages=hop1_docs
            ).summary

            # Entity Track: Extract entities from hop 1 documents
            entities_1_result = self.extract_entities(
                claim=claim, passages=hop1_docs
            )
            entities_1 = entities_1_result.entities

            # HOP 2: Dual-track query generation
            hop2_query = self.create_query_hop2(
                claim=claim,
                summary_1=summary_1,
                entities_1=entities_1
            ).query
            hop2_docs = self.retrieve_k(hop2_query).passages

            # Abstract Track: Summarize hop 2 documents
            summary_2 = self.summarize2(
                claim=claim, context=summary_1, passages=hop2_docs
            ).summary

            # Entity Track: Extract and accumulate entities from hop 2 documents
            entities_2_result = self.extract_entities(
                claim=claim, passages=hop2_docs
            )
            entities_2 = entities_2_result.entities

            # HOP 3: Dual-track query generation with all accumulated context
            hop3_query = self.create_query_hop3(
                claim=claim,
                summary_1=summary_1,
                summary_2=summary_2,
                entities_1=entities_1,
                entities_2=entities_2
            ).query
            hop3_docs = self.retrieve_k(hop3_query).passages

            return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
