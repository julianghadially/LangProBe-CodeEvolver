import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class BridgeEntityQueryGeneration(dspy.Signature):
    """Generate three targeted queries to retrieve documents for a multi-hop claim.

    The goal is to identify and retrieve documents about:
    1. Primary entities (main subjects/people mentioned in the claim)
    2. Bridge entities (connecting elements like films, locations, works that link people/entities)
    3. Secondary/attribute entities (properties, characteristics, or related entities)

    Use chain-of-thought reasoning to explicitly identify what the bridge entities are.
    Bridge entities are critical connecting documents that link primary and secondary entities in multi-hop claims."""

    claim: str = dspy.InputField(desc="The claim to verify")
    primary_query: str = dspy.OutputField(desc="Query targeting primary entities/subjects in the claim")
    bridge_query: str = dspy.OutputField(desc="Query targeting bridge/connecting entities (e.g., films, locations, works) that link the primary entities")
    secondary_query: str = dspy.OutputField(desc="Query targeting secondary/attribute entities or properties mentioned in the claim")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()
        self.query_generator = dspy.ChainOfThought(BridgeEntityQueryGeneration)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Generate three targeted queries with explicit bridge entity identification
            query_result = self.query_generator(claim=claim)

            # Retrieve k=30 documents per query (90 total)
            retrieve_30 = dspy.Retrieve(k=30)
            primary_docs = retrieve_30(query_result.primary_query).passages
            bridge_docs = retrieve_30(query_result.bridge_query).passages
            secondary_docs = retrieve_30(query_result.secondary_query).passages

            # Deduplicate documents and track their frequency
            doc_frequency = {}
            all_docs = primary_docs + bridge_docs + secondary_docs

            for doc in all_docs:
                if doc not in doc_frequency:
                    doc_frequency[doc] = 0
                doc_frequency[doc] += 1

            # Sort by frequency (descending), then maintain original order for ties
            # This gives higher rank to documents appearing in multiple query results
            unique_docs = []
            seen = set()
            for doc in all_docs:
                if doc not in seen:
                    unique_docs.append(doc)
                    seen.add(doc)

            # Sort by frequency (descending)
            ranked_docs = sorted(unique_docs, key=lambda doc: doc_frequency[doc], reverse=True)

            # Return top 21 documents
            final_docs = ranked_docs[:21]

            return dspy.Prediction(retrieved_docs=final_docs)
