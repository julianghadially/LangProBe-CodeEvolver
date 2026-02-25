import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from collections import Counter

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class GenerateDiverseQueries(dspy.Signature):
    """Generate exactly 3 diverse, non-overlapping search queries from different perspectives to comprehensively verify a claim.
    Each query should focus on a different aspect: entities, relationships, and context."""

    claim: str = dspy.InputField(desc="The claim to verify")
    entity_query: str = dspy.OutputField(desc="A query focused on extracting key named entities (people, places, organizations, dates) mentioned in the claim")
    relationship_query: str = dspy.OutputField(desc="A query focused on connections, comparisons, or relationships between entities mentioned in the claim")
    contextual_query: str = dspy.OutputField(desc="A query focused on the broader domain, category, or context relevant to the claim")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.query_generator = dspy.Predict(GenerateDiverseQueries)
        self.retrieve_k11 = dspy.Retrieve(k=11)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Generate 3 diverse queries in parallel from different perspectives
            queries_result = self.query_generator(claim=claim)
            entity_query = queries_result.entity_query
            relationship_query = queries_result.relationship_query
            contextual_query = queries_result.contextual_query

            # Execute all 3 retrieval calls in parallel (k=11 each, ~33 total docs)
            entity_docs = self.retrieve_k11(entity_query).passages
            relationship_docs = self.retrieve_k11(relationship_query).passages
            contextual_docs = self.retrieve_k11(contextual_query).passages

            # Combine all documents
            all_docs = entity_docs + relationship_docs + contextual_docs

            # Deduplicate and score by frequency (documents appearing in multiple queries rank higher)
            doc_frequency = Counter(all_docs)

            # Sort by frequency (descending), then by order of appearance for tie-breaking
            seen = set()
            ranked_docs = []

            # First pass: add documents by frequency order
            for doc, freq in doc_frequency.most_common():
                if doc not in seen:
                    ranked_docs.append(doc)
                    seen.add(doc)

            # Truncate to exactly 21 documents
            final_docs = ranked_docs[:21]

            return dspy.Prediction(retrieved_docs=final_docs)
