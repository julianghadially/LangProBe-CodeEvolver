import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ClaimEntityExtractor(dspy.Signature):
    """Extract 3-5 key named entities (people, places, organizations, titles) from the claim.
    Focus on specific entities that would be searchable in documents."""

    claim: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="3-5 key named entities from the claim")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.entity_extractor = dspy.Predict(ClaimEntityExtractor)
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=50)  # Retrieve more for entity queries
        self.retrieve_hop = dspy.Retrieve(k=self.k)  # For hops 2 and 3
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Extract entities from claim for entity-focused query decomposition
            entity_result = self.entity_extractor(claim=claim)
            entities = entity_result.entities

            # Limit to top 3 entities for constraint compliance
            top_entities = entities[:3] if isinstance(entities, list) else [entities]

            # HOP 1: Entity-focused retrieval
            # Retrieve documents for each entity (k=50 per entity, up to 150 total)
            all_hop1_docs = []
            doc_scores = {}  # Track document frequency as utility score

            for entity in top_entities:
                entity_docs = self.retrieve_k(entity).passages
                for doc in entity_docs:
                    all_hop1_docs.append(doc)
                    # Higher score for documents retrieved by multiple entity queries
                    doc_scores[doc] = doc_scores.get(doc, 0) + 1

            # Apply utility-based reranking: prioritize docs that appear in multiple entity searches
            # Remove duplicates and sort by utility score (frequency across entity queries)
            unique_docs = []
            seen_titles = set()

            # Sort by utility score (frequency), then by original retrieval order
            sorted_docs = sorted(all_hop1_docs, key=lambda doc: (-doc_scores[doc], all_hop1_docs.index(doc)))

            for doc in sorted_docs:
                doc_title = doc.split(" | ")[0] if " | " in doc else doc
                if doc_title not in seen_titles:
                    seen_titles.add(doc_title)
                    unique_docs.append(doc)

            # Select top 7 documents for hop1 (to maintain 21 total: 7+7+7)
            hop1_docs = unique_docs[:self.k]

            # Summarize hop1 results
            summary_1 = self.summarize1(
                claim=claim, passages=hop1_docs
            ).summary

            # HOP 2
            hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
            hop2_docs = self.retrieve_hop(hop2_query).passages
            summary_2 = self.summarize2(
                claim=claim, context=summary_1, passages=hop2_docs
            ).summary

            # HOP 3
            hop3_query = self.create_query_hop3(
                claim=claim, summary_1=summary_1, summary_2=summary_2
            ).query
            hop3_docs = self.retrieve_hop(hop3_query).passages

            # Return all documents from three hops (21 total)
            return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
