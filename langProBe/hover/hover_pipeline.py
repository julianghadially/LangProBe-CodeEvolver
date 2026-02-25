import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ExtractKeyEntities(dspy.Signature):
    """Analyze the claim to identify 3-5 named entities (people, places, organizations, specific works) that are likely to be supporting facts."""
    claim: str = dspy.InputField()
    entities: str = dspy.OutputField(desc="A comma-separated list of 3-5 named entities from the claim (people, places, organizations, specific works)")


class EntityToQuery(dspy.Signature):
    """Convert an entity name into a retrieval-optimized query."""
    entity: str = dspy.InputField()
    claim_context: str = dspy.InputField(desc="The original claim for context")
    query: str = dspy.OutputField(desc="A retrieval-optimized query targeting this entity")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()
        # Entity-first retrieval components
        self.extract_entities = dspy.Predict(ExtractKeyEntities)
        self.entity_to_query = dspy.Predict(EntityToQuery)
        self.retrieve_12 = dspy.Retrieve(k=12)
        self.retrieve_21 = dspy.Retrieve(k=21)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Extract key entities from the claim
            entities_result = self.extract_entities(claim=claim)
            entity_list = [e.strip() for e in entities_result.entities.split(',') if e.strip()]

            # Limit to first 3 entities to stay within constraint
            entity_list = entity_list[:3]

            # Step 2: Generate targeted queries for each entity
            all_docs = []
            seen_titles = set()

            for entity in entity_list:
                # Generate entity-specific query
                query_result = self.entity_to_query(entity=entity, claim_context=claim)
                entity_query = query_result.query

                # Step 3: Retrieve k=12 documents per entity query
                entity_docs = self.retrieve_12(entity_query).passages

                # Step 4: Deduplicate by title
                for doc in entity_docs:
                    # Extract title from document (format: "title | content")
                    title = doc.split(" | ")[0] if " | " in doc else doc
                    if title not in seen_titles:
                        seen_titles.add(title)
                        all_docs.append(doc)

            # Step 5: If fewer than 21 unique documents, add broader query
            if len(all_docs) < 21:
                remaining_slots = 21 - len(all_docs)
                broader_docs = self.retrieve_21(claim).passages

                for doc in broader_docs:
                    if len(all_docs) >= 21:
                        break
                    title = doc.split(" | ")[0] if " | " in doc else doc
                    if title not in seen_titles:
                        seen_titles.add(title)
                        all_docs.append(doc)

            # Step 6: Apply lightweight relevance filter
            # Score documents based on entity name overlap with claim
            entity_set = set()
            for entity in entity_list:
                entity_set.update(entity.lower().split())

            def relevance_score(doc):
                doc_text = doc.lower()
                score = sum(1 for entity_word in entity_set if entity_word in doc_text)
                return score

            # Sort by relevance score (descending) and keep top 21
            scored_docs = [(doc, relevance_score(doc)) for doc in all_docs]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_docs)
