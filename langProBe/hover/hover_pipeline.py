import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityExtraction(dspy.Signature):
    """Extract all key entities from the claim that need to be verified.
    Entities include named entities (people, places, organizations), dates, events, and key concepts.
    Focus on entities that would require document evidence to verify the claim."""

    claim: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="list of key entities that need verification, ordered by importance")


class EntityQueryGenerator(dspy.Signature):
    """Generate a focused search query to find documents about a specific entity mentioned in the claim.
    The query should be designed to retrieve documents that would help verify the claim's statements about this entity."""

    claim: str = dspy.InputField()
    entity: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a focused search query for this entity")


class GapAnalysis(dspy.Signature):
    """Analyze retrieved documents to identify which entities from the claim lack supporting evidence.
    Compare the extracted entities against the retrieved documents to find coverage gaps."""

    claim: str = dspy.InputField()
    entities: list[str] = dspy.InputField(desc="key entities from the claim")
    retrieved_docs: list[str] = dspy.InputField(desc="documents retrieved so far")
    missing_entities: list[str] = dspy.OutputField(desc="entities that are not adequately covered in the retrieved documents")


class FillInQueryGenerator(dspy.Signature):
    """Generate a search query to find documents covering the missing entities that were not found in initial retrieval.
    The query should target the gaps in coverage to ensure all claim components are addressed."""

    claim: str = dspy.InputField()
    missing_entities: list[str] = dspy.InputField()
    query: str = dspy.OutputField(desc="a search query targeting the missing entities")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Entity-aware retrieval modules
        self.entity_extractor = dspy.Predict(EntityExtraction)
        self.entity_query_gen = dspy.Predict(EntityQueryGenerator)
        self.gap_analyzer = dspy.Predict(GapAnalysis)
        self.fillin_query_gen = dspy.Predict(FillInQueryGenerator)
        self.retrieve_k15 = dspy.Retrieve(k=15)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Stage 1: Extract key entities from the claim
            entity_result = self.entity_extractor(claim=claim)
            entities = entity_result.entities

            # Limit to max 3 entities to stay under query limit
            entities = entities[:3]

            # Stage 2 & 3: Generate focused query per entity and retrieve k=15 docs
            all_docs = []
            for entity in entities:
                query_result = self.entity_query_gen(claim=claim, entity=entity)
                query = query_result.query
                docs = self.retrieve_k15(query).passages
                all_docs.extend(docs)

            # Stage 4: Gap analysis - identify entities lacking coverage
            gap_result = self.gap_analyzer(
                claim=claim,
                entities=entities,
                retrieved_docs=all_docs
            )
            missing_entities = gap_result.missing_entities

            # Stage 5: Fill-in query if gaps exist and we haven't hit query limit
            # We've used len(entities) queries so far, can use 1 more if needed
            if missing_entities and len(entities) < 3:
                fillin_query_result = self.fillin_query_gen(
                    claim=claim,
                    missing_entities=missing_entities
                )
                fillin_query = fillin_query_result.query
                fillin_docs = self.retrieve_k15(fillin_query).passages
                all_docs.extend(fillin_docs)

            # Stage 6: Deduplication and scoring
            # Deduplicate by title (before " | " separator)
            seen_titles = set()
            unique_docs = []
            for doc in all_docs:
                title = doc.split(" | ")[0] if " | " in doc else doc
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_docs.append(doc)

            # Score documents by counting how many entities they mention
            def score_document(doc):
                doc_lower = doc.lower()
                score = 0
                for entity in entities:
                    if entity.lower() in doc_lower:
                        score += 1
                return score

            # Sort by score (descending) and select top 21
            scored_docs = [(doc, score_document(doc)) for doc in unique_docs]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, score in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=final_docs)
