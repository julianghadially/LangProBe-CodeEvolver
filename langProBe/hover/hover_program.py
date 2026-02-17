import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimEntityExtractor(dspy.Signature):
    """Extract 2-4 key entities from the claim that would be important for fact-checking.
    Entities should be specific named entities (people, places, organizations, dates, etc.)
    that are central to verifying or refuting the claim."""

    claim = dspy.InputField(desc="The claim to extract entities from")
    entities: list[str] = dspy.OutputField(
        desc="A list of 2-4 key entities from the claim (e.g., person names, places, organizations, dates)"
    )


class CoverageAwareQueryGenerator(dspy.Signature):
    """Generate a search query that targets entities from the claim that have not yet been covered
    by the already-retrieved documents. Focus on uncovered entities to maximize information diversity."""

    claim = dspy.InputField(desc="The claim to fact-check")
    key_entities: list[str] = dspy.InputField(desc="Key entities extracted from the claim")
    retrieved_titles: list[str] = dspy.InputField(
        desc="Titles of documents already retrieved in previous hops"
    )
    query = dspy.OutputField(
        desc="A search query targeting entities not yet covered in the retrieved documents"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.entity_extractor = dspy.ChainOfThought(ClaimEntityExtractor)
        self.query_generator = dspy.ChainOfThought(CoverageAwareQueryGenerator)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _normalize_title(self, doc):
        """Extract and normalize the title from a document string."""
        # Documents are in format "title | content"
        title = doc.split(" | ")[0] if " | " in doc else doc
        return dspy.evaluate.normalize_text(title)

    def _get_covered_entities(self, key_entities, retrieved_titles):
        """Check which entities appear in the retrieved document titles (case-insensitive)."""
        covered_entities = set()
        normalized_titles = [title.lower() for title in retrieved_titles]

        for entity in key_entities:
            entity_lower = entity.lower()
            for title in normalized_titles:
                if entity_lower in title:
                    covered_entities.add(entity)
                    break

        return covered_entities

    def forward(self, claim):
        # Extract key entities from the claim
        entities_result = self.entity_extractor(claim=claim)
        key_entities = entities_result.entities

        # Track all documents and their normalized titles for deduplication
        all_docs = []
        seen_titles = set()

        # HOP 1: Initial retrieval based on the claim
        hop1_docs = self.retrieve_k(claim).passages
        for doc in hop1_docs:
            normalized_title = self._normalize_title(doc)
            if normalized_title not in seen_titles:
                all_docs.append(doc)
                seen_titles.add(normalized_title)

        # Get titles for coverage tracking
        retrieved_titles = [doc.split(" | ")[0] for doc in all_docs]

        # HOP 2: Generate query targeting uncovered entities
        hop2_query = self.query_generator(
            claim=claim,
            key_entities=key_entities,
            retrieved_titles=retrieved_titles
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        for doc in hop2_docs:
            normalized_title = self._normalize_title(doc)
            if normalized_title not in seen_titles:
                all_docs.append(doc)
                seen_titles.add(normalized_title)

        # Update retrieved titles for next hop
        retrieved_titles = [doc.split(" | ")[0] for doc in all_docs]

        # HOP 3: Generate query targeting remaining uncovered entities
        hop3_query = self.query_generator(
            claim=claim,
            key_entities=key_entities,
            retrieved_titles=retrieved_titles
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        for doc in hop3_docs:
            normalized_title = self._normalize_title(doc)
            if normalized_title not in seen_titles:
                all_docs.append(doc)
                seen_titles.add(normalized_title)

        return dspy.Prediction(retrieved_docs=all_docs)
