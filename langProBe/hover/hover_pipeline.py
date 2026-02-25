import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class DecomposeClaimEntities(dspy.Signature):
    """Extract ALL named entities (people, places, organizations, works, dates) from the claim and identify the specific factual relationships being claimed (e.g., 'X directed Y', 'A and B both are C')."""

    claim: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="List of all named entities mentioned in the claim (people, places, organizations, works, dates)")
    relationships: list[str] = dspy.OutputField(desc="List of specific factual relationships or comparisons being claimed (e.g., 'X directed Y', 'A won award B', 'X and Y are both Z')")


class GenerateTargetedQueries(dspy.Signature):
    """Generate exactly 3 highly specific queries targeting retrievable facts: one for the primary entity with its key attributes, one for the secondary entity with its relationships, and one for the connecting relationship or comparison point."""

    claim: str = dspy.InputField()
    entities: list[str] = dspy.InputField()
    relationships: list[str] = dspy.InputField()
    query_1: str = dspy.OutputField(desc="Query targeting the primary entity with its key attributes")
    query_2: str = dspy.OutputField(desc="Query targeting the secondary entity with its relationships")
    query_3: str = dspy.OutputField(desc="Query targeting the connecting relationship or comparison point")


class RankByEntityCoverage(dspy.Signature):
    """Score each document by counting how many of the extracted entities and relationships it mentions. Return a ranked list of document indices from most to least relevant."""

    claim: str = dspy.InputField()
    entities: list[str] = dspy.InputField()
    relationships: list[str] = dspy.InputField()
    documents: list[str] = dspy.InputField(desc="List of document titles and snippets")
    ranked_indices: list[int] = dspy.OutputField(desc="List of document indices ranked by entity/relationship coverage (0-indexed)")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

        # Initialize Entity-Relationship Query Decomposition components
        self.decompose_entities = dspy.ChainOfThought(DecomposeClaimEntities)
        self.generate_queries = dspy.ChainOfThought(GenerateTargetedQueries)
        self.rank_by_coverage = dspy.ChainOfThought(RankByEntityCoverage)
        self.retrieve_k = dspy.Retrieve(k=11)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Extract entities and relationships from claim
            decomposition = self.decompose_entities(claim=claim)
            entities = decomposition.entities
            relationships = decomposition.relationships

            # Step 2: Generate 3 targeted queries
            queries_result = self.generate_queries(
                claim=claim,
                entities=entities,
                relationships=relationships
            )
            queries = [queries_result.query_1, queries_result.query_2, queries_result.query_3]

            # Step 3: Retrieve k=11 documents per query (33 total)
            all_docs = []
            for query in queries:
                docs = self.retrieve_k(query).passages
                all_docs.extend(docs)

            # Step 4: Deduplicate documents by title
            seen_titles = set()
            unique_docs = []
            for doc in all_docs:
                # Extract title (format: "Title | content")
                title = doc.split(" | ")[0]
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_docs.append(doc)

            # Step 5: Rank by entity coverage and keep top 21
            if len(unique_docs) <= 21:
                # If we have 21 or fewer unique docs, return them all
                return dspy.Prediction(retrieved_docs=unique_docs)
            else:
                # Rank documents by entity/relationship coverage
                ranking_result = self.rank_by_coverage(
                    claim=claim,
                    entities=entities,
                    relationships=relationships,
                    documents=unique_docs
                )
                ranked_indices = ranking_result.ranked_indices

                # Keep top 21 documents
                top_21_docs = []
                for idx in ranked_indices[:21]:
                    if 0 <= idx < len(unique_docs):
                        top_21_docs.append(unique_docs[idx])

                # If we don't have enough valid indices, fill with remaining docs
                if len(top_21_docs) < 21:
                    used_indices = set(idx for idx in ranked_indices[:21] if 0 <= idx < len(unique_docs))
                    for i, doc in enumerate(unique_docs):
                        if i not in used_indices and len(top_21_docs) < 21:
                            top_21_docs.append(doc)

                return dspy.Prediction(retrieved_docs=top_21_docs)
