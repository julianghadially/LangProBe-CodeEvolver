import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityExtractionHop1Signature(dspy.Signature):
    """Extract 3-5 key entities from retrieved passages that are mentioned in the claim
    but need more information. Focus on specific names, organizations, dates, locations,
    or titles that would benefit from additional retrieval."""

    claim = dspy.InputField(desc="The claim to verify")
    passages = dspy.InputField(desc="Retrieved passages from hop 1")
    entities: list[str] = dspy.OutputField(
        desc="List of 3-5 key entities (people, bands, organizations, dates, titles) that need more context"
    )


class EntityExtractionHop2Signature(dspy.Signature):
    """Extract bridging entities from hop 2 passages that connect to the original entities.
    These should be new entities that help establish relationships with entities from hop 1."""

    claim = dspy.InputField(desc="The claim to verify")
    original_entities = dspy.InputField(desc="Entities extracted from hop 1")
    passages = dspy.InputField(desc="Retrieved passages from hop 2")
    bridging_entities: list[str] = dspy.OutputField(
        desc="List of 3-5 new entities that bridge/connect to the original entities"
    )


class QueryHop2Signature(dspy.Signature):
    """Generate a search query that explores relationships between the claim and extracted entities."""

    claim = dspy.InputField(desc="The claim to verify")
    entities = dspy.InputField(desc="Key entities from hop 1 that need more context")
    query = dspy.OutputField(desc="Search query asking about relationships between these entities")


class QueryHop3Signature(dspy.Signature):
    """Generate a search query that explicitly connects original claim entities to bridging entities."""

    claim = dspy.InputField(desc="The claim to verify")
    original_entities = dspy.InputField(desc="Entities from hop 1")
    bridging_entities = dspy.InputField(desc="Bridging entities from hop 2")
    query = dspy.OutputField(
        desc="Search query asking about the connection between original and bridging entities"
    )


class ClaimEntityExtraction(dspy.Signature):
    """Extract ALL key entities mentioned in the claim for comprehensive fact verification.
    Identify every person, organization, title, date, location, and other named entities."""

    claim = dspy.InputField(desc="The claim to verify")
    entities: list[str] = dspy.OutputField(
        desc="Complete list of ALL entities in the claim: people, organizations, titles, dates, locations, events, etc."
    )


class EntityDiversityReranker(dspy.Module):
    """Reranks retrieved documents using a greedy set-cover algorithm to maximize entity coverage."""

    def __init__(self):
        super().__init__()
        self.extract_claim_entities = dspy.ChainOfThought(ClaimEntityExtraction)

    def forward(self, claim, documents):
        """
        Rerank documents to maximize coverage of entities in the claim.

        Args:
            claim: The claim to verify
            documents: List of retrieved documents (21 total from 3 hops)

        Returns:
            dspy.Prediction with reranked_docs (list of documents)
        """
        # Step 1: Extract all entities from the claim
        extraction_result = self.extract_claim_entities(claim=claim)
        claim_entities = extraction_result.entities  # list[str]

        # Normalize entities to lowercase for matching
        claim_entities_normalized = set(entity.lower() for entity in claim_entities)

        # Step 2: Score each document based on unique entity coverage
        doc_scores = []
        for idx, doc in enumerate(documents):
            doc_lower = doc.lower()
            # Count how many claim entities appear in this document
            covered_entities = {entity for entity in claim_entities_normalized if entity in doc_lower}
            doc_scores.append({
                'idx': idx,
                'doc': doc,
                'covered_entities': covered_entities,
                'score': len(covered_entities)
            })

        # Step 3: Greedy set-cover algorithm to select documents
        selected_docs = []
        covered_so_far = set()
        remaining_docs = doc_scores.copy()

        # Greedily select documents that cover the most uncovered entities
        while remaining_docs and len(selected_docs) < len(documents):
            # Calculate marginal coverage for each remaining document
            best_doc = None
            best_marginal_coverage = 0

            for doc_info in remaining_docs:
                # How many NEW entities does this document cover?
                marginal_coverage = len(doc_info['covered_entities'] - covered_so_far)

                if marginal_coverage > best_marginal_coverage:
                    best_marginal_coverage = marginal_coverage
                    best_doc = doc_info

            # If no document adds new coverage, break
            if best_doc is None or best_marginal_coverage == 0:
                # Add remaining documents in their original order
                for doc_info in remaining_docs:
                    selected_docs.append(doc_info['doc'])
                break

            # Select the best document
            selected_docs.append(best_doc['doc'])
            covered_so_far.update(best_doc['covered_entities'])
            remaining_docs.remove(best_doc)

            # If all entities are covered, add remaining docs in original order
            if covered_so_far >= claim_entities_normalized:
                for doc_info in remaining_docs:
                    selected_docs.append(doc_info['doc'])
                break

        return dspy.Prediction(reranked_docs=selected_docs)


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7

        # Entity extraction modules (use ChainOfThought for better reasoning)
        self.extract_entities_hop1 = dspy.ChainOfThought(EntityExtractionHop1Signature)
        self.extract_entities_hop2 = dspy.ChainOfThought(EntityExtractionHop2Signature)

        # Query generation modules (use ChainOfThought for complex reasoning)
        self.create_query_hop2 = dspy.ChainOfThought(QueryHop2Signature)
        self.create_query_hop3 = dspy.ChainOfThought(QueryHop3Signature)

        # Retrieval module
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Diversity-aware reranker
        self.reranker = EntityDiversityReranker()

    def forward(self, claim):
        # HOP 1: Initial retrieval with claim
        hop1_docs = self.retrieve_k(claim).passages

        # Extract key entities from hop1 that need more information
        entities_1_result = self.extract_entities_hop1(
            claim=claim,
            passages=hop1_docs
        )
        entities_1 = entities_1_result.entities  # list[str]

        # HOP 2: Query about relationships between claim and extracted entities
        hop2_query_result = self.create_query_hop2(
            claim=claim,
            entities=", ".join(entities_1)  # Convert list to comma-separated string
        )
        hop2_query = hop2_query_result.query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # Extract bridging entities from hop2 that connect to original entities
        entities_2_result = self.extract_entities_hop2(
            claim=claim,
            original_entities=", ".join(entities_1),
            passages=hop2_docs
        )
        bridging_entities = entities_2_result.bridging_entities  # list[str]

        # HOP 3: Query about connection between original and bridging entities
        hop3_query_result = self.create_query_hop3(
            claim=claim,
            original_entities=", ".join(entities_1),
            bridging_entities=", ".join(bridging_entities)
        )
        hop3_query = hop3_query_result.query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all retrieved documents (21 total: 7 per hop)
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # DIVERSITY-AWARE RERANKING: Maximize entity coverage
        reranking_result = self.reranker(claim=claim, documents=all_docs)
        reranked_docs = reranking_result.reranked_docs

        # Return reranked documents
        return dspy.Prediction(retrieved_docs=reranked_docs)


