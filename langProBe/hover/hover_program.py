import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityExtraction(dspy.Signature):
    """Extract 3-4 distinct named entities from the claim that require separate Wikipedia searches.
    Focus on specific people, places, events, and organizations that are central to verifying the claim.
    Each entity should be something that would have its own Wikipedia article."""

    claim = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="3-4 specific named entities (people, places, events, organizations) that need individual Wikipedia searches")
    reasoning = dspy.OutputField(desc="Brief explanation of why these entities are critical for verification")


class EntityTargeting(dspy.Signature):
    """Determine which specific entity to target next based on what information is still missing.
    Choose the entity most likely to fill gaps in verification evidence."""

    claim = dspy.InputField()
    entities = dspy.InputField(desc="List of entities extracted from the claim")
    retrieved_titles = dspy.InputField(desc="Titles of documents already retrieved")
    hop_number = dspy.InputField(desc="Current hop number (1, 2, or 3)")

    target_entity = dspy.OutputField(desc="The specific entity to search for in this hop")
    reasoning = dspy.OutputField(desc="Why this entity is the priority for this hop")
    search_query = dspy.OutputField(desc="Focused search query for this specific entity (not a broad claim reformulation)")


class RerankDocuments(dspy.Signature):
    """Score and rank documents by their relevance to verifying the claim.
    Analyze each document's content and select the top 21 most relevant documents that provide
    the best supporting evidence for fact-checking the claim."""

    claim = dspy.InputField()
    documents = dspy.InputField(desc="List of retrieved documents to rerank")

    reasoning = dspy.OutputField(desc="Analysis of document relevance and ranking criteria")
    top_document_indices: list[int] = dspy.OutputField(desc="Exactly 21 indices (0-based) of the most relevant documents, ordered from most to least relevant")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 12  # 12 docs per hop = 36 total docs (before reranking to top 21)

        # Entity extraction with reasoning
        self.extract_entities = dspy.ChainOfThought(EntityExtraction)

        # Entity-targeted hop planning with reasoning
        self.target_entity = dspy.ChainOfThought(EntityTargeting)

        # Retriever
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Reranker to select top 21 most relevant documents
        self.rerank = dspy.ChainOfThought(RerankDocuments)

    def _extract_title(self, doc):
        """Extract the title (first line) from a document."""
        if "\n" in doc:
            return doc.split("\n")[0]
        return doc[:100]

    def _deduplicate_by_title(self, docs):
        """Deduplicate documents by title while preserving order."""
        seen_titles = set()
        deduplicated = []

        for doc in docs:
            title = self._extract_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                deduplicated.append(doc)

        return deduplicated

    def forward(self, claim):
        # Step 1: Extract 3-4 distinct named entities from the claim
        entity_result = self.extract_entities(claim=claim)
        entities = entity_result.entities

        all_docs = []
        retrieved_titles_set = set()

        # Step 2: Execute 3 entity-focused hops
        for hop_num in [1, 2, 3]:
            # Determine which entity to target in this hop
            retrieved_titles_str = ", ".join(sorted(retrieved_titles_set)) if retrieved_titles_set else "None yet"

            target_result = self.target_entity(
                claim=claim,
                entities=", ".join(entities) if isinstance(entities, list) else str(entities),
                retrieved_titles=retrieved_titles_str,
                hop_number=str(hop_num)
            )

            # Retrieve documents for the targeted entity with focused query
            query = target_result.search_query
            hop_docs = self.retrieve_k(query).passages

            # Track retrieved titles for next hop
            for doc in hop_docs:
                retrieved_titles_set.add(self._extract_title(doc))

            # Add to document collection
            all_docs.extend(hop_docs)

        # Step 3: Deduplicate by document title while preserving order
        deduplicated_docs = self._deduplicate_by_title(all_docs)

        # Step 4: Rerank documents and select top 21 most relevant
        if len(deduplicated_docs) <= 21:
            # If we have 21 or fewer docs after deduplication, return all
            return dspy.Prediction(retrieved_docs=deduplicated_docs)

        # Format documents for reranking (with indices for reference)
        docs_for_reranking = "\n\n".join([
            f"[Document {i}]\n{doc}"
            for i, doc in enumerate(deduplicated_docs)
        ])

        # Rerank and get top 21 document indices
        rerank_result = self.rerank(
            claim=claim,
            documents=docs_for_reranking
        )

        # Extract top 21 indices (handle potential issues with output format)
        top_indices = rerank_result.top_document_indices[:21]

        # Validate indices and select top 21 documents
        valid_indices = [
            idx for idx in top_indices
            if isinstance(idx, int) and 0 <= idx < len(deduplicated_docs)
        ]

        # If we don't have enough valid indices, fill with remaining docs in order
        if len(valid_indices) < 21:
            remaining_indices = [
                i for i in range(len(deduplicated_docs))
                if i not in valid_indices
            ]
            valid_indices.extend(remaining_indices[:21 - len(valid_indices)])

        # Select the top 21 reranked documents
        reranked_docs = [deduplicated_docs[idx] for idx in valid_indices[:21]]

        return dspy.Prediction(retrieved_docs=reranked_docs)


