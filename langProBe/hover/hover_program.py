import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        # Entity tracking signature
        self.entity_tracker = dspy.ChainOfThought("claim,passages->key_entities")
        # Dual query generation for hop 2 and 3
        self.create_entity_query_hop2 = dspy.ChainOfThought("claim,key_entities->query")
        self.create_semantic_query_hop2 = dspy.ChainOfThought("claim,passages->query")
        self.create_entity_query_hop3 = dspy.ChainOfThought("claim,key_entities,context->query")
        self.create_semantic_query_hop3 = dspy.ChainOfThought("claim,context,passages->query")
        # Retrievers for dual paths
        self.retrieve_k_half = dspy.Retrieve(k=self.k // 2)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _mmr_deduplicate(self, all_docs, tracked_entities, target_size=21):
        """Implements MMR-style deduplication with entity tracking and semantic diversity.

        Args:
            all_docs: List of document strings
            tracked_entities: List of entity strings extracted from passages
            target_size: Final number of documents to return (default 21)

        Returns:
            List of deduplicated documents maximizing entity coverage and semantic diversity
        """
        if not all_docs:
            return []

        # Step 1: Remove exact duplicates while preserving order
        seen_docs = set()
        unique_docs = []
        for doc in all_docs:
            if doc not in seen_docs:
                seen_docs.add(doc)
                unique_docs.append(doc)

        # Step 2: Extract titles and compute uniqueness scores
        doc_metadata = []
        seen_titles = {}

        for doc in unique_docs:
            # Extract title (assuming format "Title | content" or first sentence)
            title = doc.split('|')[0].strip() if '|' in doc else doc.split('.')[0].strip()

            # Count entity mentions in document
            entity_count = sum(1 for entity in tracked_entities if entity.lower() in doc.lower())

            # Title uniqueness score (lower is better for duplicates)
            title_count = seen_titles.get(title, 0)
            seen_titles[title] = title_count + 1

            doc_metadata.append({
                'doc': doc,
                'title': title,
                'entity_count': entity_count,
                'title_occurrence': title_count,
                'selected': False
            })

        # Step 3: MMR-style selection
        selected_docs = []

        # First pass: prioritize documents with entities and unique titles
        for meta in doc_metadata:
            if len(selected_docs) >= target_size:
                break
            # Select if has entities and is first occurrence of title
            if meta['entity_count'] > 0 and meta['title_occurrence'] == 0:
                selected_docs.append(meta['doc'])
                meta['selected'] = True

        # Second pass: add documents with entities (even if duplicate titles)
        for meta in doc_metadata:
            if len(selected_docs) >= target_size:
                break
            if not meta['selected'] and meta['entity_count'] > 0:
                selected_docs.append(meta['doc'])
                meta['selected'] = True

        # Third pass: add unique title documents for semantic diversity
        for meta in doc_metadata:
            if len(selected_docs) >= target_size:
                break
            if not meta['selected'] and meta['title_occurrence'] == 0:
                selected_docs.append(meta['doc'])
                meta['selected'] = True

        # Fourth pass: fill remaining slots with any remaining documents
        for meta in doc_metadata:
            if len(selected_docs) >= target_size:
                break
            if not meta['selected']:
                selected_docs.append(meta['doc'])
                meta['selected'] = True

        return selected_docs[:target_size]

    def forward(self, claim):
        # HOP 1: Initial retrieval and entity extraction
        hop1_docs = self.retrieve_k(claim).passages

        # Extract entities from hop 1 documents
        entities_1 = self.entity_tracker(
            claim=claim, passages=hop1_docs
        ).key_entities

        # Parse entities into a list for tracking
        tracked_entities = [e.strip() for e in entities_1.split(',') if e.strip()]

        # HOP 2: Dual-path retrieval
        # Entity-focused query
        entity_query_hop2 = self.create_entity_query_hop2(
            claim=claim, key_entities=entities_1
        ).query
        hop2_entity_docs = self.retrieve_k_half(entity_query_hop2).passages

        # Semantic query
        semantic_query_hop2 = self.create_semantic_query_hop2(
            claim=claim, passages=hop1_docs
        ).query
        hop2_semantic_docs = self.retrieve_k_half(semantic_query_hop2).passages

        # Combine hop 2 documents
        hop2_docs = hop2_entity_docs + hop2_semantic_docs

        # Extract entities from hop 2 documents
        entities_2 = self.entity_tracker(
            claim=claim, passages=hop2_docs
        ).key_entities

        # Update tracked entities
        new_entities = [e.strip() for e in entities_2.split(',') if e.strip()]
        tracked_entities.extend(new_entities)
        tracked_entities = list(set(tracked_entities))  # Remove duplicates

        # HOP 3: Dual-path retrieval with accumulated context
        context = f"Entities found: {', '.join(tracked_entities)}"

        # Entity-focused query
        entity_query_hop3 = self.create_entity_query_hop3(
            claim=claim, key_entities=', '.join(tracked_entities), context=context
        ).query
        hop3_entity_docs = self.retrieve_k_half(entity_query_hop3).passages

        # Semantic query
        semantic_query_hop3 = self.create_semantic_query_hop3(
            claim=claim, context=context, passages=hop2_docs
        ).query
        hop3_semantic_docs = self.retrieve_k_half(semantic_query_hop3).passages

        # Combine hop 3 documents
        hop3_docs = hop3_entity_docs + hop3_semantic_docs

        # Extract final entities from hop 3
        entities_3 = self.entity_tracker(
            claim=claim, passages=hop3_docs
        ).key_entities
        new_entities = [e.strip() for e in entities_3.split(',') if e.strip()]
        tracked_entities.extend(new_entities)
        tracked_entities = list(set(tracked_entities))  # Remove duplicates

        # Combine all documents
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Apply MMR-style deduplication with entity tracking
        final_docs = self._mmr_deduplicate(all_docs, tracked_entities, target_size=21)

        return dspy.Prediction(retrieved_docs=final_docs)
