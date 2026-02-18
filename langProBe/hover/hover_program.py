import re
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
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


class HoverParallelEntityRetrieval(LangProBeDSPyMetaProgram, dspy.Module):
    '''Parallel entity-focused retrieval system for multi-hop claims.

    EVALUATION
    - Retrieves documents based on entity decomposition
    - Performs 2-3 parallel entity queries (k=10 each)
    - Reranks all documents to select top 21 most relevant
    - Must provide exactly 21 documents at the end
    '''

    def __init__(self):
        super().__init__()
        self.k_per_entity = 10
        self.max_final_docs = 21

        # DSPy Modules
        self.decompose_claim = dspy.ChainOfThought("claim -> entities, relationships")
        self.generate_entity_query = dspy.ChainOfThought("claim, entity, relationships -> query")
        self.retrieve_k = dspy.Retrieve(k=self.k_per_entity)
        self.rerank_documents = dspy.ChainOfThought("claim, documents -> ranked_indices")

    def forward(self, claim):
        # Step 1: Decompose claim into entities
        decomposition = self.decompose_claim(claim=claim)
        entities = self._parse_entities(decomposition.entities)
        relationships = decomposition.relationships

        # Normalize to 2-3 entities (constraint compliance)
        entities = self._normalize_entity_count(entities)

        # Step 2: Generate entity-specific queries
        entity_queries = []
        for entity in entities:
            query_result = self.generate_entity_query(
                claim=claim, entity=entity, relationships=relationships
            )
            entity_queries.append({'entity': entity, 'query': query_result.query})

        # Step 3: Retrieve documents for each query
        all_documents = []
        doc_metadata = []
        for eq in entity_queries:
            docs = self.retrieve_k(eq['query']).passages
            for doc in docs:
                all_documents.append(doc)
                doc_metadata.append({'entity': eq['entity'], 'query': eq['query'], 'doc': doc})

        # Step 4: Deduplicate
        unique_docs, unique_metadata = self._deduplicate_with_metadata(all_documents, doc_metadata)

        # Step 5: Rerank and select top 21
        if len(unique_docs) > self.max_final_docs:
            final_docs = self._rerank_and_select(claim, unique_docs, unique_metadata)
        else:
            final_docs = unique_docs

        # Step 6: Ensure exactly 21 documents
        final_docs = self._pad_to_target_size(final_docs, self.max_final_docs)

        return dspy.Prediction(retrieved_docs=final_docs[:self.max_final_docs])

    def _parse_entities(self, entities_raw):
        """Parse entities from LLM output (string or list format)"""
        # Handle list format
        if isinstance(entities_raw, list):
            return entities_raw

        # Handle string formats: numbered list or comma-separated
        if '\n' in entities_raw:
            entities = [line.strip() for line in entities_raw.split('\n') if line.strip()]
            entities = [re.sub(r'^\d+\.\s*', '', e) for e in entities]
        else:
            entities = [e.strip() for e in entities_raw.split(',')]

        return [e for e in entities if e]

    def _normalize_entity_count(self, entities):
        """Ensure 2-3 entities for query constraint"""
        if len(entities) < 2:
            return [entities[0], entities[0]] if entities else []
        elif len(entities) > 3:
            return entities[:3]
        return entities

    def _deduplicate_with_metadata(self, documents, metadata):
        """Remove duplicates while preserving metadata"""
        seen = set()
        unique_docs = []
        unique_metadata = []

        for doc, meta in zip(documents, metadata):
            doc_normalized = doc.strip().lower()
            if doc_normalized not in seen:
                seen.add(doc_normalized)
                unique_docs.append(doc)
                unique_metadata.append(meta)

        return unique_docs, unique_metadata

    def _rerank_and_select(self, claim, documents, metadata):
        """Rerank documents and select top 21"""
        formatted_docs = self._format_docs_for_ranking(documents, metadata)
        rerank_result = self.rerank_documents(claim=claim, documents=formatted_docs)
        ranked_indices = self._parse_ranked_indices(rerank_result.ranked_indices, len(documents))
        return [documents[i] for i in ranked_indices[:self.max_final_docs]]

    def _format_docs_for_ranking(self, documents, metadata):
        """Format documents with indices for reranking"""
        formatted = []
        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
            formatted.append(f"[{i}] (Entity: {meta['entity']}) {doc_preview}")
        return "\n\n".join(formatted)

    def _parse_ranked_indices(self, ranked_output, max_index):
        """Parse ranked indices from LLM output"""
        if isinstance(ranked_output, list):
            indices = [int(x) for x in ranked_output if str(x).isdigit()]
        else:
            numbers = re.findall(r'\d+', str(ranked_output))
            indices = [int(x) for x in numbers]

        # Validate and fill missing indices
        indices = [i for i in indices if 0 <= i < max_index]
        if len(indices) < max_index:
            missing = [i for i in range(max_index) if i not in indices]
            indices.extend(missing)

        return indices

    def _pad_to_target_size(self, documents, target_size):
        """Pad to exactly 21 documents if needed"""
        if len(documents) >= target_size:
            return documents

        padding_needed = target_size - len(documents)
        padding = ["[No additional relevant documents found]"] * padding_needed
        return documents + padding
