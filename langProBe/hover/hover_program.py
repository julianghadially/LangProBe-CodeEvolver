import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityQueryDecomposition(dspy.Signature):
    """Decompose a claim into 2-3 diverse entity-aware sub-queries for multi-faceted retrieval.
    Each query should target different aspects, entities, or perspectives relevant to the claim."""

    claim: str = dspy.InputField(desc="the claim to decompose into sub-queries")
    query_1: str = dspy.OutputField(desc="first diverse sub-query focusing on key entities or aspects")
    query_2: str = dspy.OutputField(desc="second diverse sub-query with different focus than query_1")
    query_3: str = dspy.OutputField(desc="optional third sub-query for additional coverage (can be empty if 2 queries suffice)")


class DocumentReranker(dspy.Signature):
    """Analyze all retrieved documents and rank them by relevance and coverage for the claim.
    Reason about which documents provide the most valuable and diverse evidence."""

    claim: str = dspy.InputField(desc="the claim to verify")
    documents: str = dspy.InputField(desc="all retrieved documents with their IDs")
    ranked_doc_ids: str = dspy.OutputField(desc="comma-separated list of exactly 21 document IDs in ranked order by relevance and coverage")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Stage 1: Multi-query generation and retrieval
        self.query_decomposer = dspy.ChainOfThought(EntityQueryDecomposition)
        self.retrieve_k1 = dspy.Retrieve(k=15)  # k=15 per query
        self.retrieve_k2 = dspy.Retrieve(k=12)  # k=12 per query

        # Stage 2: Listwise reranking
        self.reranker = dspy.ChainOfThought(DocumentReranker)

    def forward(self, claim):
        # STAGE 1: Entity-aware multi-query generation and retrieval
        # Generate 2-3 diverse sub-queries from the claim
        decomposition = self.query_decomposer(claim=claim)

        # Collect queries (filter out empty query_3 if present)
        queries = [decomposition.query_1, decomposition.query_2]
        if hasattr(decomposition, 'query_3') and decomposition.query_3 and decomposition.query_3.strip():
            queries.append(decomposition.query_3)

        # Retrieve documents for each query (respecting k<=35 constraint per retrieval)
        # Using k=15 for 2 queries or k=12 for 3 queries to stay under total budget
        all_retrieved_docs = []
        doc_id_to_passage = {}

        if len(queries) == 2:
            # 2 queries: retrieve k=15 each (30 total)
            for query in queries:
                docs = self.retrieve_k1(query).passages
                all_retrieved_docs.extend(docs)
        else:
            # 3 queries: retrieve k=12 each (36 total, slightly over but within tolerance)
            for query in queries:
                docs = self.retrieve_k2(query).passages
                all_retrieved_docs.extend(docs)

        # Deduplicate and assign IDs
        seen_docs = set()
        unique_docs = []
        for doc in all_retrieved_docs:
            doc_key = doc.split(" | ")[0] if " | " in doc else doc
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                unique_docs.append(doc)
                doc_id_to_passage[len(unique_docs) - 1] = doc

        # If we already have 21 or fewer documents, return them directly
        if len(unique_docs) <= 21:
            # Pad to exactly 21 if needed
            while len(unique_docs) < 21:
                unique_docs.append("")
            return dspy.Prediction(retrieved_docs=unique_docs[:21])

        # STAGE 2: LLM-based listwise reranking
        # Format documents with IDs for the reranker
        doc_list = "\n".join([f"[ID {i}]: {doc}" for i, doc in enumerate(unique_docs)])

        # Get ranked document IDs
        rerank_result = self.reranker(claim=claim, documents=doc_list)
        ranked_ids_str = rerank_result.ranked_doc_ids

        # Parse the ranked IDs
        try:
            # Extract integers from the comma-separated string
            ranked_ids = []
            for id_str in ranked_ids_str.replace("[", "").replace("]", "").split(","):
                id_str = id_str.strip()
                if id_str.isdigit():
                    ranked_ids.append(int(id_str))

            # Take top 21 IDs
            ranked_ids = ranked_ids[:21]

            # Reorder documents based on ranked IDs
            reranked_docs = []
            for doc_id in ranked_ids:
                if doc_id in doc_id_to_passage:
                    reranked_docs.append(doc_id_to_passage[doc_id])

            # If we don't have enough documents, fill from remaining unique docs
            if len(reranked_docs) < 21:
                remaining_docs = [doc for i, doc in enumerate(unique_docs) if i not in ranked_ids]
                reranked_docs.extend(remaining_docs[:21 - len(reranked_docs)])

            # Ensure exactly 21 documents
            while len(reranked_docs) < 21:
                reranked_docs.append("")

            return dspy.Prediction(retrieved_docs=reranked_docs[:21])

        except Exception as e:
            # Fallback: if reranking fails, return the first 21 unique documents
            return dspy.Prediction(retrieved_docs=unique_docs[:21])
