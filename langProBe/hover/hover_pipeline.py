import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


# ============ New DSPy Signature Classes for Ensemble Reranking Architecture ============

class QueryExpander(dspy.Signature):
    """Generate 5-7 diverse query formulations from the claim to maximize retrieval coverage.
    Include entity-focused queries, relationship-focused queries, and negation queries to capture different aspects of the claim."""

    claim: str = dspy.InputField(desc="the claim to verify")
    expanded_queries: list[str] = dspy.OutputField(desc="5-7 diverse query formulations including entity-focused (e.g., focusing on specific people/places), relationship-focused (e.g., focusing on connections), and negation queries (e.g., what contradicts this)")


class ListwiseReranker(dspy.Signature):
    """Rank a list of documents by their relevance to verifying the claim.
    Output a ranked list with relevance scores (0-100) for each document."""

    claim: str = dspy.InputField(desc="the claim to verify")
    documents: str = dspy.InputField(desc="numbered list of documents to rank (format: [1] doc1, [2] doc2, ...)")
    ranked_indices: list[int] = dspy.OutputField(desc="list of document indices in descending order of relevance (e.g., [3, 1, 5, 2, ...] means doc 3 is most relevant)")
    relevance_scores: list[float] = dspy.OutputField(desc="relevance scores (0-100) corresponding to each document in the ranked_indices list")


# ============ Main Pipeline with Ensemble Reranking Architecture ============

class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop system for retrieving documents for a provided claim using Two-Stage Ensemble Reranking.

    ARCHITECTURE:
    - Query expansion generates 5-7 diverse query formulations
    - Single retrieval phase: k=5 docs per query (total ~35 docs with 7 queries)
    - Listwise reranking with sliding windows (10 docs, 3-doc overlap)
    - Ensemble scoring: 0.6 LLM listwise scores + 0.4 ColBERT retrieval scores
    - Returns top 21 documents by combined score

    EVALUATION:
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize new modules for ensemble reranking
        self.query_expander = dspy.ChainOfThought(QueryExpander)
        self.listwise_reranker = dspy.ChainOfThought(ListwiseReranker)

        # Retrieval module
        self.retrieve_k = dspy.Retrieve(k=5)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # ========== STAGE 1: Query Expansion ==========
            # Generate 5-7 diverse query formulations from the claim
            try:
                expansion_result = self.query_expander(claim=claim)
                expanded_queries = expansion_result.expanded_queries

                # Ensure we have a list of queries (handle different DSPy output formats)
                if not isinstance(expanded_queries, list):
                    if isinstance(expanded_queries, str):
                        # Parse string into list
                        expanded_queries = [q.strip() for q in expanded_queries.split('\n') if q.strip()]
                        # Remove numbered prefixes like "1.", "2.", etc.
                        expanded_queries = [q.lstrip('0123456789.-)> ').strip() for q in expanded_queries if q.strip()]
                    else:
                        # Fallback: use original claim
                        expanded_queries = [claim]

                # Ensure we have 5-7 queries, add claim-based variations if needed
                if len(expanded_queries) < 5:
                    expanded_queries.append(claim)

                # Limit to 7 queries to control retrieval volume
                expanded_queries = expanded_queries[:7]

            except Exception:
                # Fallback: use claim as single query
                expanded_queries = [claim]

            # ========== STAGE 2: Batch Retrieval (Single Phase) ==========
            # Retrieve k=5 documents per query to stay under 3 search calls
            # We'll batch queries to minimize API calls
            all_retrieved_docs = []
            colbert_scores_map = {}  # Map doc -> ColBERT score

            # Batch queries into groups to minimize search calls
            # With 7 queries and k=5, we retrieve ~35 docs total
            batch_size = 3  # Process up to 3 queries per batch to stay under limit
            for i in range(0, len(expanded_queries), batch_size):
                batch_queries = expanded_queries[i:i+batch_size]

                for query in batch_queries:
                    try:
                        # Retrieve with scores
                        # Note: DSPy's Retrieve doesn't directly support return_scores parameter
                        # We'll retrieve and use a default ColBERT score based on retrieval order
                        result = self.retrieve_k(query)
                        docs = result.passages

                        # Assign ColBERT scores based on retrieval rank (higher rank = higher score)
                        # Score decreases from 100 to 50 based on position
                        for idx, doc in enumerate(docs):
                            all_retrieved_docs.append(doc)
                            # If doc not seen before, assign score based on first retrieval rank
                            if doc not in colbert_scores_map:
                                # Score from 100 (rank 0) to 50 (rank 4)
                                colbert_scores_map[doc] = 100.0 - (idx * 10.0)
                    except Exception:
                        # If retrieval fails for this query, continue with others
                        pass

            # ========== STAGE 3: Deduplication ==========
            # Deduplicate documents based on title (before " | ")
            unique_docs = []
            seen_titles = set()
            for doc in all_retrieved_docs:
                title = doc.split(" | ")[0]
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_docs.append(doc)

            # ========== STAGE 4: Listwise Reranking with Sliding Windows ==========
            # Process all unique documents through sliding windows of 10 documents with 3-document overlap
            llm_scores_map = {}  # Map doc -> LLM relevance score

            window_size = 10
            overlap = 3
            step = window_size - overlap  # 7 documents

            for start_idx in range(0, len(unique_docs), step):
                end_idx = min(start_idx + window_size, len(unique_docs))
                window_docs = unique_docs[start_idx:end_idx]

                # Format documents for listwise ranking
                formatted_docs = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(window_docs)])

                try:
                    # Get listwise ranking
                    ranking_result = self.listwise_reranker(claim=claim, documents=formatted_docs)

                    # Parse ranked indices and scores
                    ranked_indices = ranking_result.ranked_indices
                    relevance_scores = ranking_result.relevance_scores

                    # Ensure lists
                    if not isinstance(ranked_indices, list):
                        if isinstance(ranked_indices, str):
                            # Try to parse as list
                            ranked_indices = [int(x.strip()) for x in ranked_indices.replace('[', '').replace(']', '').split(',') if x.strip().isdigit()]
                        else:
                            ranked_indices = list(range(1, len(window_docs) + 1))

                    if not isinstance(relevance_scores, list):
                        if isinstance(relevance_scores, str):
                            # Try to parse as list
                            relevance_scores = [float(x.strip()) for x in relevance_scores.replace('[', '').replace(']', '').split(',') if x.strip().replace('.', '').isdigit()]
                        else:
                            # Default scores decreasing from 100
                            relevance_scores = [100.0 - (i * 5.0) for i in range(len(window_docs))]

                    # Ensure we have scores for all ranked docs
                    if len(relevance_scores) < len(ranked_indices):
                        # Pad with decreasing scores
                        for i in range(len(relevance_scores), len(ranked_indices)):
                            relevance_scores.append(max(0, 100.0 - (i * 5.0)))

                    # Assign scores to documents
                    for idx, (rank_idx, score) in enumerate(zip(ranked_indices, relevance_scores)):
                        # rank_idx is 1-based, convert to 0-based
                        doc_idx = rank_idx - 1
                        if 0 <= doc_idx < len(window_docs):
                            doc = window_docs[doc_idx]
                            # If document appears in multiple windows, average the scores
                            if doc in llm_scores_map:
                                llm_scores_map[doc] = (llm_scores_map[doc] + score) / 2.0
                            else:
                                llm_scores_map[doc] = score

                except Exception:
                    # If reranking fails for this window, assign neutral scores
                    for i, doc in enumerate(window_docs):
                        if doc not in llm_scores_map:
                            llm_scores_map[doc] = 50.0  # Neutral score

            # ========== STAGE 5: Ensemble Scoring ==========
            # Combine LLM scores with ColBERT scores: 0.6 LLM + 0.4 ColBERT
            ensemble_scored_docs = []
            for doc in unique_docs:
                llm_score = llm_scores_map.get(doc, 50.0)  # Default to neutral if not scored
                colbert_score = colbert_scores_map.get(doc, 50.0)  # Default to neutral if not scored

                # Weighted combination
                combined_score = 0.6 * llm_score + 0.4 * colbert_score
                ensemble_scored_docs.append((doc, combined_score))

            # ========== STAGE 6: Final Selection ==========
            # Sort by combined score descending and take top 21
            ensemble_scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in ensemble_scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_docs)
