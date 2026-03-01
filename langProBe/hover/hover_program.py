import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityAndGapAnalyzer(dspy.Signature):
    """Analyze the claim to extract multiple entity chains and identify information gaps that need to be verified.

    Entity chains are sequences of related entities mentioned in the claim (e.g., person -> organization -> location).
    Information gaps are missing pieces of information needed to verify the claim.
    Generate 2-3 distinct search queries targeting different entity chains or information gaps."""

    claim: str = dspy.InputField(desc="The claim to analyze for entities and information gaps")
    entity_chains: str = dspy.OutputField(desc="2-3 distinct entity chains or key topics extracted from the claim, separated by newlines")
    queries: list[str] = dspy.OutputField(desc="2-3 parallel search queries targeting different entity chains or information gaps (must be 2-3 queries)")


class DocumentRelevanceScorer(dspy.Signature):
    """Score each document's relevance to the claim and identified entity chains.

    Consider:
    - How well the document addresses the claim
    - Coverage of mentioned entity chains
    - Factual information that helps verify the claim

    Return a relevance score from 0-100 for each document."""

    claim: str = dspy.InputField(desc="The original claim being verified")
    entity_chains: str = dspy.InputField(desc="The entity chains extracted from the claim")
    document: str = dspy.InputField(desc="A retrieved document to score")
    relevance_score: int = dspy.OutputField(desc="Relevance score from 0-100 indicating how relevant this document is to verifying the claim")


class ConfidenceEvaluator(dspy.Signature):
    """Assess whether retrieved documents provide sufficient evidence to verify the claim.

    Analyze the coverage of entity chains and facts in the retrieved documents:
    - Identify which entity chains or facts are well-covered
    - Identify which entity chains, bridging entities, or facts are missing or have insufficient coverage
    - Consider indirect references and incomplete entity chain coverage

    Output a confidence score (0-100) indicating sufficiency of evidence and list specific information gaps."""

    claim: str = dspy.InputField(desc="The claim being verified")
    entity_chains: str = dspy.InputField(desc="The entity chains extracted from the claim")
    retrieved_documents: str = dspy.InputField(desc="The documents retrieved in round 1, concatenated")
    confidence_score: int = dspy.OutputField(desc="Confidence score from 0-100 indicating whether documents provide sufficient evidence to verify the claim")
    missing_information: str = dspy.OutputField(desc="Specific information gaps, missing entity chains, or bridging entities that need to be retrieved")


class TargetedQueryGenerator(dspy.Signature):
    """Generate 1-2 highly targeted follow-up queries to address missing information gaps.

    Based on the claim, already-retrieved documents, and identified gaps, generate specific queries that:
    - Target missing bridging entities or entity chains
    - Address indirect references not covered in round 1
    - Fill specific factual gaps identified in the confidence evaluation

    Generate between 1-2 queries maximum."""

    claim: str = dspy.InputField(desc="The claim being verified")
    retrieved_documents: str = dspy.InputField(desc="Documents already retrieved in round 1")
    missing_information: str = dspy.InputField(desc="Specific information gaps that need to be addressed")
    targeted_queries: list[str] = dspy.OutputField(desc="1-2 highly targeted follow-up queries addressing the missing information (must be 1-2 queries)")


class ListwiseDocumentReranker(dspy.Signature):
    """Rerank all retrieved documents by evaluating them together to identify multi-hop relationships and document interdependencies.

    Unlike scoring documents independently, this listwise approach considers:
    - Multi-hop relationships between documents (e.g., bridging entities connecting two facts)
    - Document interdependencies where one document provides context for another
    - Complementary information across documents that jointly support the claim
    - Coverage of all entity chains mentioned in the claim

    Return a ranked list of document indices representing the optimal ordering, where documents that form multi-hop chains
    or have strong interdependencies are ranked higher."""

    claim: str = dspy.InputField(desc="The claim being verified")
    entity_chains: str = dspy.InputField(desc="The entity chains extracted from the claim")
    all_documents: str = dspy.InputField(desc="All retrieved documents concatenated with separators '---DOCUMENT {i}---'")
    ranked_indices: str = dspy.OutputField(desc="Comma-separated list of document indices in ranked order (e.g., '0,5,12,3,8,...') representing the optimal ordering based on multi-hop relationships and relevance")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Parallel retrieval with larger k per query
        self.k_per_query_round1 = 23  # Retrieve 23 documents per query in round 1
        self.k_per_query_round2 = 15  # Retrieve 15 documents per query in round 2
        self.max_final_docs = 21  # Final output constraint
        self.confidence_threshold = 80  # Threshold for triggering round 2

        # Entity and gap analysis module
        self.entity_gap_analyzer = dspy.ChainOfThought(EntityAndGapAnalyzer)

        # Round 1 retrieval module
        self.retrieve_round1 = dspy.Retrieve(k=self.k_per_query_round1)

        # Round 2 retrieval module
        self.retrieve_round2 = dspy.Retrieve(k=self.k_per_query_round2)

        # Confidence evaluator for assessing coverage after round 1
        self.confidence_evaluator = dspy.ChainOfThought(ConfidenceEvaluator)

        # Targeted query generator for round 2
        self.targeted_query_generator = dspy.ChainOfThought(TargetedQueryGenerator)

        # Listwise document reranker (replaces individual scoring)
        self.listwise_reranker = dspy.ChainOfThought(ListwiseDocumentReranker)

    def forward(self, claim):
        # ===== ROUND 1: Initial parallel retrieval =====
        # Step 1: Analyze claim to extract entity chains and generate parallel queries
        analysis = self.entity_gap_analyzer(claim=claim)
        entity_chains = analysis.entity_chains
        queries = analysis.queries

        # Ensure we have 2-3 queries (constraint: max 3 queries per claim)
        if len(queries) < 2:
            # If only 1 query generated, duplicate with slight variation
            queries = [queries[0], claim][:3]
        elif len(queries) > 3:
            # If more than 3, take top 3
            queries = queries[:3]

        # Step 2: Round 1 parallel retrieval - retrieve k documents for each query
        round1_retrieved_docs = []
        for query in queries:
            docs = self.retrieve_round1(query).passages
            round1_retrieved_docs.extend(docs)

        # Deduplicate round 1 documents while preserving order
        seen = set()
        round1_unique_docs = []
        for doc in round1_retrieved_docs:
            if doc not in seen:
                seen.add(doc)
                round1_unique_docs.append(doc)

        # ===== CONFIDENCE EVALUATION =====
        # Step 3: Evaluate confidence and identify information gaps
        # Concatenate round 1 documents for evaluation (limit to reasonable size)
        retrieved_docs_str = "\n\n".join(round1_unique_docs[:50])  # Limit to 50 docs to avoid token limits

        try:
            confidence_eval = self.confidence_evaluator(
                claim=claim,
                entity_chains=entity_chains,
                retrieved_documents=retrieved_docs_str
            )
            confidence_score = confidence_eval.confidence_score

            # Ensure confidence score is an integer
            if isinstance(confidence_score, str):
                import re
                match = re.search(r'\d+', confidence_score)
                confidence_score = int(match.group()) if match else 50
            else:
                confidence_score = int(confidence_score)

            missing_information = confidence_eval.missing_information
        except Exception:
            # If evaluation fails, assume we need round 2 (conservative approach)
            confidence_score = 50
            missing_information = "Unable to evaluate coverage, proceeding with additional retrieval"

        # ===== ROUND 2: Targeted follow-up retrieval (conditional) =====
        all_unique_docs = round1_unique_docs.copy()

        if confidence_score < self.confidence_threshold:
            # Step 4: Generate targeted follow-up queries
            try:
                targeted_query_result = self.targeted_query_generator(
                    claim=claim,
                    retrieved_documents=retrieved_docs_str,
                    missing_information=missing_information
                )
                targeted_queries = targeted_query_result.targeted_queries

                # Ensure we have 1-2 queries (constraint for round 2)
                if len(targeted_queries) < 1:
                    targeted_queries = [claim]
                elif len(targeted_queries) > 2:
                    targeted_queries = targeted_queries[:2]

                # Step 5: Round 2 retrieval with targeted queries
                round2_retrieved_docs = []
                for query in targeted_queries:
                    docs = self.retrieve_round2(query).passages
                    round2_retrieved_docs.extend(docs)

                # Step 6: Deduplicate across both rounds
                for doc in round2_retrieved_docs:
                    if doc not in seen:
                        seen.add(doc)
                        all_unique_docs.append(doc)
            except Exception:
                # If round 2 fails, continue with round 1 results
                pass

        # ===== LISTWISE RERANKING =====
        # Step 7: Apply ListwiseDocumentReranker to all unique documents together
        # If we have fewer than max_final_docs, return all
        if len(all_unique_docs) <= self.max_final_docs:
            return dspy.Prediction(retrieved_docs=all_unique_docs)

        # Concatenate all documents with clear separators
        concatenated_docs = "\n\n".join([
            f"---DOCUMENT {i}---\n{doc}"
            for i, doc in enumerate(all_unique_docs)
        ])

        # Call the listwise reranker
        try:
            rerank_result = self.listwise_reranker(
                claim=claim,
                entity_chains=entity_chains,
                all_documents=concatenated_docs
            )
            ranked_indices_str = rerank_result.ranked_indices

            # Parse the returned indices string (e.g., "0,5,12,3,8,...")
            import re
            # Extract all numbers from the string
            indices = re.findall(r'\d+', ranked_indices_str)
            # Convert to integers and filter valid indices
            valid_indices = [
                int(idx) for idx in indices
                if idx.isdigit() and 0 <= int(idx) < len(all_unique_docs)
            ]

            # Remove duplicates while preserving order
            seen_indices = set()
            unique_indices = []
            for idx in valid_indices:
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    unique_indices.append(idx)

            # Select top max_final_docs documents in the specified order
            top_docs = [all_unique_docs[idx] for idx in unique_indices[:self.max_final_docs]]

            # If we didn't get enough documents from the reranker, fill with remaining docs
            if len(top_docs) < self.max_final_docs:
                remaining_docs = [
                    doc for i, doc in enumerate(all_unique_docs)
                    if i not in seen_indices
                ]
                top_docs.extend(remaining_docs[:self.max_final_docs - len(top_docs)])

        except Exception:
            # If listwise reranking fails, fall back to returning first max_final_docs
            top_docs = all_unique_docs[:self.max_final_docs]

        return dspy.Prediction(retrieved_docs=top_docs)
