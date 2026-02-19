import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ClaimDecomposer(dspy.Signature):
    """Decompose a claim into 3 distinct sub-questions to facilitate comprehensive retrieval.
    Generate sub-questions focusing on: (1) main entities, (2) relationships/connections, and (3) attributes/facts."""

    claim: str = dspy.InputField()
    sub_question_1: str = dspy.OutputField(desc="Question focusing on the main entities mentioned in the claim")
    sub_question_2: str = dspy.OutputField(desc="Question focusing on the relationships and connections between entities")
    sub_question_3: str = dspy.OutputField(desc="Question focusing on specific attributes, facts, or properties")


class MissingEntityIdentifier(dspy.Signature):
    """Analyze the claim against retrieved documents to identify specific missing entities by name.
    Focus on concrete named entities (people, places, organizations, events) that are mentioned in the claim but not well-covered in the retrieved documents."""

    claim: str = dspy.InputField()
    retrieved_content: str = dspy.InputField(desc="Summary of documents retrieved so far")
    missing_entities: str = dspy.OutputField(desc="Comma-separated list of specific missing entity names that need targeted retrieval")


class DocumentRelevanceScorer(dspy.Signature):
    """Score each document for relevance to the claim on a scale of 0-100.
    Consider how directly the document supports verifying or refuting the claim."""

    claim: str = dspy.InputField()
    document_title: str = dspy.InputField()
    document_content: str = dspy.InputField()
    relevance_score: int = dspy.OutputField(desc="Relevance score from 0-100, where 100 is most relevant")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using claim decomposition with mandatory deduplication.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Stage 1: Claim Decomposition
        self.decomposer = dspy.Predict(ClaimDecomposer)

        # Stage 2: Gap Filling
        self.gap_identifier = dspy.ChainOfThought(MissingEntityIdentifier)

        # Stage 3: Relevance Scoring
        self.relevance_scorer = dspy.Predict(DocumentRelevanceScorer)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # STAGE 1: Claim Decomposition (1 search with k=50)
            decomposition = self.decomposer(claim=claim)

            # Combine all 3 sub-questions into a single comprehensive query
            combined_query = f"{decomposition.sub_question_1} {decomposition.sub_question_2} {decomposition.sub_question_3}"

            # Single retrieval with k=50
            stage1_docs = dspy.Retrieve(k=50)(combined_query).passages

            # STAGE 2: Targeted Gap Filling (2 searches with k=25 each)
            # Create a summary of retrieved content for gap analysis
            retrieved_titles = [doc.split(" | ")[0] for doc in stage1_docs[:10]]  # Sample first 10 titles
            retrieved_summary = f"Retrieved documents about: {', '.join(retrieved_titles)}"

            # Identify missing entities for first gap-filling search
            gap_analysis_1 = self.gap_identifier(claim=claim, retrieved_content=retrieved_summary)
            stage2_docs_1 = dspy.Retrieve(k=25)(gap_analysis_1.missing_entities).passages

            # Identify missing entities for second gap-filling search
            # Update summary with new documents
            new_titles = [doc.split(" | ")[0] for doc in stage2_docs_1[:5]]
            updated_summary = retrieved_summary + f" Additional: {', '.join(new_titles)}"

            gap_analysis_2 = self.gap_identifier(claim=claim, retrieved_content=updated_summary)
            stage2_docs_2 = dspy.Retrieve(k=25)(gap_analysis_2.missing_entities).passages

            # STAGE 3: Aggressive Deduplication & Reranking
            # Combine all documents from all stages
            all_docs = stage1_docs + stage2_docs_1 + stage2_docs_2

            # Deduplicate by title (keep first occurrence of each unique title)
            seen_titles = set()
            unique_docs = []
            for doc in all_docs:
                # Extract title (before " | ")
                title = doc.split(" | ")[0]
                # Normalize title for comparison
                normalized_title = title.lower().strip()

                if normalized_title not in seen_titles:
                    seen_titles.add(normalized_title)
                    unique_docs.append(doc)

            # Score each unique document for relevance
            scored_docs = []
            for doc in unique_docs:
                # Extract title and content
                parts = doc.split(" | ", 1)
                doc_title = parts[0]
                doc_content = parts[1] if len(parts) > 1 else ""

                # Score the document
                # Use a simpler scoring approach to avoid too many LM calls
                # We'll use a batch approach or limit scoring to top candidates
                scored_docs.append((doc, doc_title, doc_content))

            # For efficiency, score only the first 40 unique documents
            # (since we need 21 and have already deduplicated)
            docs_to_score = scored_docs[:40]

            scored_results = []
            for doc, doc_title, doc_content in docs_to_score:
                try:
                    # Score each document
                    score_result = self.relevance_scorer(
                        claim=claim,
                        document_title=doc_title,
                        document_content=doc_content[:500]  # Limit content length
                    )
                    score = score_result.relevance_score
                    # Ensure score is an integer
                    if isinstance(score, str):
                        # Extract number from string
                        import re
                        match = re.search(r'\d+', score)
                        score = int(match.group()) if match else 50
                    scored_results.append((doc, int(score)))
                except:
                    # If scoring fails, assign a default score
                    scored_results.append((doc, 50))

            # Sort by score (descending) and take top 21
            scored_results.sort(key=lambda x: x[1], reverse=True)
            top_21_docs = [doc for doc, score in scored_results[:21]]

            # If we have fewer than 21, pad with remaining unique docs
            if len(top_21_docs) < 21:
                remaining_docs = [doc for doc, _, _ in scored_docs[40:] if doc not in top_21_docs]
                top_21_docs.extend(remaining_docs[:21 - len(top_21_docs)])

            return dspy.Prediction(retrieved_docs=top_21_docs[:21])
