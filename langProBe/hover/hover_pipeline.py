import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class GapAnalyzerSignature(dspy.Signature):
    """Analyze what information is still missing from the claim that hasn't been covered by retrieved documents."""
    claim: str = dspy.InputField(desc="the claim to verify")
    retrieved_info: str = dspy.InputField(desc="summary of information from documents retrieved so far")
    missing_info: str = dspy.OutputField(desc="what key information is still missing to fully verify the claim")


class UtilityRerankerSignature(dspy.Signature):
    """Score a document based on relevance to claim, coverage of information gaps, and novelty."""
    claim: str = dspy.InputField(desc="the claim to verify")
    document: str = dspy.InputField(desc="the document to score")
    gaps: str = dspy.InputField(desc="information gaps that need to be filled")
    already_covered: str = dspy.InputField(desc="information already covered by selected documents")
    utility_score: float = dspy.OutputField(desc="utility score from 0-10 based on relevance, gap coverage, and novelty")
    reasoning: str = dspy.OutputField(desc="brief explanation of the score")


class NextQuerySignature(dspy.Signature):
    """Generate a search query targeting specific information gaps."""
    claim: str = dspy.InputField(desc="the claim to verify")
    gaps: str = dspy.InputField(desc="information gaps to target")
    previous_summaries: str = dspy.InputField(desc="summaries from previous hops")
    query: str = dspy.OutputField(desc="search query targeting the information gaps")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim with gap-aware retrieval and utility-based reranking.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Retrieval modules - retrieve more documents per hop for reranking
        self.retrieve_k = dspy.Retrieve(k=40)

        # Gap analysis and reranking modules
        self.gap_analyzer = dspy.ChainOfThought(GapAnalyzerSignature)
        self.utility_scorer = dspy.ChainOfThought(UtilityRerankerSignature)
        self.next_query_gen = dspy.ChainOfThought(NextQuerySignature)

        # Summarization modules
        self.summarize = dspy.ChainOfThought("claim,passages->summary")

    def _extract_title(self, passage):
        """Extract document title from passage string (format: 'title | content')."""
        return passage.split(" | ")[0] if " | " in passage else passage.split("\n")[0]

    def _deduplicate_docs(self, new_docs, seen_titles):
        """Remove documents whose titles have already been seen."""
        unique_docs = []
        for doc in new_docs:
            title = self._extract_title(doc)
            normalized_title = dspy.evaluate.normalize_text(title)
            if normalized_title not in seen_titles:
                unique_docs.append(doc)
                seen_titles.add(normalized_title)
        return unique_docs

    def _rerank_by_utility(self, claim, docs, gaps, already_covered, top_k=10):
        """Rerank documents by utility score and return top_k."""
        if not docs:
            return []

        scored_docs = []
        for doc in docs:
            try:
                result = self.utility_scorer(
                    claim=claim,
                    document=doc[:500],  # Truncate for efficiency
                    gaps=gaps,
                    already_covered=already_covered
                )
                # Parse score from output
                try:
                    score = float(result.utility_score)
                except (ValueError, AttributeError):
                    # If parsing fails, try to extract from reasoning
                    score = 5.0  # Default middle score
                scored_docs.append((score, doc))
            except Exception:
                # If scoring fails, give it a middle score
                scored_docs.append((5.0, doc))

        # Sort by score descending and return top_k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            seen_titles = set()
            all_candidate_docs = []
            summaries = []

            # HOP 1: Initial retrieval
            hop1_docs = self.retrieve_k(claim).passages
            hop1_unique = self._deduplicate_docs(hop1_docs, seen_titles)

            # Initial gap analysis - what do we need?
            gaps_1 = "All information needed to verify the claim"

            # Rerank hop1 documents by utility
            hop1_selected = self._rerank_by_utility(
                claim=claim,
                docs=hop1_unique,
                gaps=gaps_1,
                already_covered="None yet",
                top_k=10
            )
            all_candidate_docs.extend(hop1_selected)

            # Summarize hop 1 results
            if hop1_selected:
                summary_1 = self.summarize(claim=claim, passages=hop1_selected).summary
                summaries.append(summary_1)
            else:
                summary_1 = "No relevant documents found in hop 1"
                summaries.append(summary_1)

            # Gap analysis after hop 1
            gap_result_1 = self.gap_analyzer(
                claim=claim,
                retrieved_info=summary_1
            )
            gaps_2 = gap_result_1.missing_info

            # HOP 2: Generate query targeting gaps
            hop2_query = self.next_query_gen(
                claim=claim,
                gaps=gaps_2,
                previous_summaries=summary_1
            ).query

            hop2_docs = self.retrieve_k(hop2_query).passages
            hop2_unique = self._deduplicate_docs(hop2_docs, seen_titles)

            # Rerank hop2 documents by utility
            hop2_selected = self._rerank_by_utility(
                claim=claim,
                docs=hop2_unique,
                gaps=gaps_2,
                already_covered=summary_1,
                top_k=10
            )
            all_candidate_docs.extend(hop2_selected)

            # Summarize hop 2 results
            if hop2_selected:
                summary_2 = self.summarize(
                    claim=claim,
                    passages=hop2_selected
                ).summary
                summaries.append(summary_2)
            else:
                summary_2 = "No new relevant documents found in hop 2"
                summaries.append(summary_2)

            # Gap analysis after hop 2
            combined_summary = " | ".join(summaries)
            gap_result_2 = self.gap_analyzer(
                claim=claim,
                retrieved_info=combined_summary
            )
            gaps_3 = gap_result_2.missing_info

            # HOP 3: Generate query targeting remaining gaps
            hop3_query = self.next_query_gen(
                claim=claim,
                gaps=gaps_3,
                previous_summaries=combined_summary
            ).query

            hop3_docs = self.retrieve_k(hop3_query).passages
            hop3_unique = self._deduplicate_docs(hop3_docs, seen_titles)

            # Rerank hop3 documents by utility
            hop3_selected = self._rerank_by_utility(
                claim=claim,
                docs=hop3_unique,
                gaps=gaps_3,
                already_covered=combined_summary,
                top_k=10
            )
            all_candidate_docs.extend(hop3_selected)

            # FINAL RERANKING: Select top 21 from all candidates
            # Build final summary of all selected docs
            final_summary = " | ".join(summaries)
            if hop3_selected:
                summary_3 = self.summarize(claim=claim, passages=hop3_selected).summary
                final_summary = final_summary + " | " + summary_3

            # Final utility-based reranking across all candidates
            final_docs = self._rerank_by_utility(
                claim=claim,
                docs=all_candidate_docs,
                gaps="Complete coverage of all aspects of the claim",
                already_covered=final_summary,
                top_k=21
            )

            return dspy.Prediction(retrieved_docs=final_docs)
