import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from difflib import SequenceMatcher


class ChainOfThoughtQueryPlanner(dspy.Signature):
    """Analyze the claim and retrieved context to strategically plan the next retrieval query.

    Decompose what entities, relationships, and facts are needed to verify the claim.
    Analyze what information has already been found versus what's still missing.
    Generate a targeted query to find the specific missing information needed for the next hop.
    """

    claim = dspy.InputField(desc="The claim that needs to be verified through multi-hop reasoning")
    retrieved_context = dspy.InputField(desc="The context retrieved so far from previous hops (may be empty for first hop)")

    reasoning = dspy.OutputField(desc="Explain the multi-hop reasoning chain needed: what entities/relationships are mentioned in the claim and how they connect")
    missing_information = dspy.OutputField(desc="Identify specific gaps: what key information was found in retrieved_context vs. what's still needed to verify the claim")
    next_query = dspy.OutputField(desc="A focused search query to find the specific missing information identified above")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 15
        self.query_planner_hop1 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.query_planner_hop2 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.query_planner_hop3 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.retrieve_k = dspy.Retrieve(k=15)

    def _extract_title(self, doc):
        """Extract title from document passage (assumes format 'Title | Text' or similar)."""
        if isinstance(doc, str):
            # Try to extract title from common formats
            if ' | ' in doc:
                return doc.split(' | ')[0].strip()
            # If no delimiter, use first 50 chars as identifier
            return doc[:50].strip()
        return str(doc)[:50].strip()

    def _deduplicate_docs(self, all_docs):
        """Remove duplicate documents by title using fuzzy matching."""
        seen_titles = {}
        unique_docs = []

        for doc in all_docs:
            title = self._extract_title(doc)

            # Check for fuzzy matches with existing titles
            is_duplicate = False
            for seen_title in seen_titles.keys():
                similarity = SequenceMatcher(None, title.lower(), seen_title.lower()).ratio()
                if similarity > 0.85:  # 85% similarity threshold
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_titles[title] = True
                unique_docs.append(doc)

        return unique_docs

    def _score_document(self, doc, claim):
        """Score document relevance based on keyword matches with the claim."""
        doc_text = str(doc).lower()
        claim_text = claim.lower()

        # Extract keywords from claim (words longer than 3 chars, excluding common words)
        stop_words = {'the', 'and', 'that', 'this', 'with', 'from', 'have', 'been', 'were', 'was'}
        claim_words = [word.strip('.,!?;:()[]{}"\'"') for word in claim_text.split()]
        keywords = [word for word in claim_words if len(word) > 3 and word not in stop_words]

        score = 0.0

        # Exact keyword matches
        for keyword in keywords:
            if keyword in doc_text:
                score += 1.0

        # Fuzzy matches for longer keywords
        for keyword in keywords:
            if len(keyword) > 6:
                for word in doc_text.split():
                    word_clean = word.strip('.,!?;:()[]{}"\'"')
                    if len(word_clean) > 6:
                        similarity = SequenceMatcher(None, keyword, word_clean).ratio()
                        if similarity > 0.8:
                            score += 0.5

        # Bonus for multiple keyword co-occurrence (indicates relevance)
        keywords_found = sum(1 for kw in keywords if kw in doc_text)
        if keywords_found > 1:
            score += keywords_found * 0.3

        return score

    def _rerank_documents(self, docs, claim, top_k=21):
        """Rerank documents by relevance and return top_k."""
        scored_docs = [(doc, self._score_document(doc, claim)) for doc in docs]
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        # Return top_k documents
        return [doc for doc, score in scored_docs[:top_k]]

    def forward(self, claim):
        # HOP 1: Initial analysis and query generation
        hop1_plan = self.query_planner_hop1(
            claim=claim,
            retrieved_context=""
        )
        hop1_query = hop1_plan.next_query
        hop1_docs = self.retrieve_k(hop1_query).passages
        hop1_context = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(hop1_docs)])

        # HOP 2: Reason about what was found and what's missing
        hop2_plan = self.query_planner_hop2(
            claim=claim,
            retrieved_context=hop1_context
        )
        hop2_query = hop2_plan.next_query
        hop2_docs = self.retrieve_k(hop2_query).passages
        hop2_context = hop1_context + "\n\n" + "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(hop2_docs)])

        # HOP 3: Final targeted retrieval for remaining gaps
        hop3_plan = self.query_planner_hop3(
            claim=claim,
            retrieved_context=hop2_context
        )
        hop3_query = hop3_plan.next_query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Collect all documents from three hops
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Deduplicate documents by title
        unique_docs = self._deduplicate_docs(all_docs)

        # Rerank by relevance and select top 21
        top_docs = self._rerank_documents(unique_docs, claim, top_k=21)

        return dspy.Prediction(retrieved_docs=top_docs)


