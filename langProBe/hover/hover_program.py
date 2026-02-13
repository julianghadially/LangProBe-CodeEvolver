import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class FirstHopPlanner(dspy.Signature):
    """Analyze the claim to determine what information is needed first to begin verification.
    Identify the primary entity or concept that requires investigation and generate a focused search query."""

    claim = dspy.InputField()
    reasoning = dspy.OutputField(desc="Analysis of what information is needed first and why this is the logical starting point")
    search_query = dspy.OutputField(desc="Focused search query to retrieve the most relevant initial documents")


class NextHopPlanner(dspy.Signature):
    """Analyze retrieved documents and the claim to identify critical information gaps.
    Determine what additional information is needed and generate the next search query to fill those gaps.
    Look for implicit entities mentioned in documents that require further investigation."""

    claim = dspy.InputField()
    previous_queries = dspy.InputField(desc="List of search queries executed so far")
    retrieved_titles = dspy.InputField(desc="Titles of documents retrieved so far")
    key_facts_found = dspy.InputField(desc="Summary of key information discovered in retrieved documents")

    information_gap = dspy.OutputField(desc="What critical information is still missing for verification")
    reasoning = dspy.OutputField(desc="Analysis of why this information gap needs to be filled and what implicit entities were discovered")
    search_query = dspy.OutputField(desc="Targeted search query to retrieve documents that fill the identified gap")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7  # 7 docs per hop = 21 total docs

        # First hop planning with reasoning
        self.plan_first_hop = dspy.ChainOfThought(FirstHopPlanner)

        # Next hop planning with reasoning
        self.plan_next_hop = dspy.ChainOfThought(NextHopPlanner)

        # Retriever
        self.retrieve_k = dspy.Retrieve(k=self.k)

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

    def _summarize_documents(self, docs):
        """Extract key facts from retrieved documents.
        Returns a concise summary of important information found in the documents.
        Uses titles and first few lines to capture key facts efficiently."""

        if not docs:
            return "No documents retrieved yet"

        # Extract title and first line of content for each doc
        summaries = []
        for doc in docs[:self.k]:  # Only summarize most recent hop (7 docs)
            lines = doc.split("\n")
            title = lines[0] if lines else doc[:100]
            # Get first line of content (second line after title)
            content_preview = lines[1][:200] if len(lines) > 1 else ""
            summaries.append(f"{title}: {content_preview}")

        return "\n".join(summaries)

    def forward(self, claim):
        # Step 1: Plan the first hop
        first_hop_result = self.plan_first_hop(claim=claim)

        # Initialize tracking
        all_docs = []
        retrieved_titles_set = set()
        previous_queries = []

        # Step 2: Execute first hop
        query = first_hop_result.search_query
        previous_queries.append(query)
        hop_docs = self.retrieve_k(query).passages

        # Track titles and accumulate documents
        for doc in hop_docs:
            retrieved_titles_set.add(self._extract_title(doc))
        all_docs.extend(hop_docs)

        # Step 3: Execute hops 2 and 3 with dynamic planning
        for hop_num in [2, 3]:
            # Summarize what we've learned so far
            key_facts = self._summarize_documents(hop_docs)

            # Plan the next hop based on accumulated knowledge
            next_hop_result = self.plan_next_hop(
                claim=claim,
                previous_queries=", ".join(previous_queries),
                retrieved_titles=", ".join(sorted(retrieved_titles_set)),
                key_facts_found=key_facts
            )

            # Execute the next hop
            query = next_hop_result.search_query
            previous_queries.append(query)
            hop_docs = self.retrieve_k(query).passages

            # Track titles and accumulate documents
            for doc in hop_docs:
                retrieved_titles_set.add(self._extract_title(doc))
            all_docs.extend(hop_docs)

        # Step 4: Deduplicate by document title while preserving order
        deduplicated_docs = self._deduplicate_by_title(all_docs)

        return dspy.Prediction(retrieved_docs=deduplicated_docs)


