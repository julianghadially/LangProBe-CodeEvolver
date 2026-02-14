import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 9
        self.create_query_hop2 = dspy.Predict("claim,retrieved_titles->query")
        self.create_query_hop3 = dspy.Predict("claim,retrieved_titles_hop1,retrieved_titles_hop2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _extract_titles(self, passages):
        """Extract document titles from passages by splitting on first '|' character."""
        titles = []
        for passage in passages:
            # Split on first '|' and take the title portion
            if '|' in passage:
                title = passage.split('|', 1)[0].strip()
                titles.append(title)
            else:
                # If no '|', use the first part of the passage as title
                titles.append(passage.split()[0] if passage else "")
        return titles

    def _deduplicate_by_title(self, all_docs, target_count=21):
        """Deduplicate documents by title to ensure exactly target_count unique documents."""
        seen_titles = set()
        unique_docs = []

        for doc in all_docs:
            # Extract title for deduplication check
            if '|' in doc:
                title = doc.split('|', 1)[0].strip()
            else:
                title = doc.split()[0] if doc else ""

            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)

                if len(unique_docs) == target_count:
                    break

        return unique_docs

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        hop1_titles = self._extract_titles(hop1_docs)
        retrieved_titles_hop1 = ", ".join(hop1_titles)

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, retrieved_titles=retrieved_titles_hop1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        hop2_titles = self._extract_titles(hop2_docs)
        retrieved_titles_hop2 = ", ".join(hop2_titles)

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim,
            retrieved_titles_hop1=retrieved_titles_hop1,
            retrieved_titles_hop2=retrieved_titles_hop2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all documents and deduplicate by title to get exactly 21 unique documents
        all_docs = hop1_docs + hop2_docs + hop3_docs
        unique_docs = self._deduplicate_by_title(all_docs, target_count=21)

        return dspy.Prediction(retrieved_docs=unique_docs)


