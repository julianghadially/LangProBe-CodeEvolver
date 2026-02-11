import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7

        # Entity extraction module
        self.extract_entities = dspy.ChainOfThought("claim -> entities: list[str]")

        # Coverage tracking module
        self.track_coverage = dspy.Predict(
            "claim, entities: list[str], retrieved_titles: list[str] -> "
            "covered_entities: list[str], missing_entities: list[str]"
        )

        # Query generation modules with reasoning and missing entities
        self.create_query_hop2 = dspy.ChainOfThought(
            "claim, summary, missing_entities: list[str] -> reasoning, query"
        )
        self.create_query_hop3 = dspy.ChainOfThought(
            "claim, summary, missing_entities: list[str] -> reasoning, query"
        )

        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")

    def _extract_titles_from_passages(self, passages):
        """Extract titles from passage strings.

        Passages typically come in format 'Title | Content' or similar.
        This extracts the title portion before the first separator.
        """
        titles = []
        for passage in passages:
            # Try common separators
            if " | " in passage:
                title = passage.split(" | ")[0]
            elif "\n" in passage:
                title = passage.split("\n")[0]
            else:
                # Use first 100 chars as approximate title
                title = passage[:100]
            titles.append(title.strip())
        return titles

    def _deduplicate_by_title(self, all_passages):
        """Deduplicate passages by their titles to prevent retrieving same doc multiple times."""
        seen_titles = set()
        deduplicated = []

        for passage in all_passages:
            # Extract title
            if " | " in passage:
                title = passage.split(" | ")[0].strip()
            elif "\n" in passage:
                title = passage.split("\n")[0].strip()
            else:
                title = passage[:100].strip()

            # Only add if we haven't seen this title before
            if title not in seen_titles:
                seen_titles.add(title)
                deduplicated.append(passage)

        return deduplicated

    def forward(self, claim):
        # Extract all key entities that need Wikipedia articles
        entities_result = self.extract_entities(claim=claim)
        all_entities = entities_result.entities

        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        hop1_titles = self._extract_titles_from_passages(hop1_docs)

        # Track coverage after hop 1
        coverage1 = self.track_coverage(
            claim=claim,
            entities=all_entities,
            retrieved_titles=hop1_titles
        )

        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2 - Generate query targeting missing entities
        hop2_result = self.create_query_hop2(
            claim=claim,
            summary=summary_1,
            missing_entities=coverage1.missing_entities
        )
        hop2_query = hop2_result.query
        hop2_docs = self.retrieve_k(hop2_query).passages
        hop2_titles = self._extract_titles_from_passages(hop2_docs)

        # Track coverage after hop 2
        all_titles_so_far = hop1_titles + hop2_titles
        coverage2 = self.track_coverage(
            claim=claim,
            entities=all_entities,
            retrieved_titles=all_titles_so_far
        )

        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3 - Generate query targeting remaining missing entities
        hop3_result = self.create_query_hop3(
            claim=claim,
            summary=summary_2,
            missing_entities=coverage2.missing_entities
        )
        hop3_query = hop3_result.query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Deduplicate all retrieved documents by title
        all_docs = hop1_docs + hop2_docs + hop3_docs
        deduplicated_docs = self._deduplicate_by_title(all_docs)

        return dspy.Prediction(retrieved_docs=deduplicated_docs)


