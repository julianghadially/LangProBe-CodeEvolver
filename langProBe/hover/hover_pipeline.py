import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class IdentifyMissingEntities(dspy.Signature):
    """Analyze the claim's entity requirements and identify which key entities or topics are missing from the currently retrieved documents. Focus on entities that would be necessary to verify the claim."""

    claim: str = dspy.InputField(desc="the claim to be verified")
    retrieved_titles: str = dspy.InputField(desc="titles of documents retrieved so far")
    summary: str = dspy.InputField(desc="summary of findings from retrieved documents")
    missing_entities: str = dspy.OutputField(desc="key entities, topics, or concepts that are missing from the retrieved documents but needed to verify the claim")


class GenerateQueryWithGaps(dspy.Signature):
    """Generate a search query that specifically targets missing entities and coverage gaps to find diverse, complementary documents."""

    claim: str = dspy.InputField(desc="the claim to be verified")
    summary: str = dspy.InputField(desc="summary of what has been found so far")
    missing_entities: str = dspy.InputField(desc="entities or topics missing from current retrieval")
    query: str = dspy.OutputField(desc="a search query focusing on the missing entities to find complementary documents")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim with gap-aware query refinement.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.k = 8  # Retrieve more documents per hop, then deduplicate

        # Gap analysis module
        self.identify_gaps = dspy.ChainOfThought(IdentifyMissingEntities)

        # Query generation with gap awareness
        self.create_query_hop2 = dspy.ChainOfThought(GenerateQueryWithGaps)
        self.create_query_hop3 = dspy.ChainOfThought(GenerateQueryWithGaps)

        # Retrieval and summarization
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def _deduplicate_documents(self, docs):
        """Remove duplicate documents by normalized title, preserving order."""
        seen_titles = set()
        unique_docs = []

        for doc in docs:
            # Extract title (format is "title | content")
            title = doc.split(" | ")[0] if " | " in doc else doc
            # Normalize title for comparison
            normalized_title = dspy.evaluate.normalize_text(title)

            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_docs.append(doc)

        return unique_docs

    def _get_document_titles(self, docs):
        """Extract titles from documents."""
        titles = []
        for doc in docs:
            title = doc.split(" | ")[0] if " | " in doc else doc
            titles.append(title)
        return ", ".join(titles)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # HOP 1: Initial retrieval based on claim
            hop1_docs = self.retrieve_k(claim).passages
            summary_1 = self.summarize1(
                claim=claim, passages=hop1_docs
            ).summary

            # Gap analysis after hop 1
            hop1_titles = self._get_document_titles(hop1_docs)
            gap_analysis_1 = self.identify_gaps(
                claim=claim,
                retrieved_titles=hop1_titles,
                summary=summary_1
            ).missing_entities

            # HOP 2: Generate query targeting missing entities
            hop2_query = self.create_query_hop2(
                claim=claim,
                summary=summary_1,
                missing_entities=gap_analysis_1
            ).query
            hop2_docs = self.retrieve_k(hop2_query).passages

            # Combine and deduplicate after hop 2
            all_docs_so_far = self._deduplicate_documents(hop1_docs + hop2_docs)

            summary_2 = self.summarize2(
                claim=claim, context=summary_1, passages=hop2_docs
            ).summary

            # Gap analysis after hop 2
            combined_titles = self._get_document_titles(all_docs_so_far)
            combined_summary = f"{summary_1}\n{summary_2}"
            gap_analysis_2 = self.identify_gaps(
                claim=claim,
                retrieved_titles=combined_titles,
                summary=combined_summary
            ).missing_entities

            # HOP 3: Generate final query targeting remaining gaps
            hop3_query = self.create_query_hop3(
                claim=claim,
                summary=combined_summary,
                missing_entities=gap_analysis_2
            ).query
            hop3_docs = self.retrieve_k(hop3_query).passages

            # Final deduplication across all hops
            all_docs = self._deduplicate_documents(hop1_docs + hop2_docs + hop3_docs)

            # Ensure we return at most 21 documents
            final_docs = all_docs[:21]

            return dspy.Prediction(retrieved_docs=final_docs)
