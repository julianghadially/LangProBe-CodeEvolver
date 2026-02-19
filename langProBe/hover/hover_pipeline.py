import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class CoverageAnalysisSignature(dspy.Signature):
    """Analyze which aspects of the claim are covered by the retrieved passages and identify what remains uncovered.

    Identify entities, facts, and relationships mentioned in the claim.
    Then determine which aspects are well-supported by the passages and which aspects need more evidence."""

    claim: str = dspy.InputField(desc="The claim to fact-check")
    retrieved_passages: list[str] = dspy.InputField(desc="The passages retrieved so far")
    covered_entities: list[str] = dspy.OutputField(desc="List of claim entities/aspects that are well-covered by the passages")
    uncovered_aspects: str = dspy.OutputField(desc="Description of claim aspects that still need supporting evidence")


class TargetedQuerySignature(dspy.Signature):
    """Generate a targeted search query that focuses on uncovered aspects of the claim."""

    claim: str = dspy.InputField(desc="The original claim")
    uncovered_aspects: str = dspy.InputField(desc="Aspects of the claim that need more evidence")
    query: str = dspy.OutputField(desc="A targeted search query focusing on the uncovered aspects")


class CoverageRerankSignature(dspy.Signature):
    """Score a document by how well it addresses previously uncovered aspects of the claim."""

    claim: str = dspy.InputField(desc="The claim being fact-checked")
    uncovered_aspects: str = dspy.InputField(desc="Aspects that were previously uncovered")
    document: str = dspy.InputField(desc="A retrieved document to score")
    relevance_score: float = dspy.OutputField(desc="Score from 0.0 to 1.0 indicating how well this document addresses uncovered aspects")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Coverage analysis modules
        self.coverage_analyzer = dspy.ChainOfThought(CoverageAnalysisSignature)
        self.targeted_query_gen = dspy.ChainOfThought(TargetedQuerySignature)

        # Retrieval modules with k=10
        self.retrieve_k10 = dspy.Retrieve(k=10)

    def _extract_document_title(self, doc):
        """Extract the title from a document string (format: 'title | content')."""
        return doc.split(" | ")[0] if " | " in doc else doc

    def _deduplicate_by_title(self, documents):
        """Deduplicate documents by exact title match, keeping first occurrence."""
        seen_titles = set()
        unique_docs = []
        for doc in documents:
            title = self._extract_document_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
        return unique_docs

    def _score_document_for_coverage(self, claim, uncovered_aspects, document):
        """Score a document based on how well it addresses uncovered aspects.

        Uses heuristic scoring based on keyword overlap to avoid expensive LM calls."""
        if not uncovered_aspects or uncovered_aspects.strip().lower() in ["none", "all covered", ""]:
            return 0.5  # Neutral score if nothing uncovered

        # Simple heuristic: count how many uncovered aspect keywords appear in the document
        doc_lower = document.lower()
        uncovered_lower = uncovered_aspects.lower()

        # Extract meaningful words from uncovered aspects (filter out common words)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "is", "are", "was", "were", "been", "be", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "may", "might", "must", "can", "this", "that", "these", "those", "it", "its", "they", "them", "their"}

        uncovered_words = [w.strip(".,!?;:") for w in uncovered_lower.split() if len(w) > 3 and w not in stop_words]

        if not uncovered_words:
            return 0.5

        # Count matches
        matches = sum(1 for word in uncovered_words if word in doc_lower)
        score = min(1.0, matches / max(len(uncovered_words), 1))

        return score

    def _rerank_by_coverage(self, claim, uncovered_aspects, documents):
        """Rerank documents by how well they address uncovered aspects."""
        if not documents:
            return []

        # Score each document
        scored_docs = []
        for doc in documents:
            score = self._score_document_for_coverage(claim, uncovered_aspects, doc)
            scored_docs.append((score, doc))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return just the documents
        return [doc for score, doc in scored_docs]

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # HOP 1: Initial retrieval from claim
            hop1_docs = self.retrieve_k10(claim).passages

            # Coverage analysis after hop 1
            coverage1 = self.coverage_analyzer(
                claim=claim,
                retrieved_passages=hop1_docs
            )

            # HOP 2: Targeted retrieval focusing on uncovered aspects
            if coverage1.uncovered_aspects and coverage1.uncovered_aspects.strip():
                hop2_query = self.targeted_query_gen(
                    claim=claim,
                    uncovered_aspects=coverage1.uncovered_aspects
                ).query
            else:
                # Fallback if nothing uncovered (shouldn't happen, but safe)
                hop2_query = claim

            hop2_docs = self.retrieve_k10(hop2_query).passages

            # Coverage analysis after hop 2 (on all 20 documents)
            all_docs_so_far = hop1_docs + hop2_docs
            coverage2 = self.coverage_analyzer(
                claim=claim,
                retrieved_passages=all_docs_so_far
            )

            # HOP 3: Highly specific retrieval for remaining gaps
            if coverage2.uncovered_aspects and coverage2.uncovered_aspects.strip():
                hop3_query = self.targeted_query_gen(
                    claim=claim,
                    uncovered_aspects=coverage2.uncovered_aspects
                ).query
            else:
                # If everything is covered, do a final broad search
                hop3_query = f"{claim} additional evidence"

            hop3_docs = self.retrieve_k10(hop3_query).passages

            # Combine all 30 documents
            all_30_docs = hop1_docs + hop2_docs + hop3_docs

            # Coverage-based reranking: score documents by how well they address uncovered aspects
            # Use the final uncovered aspects from coverage2 for reranking
            reranked_docs = self._rerank_by_coverage(
                claim=claim,
                uncovered_aspects=coverage2.uncovered_aspects,
                documents=all_30_docs
            )

            # Deduplicate by exact title match
            unique_docs = self._deduplicate_by_title(reranked_docs)

            # Return top 21 unique documents
            final_docs = unique_docs[:21]

            return dspy.Prediction(retrieved_docs=final_docs)
