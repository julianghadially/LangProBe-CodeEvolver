import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class RequiredDocumentAnalysis(dspy.Signature):
    """Analyze a claim to identify 2-4 specific document titles or topics that MUST be retrieved to verify the claim.
    Focus on concrete entities (people, artworks, events, organizations) rather than broad concepts."""

    claim: str = dspy.InputField(desc="the claim to be verified")
    required_documents: list[dict] = dspy.OutputField(
        desc="a list of 2-4 dictionaries, each with 'title' (specific document title or topic, e.g., 'The Dinner Party artwork', 'Sojourner Truth biography') and 'rationale' (brief explanation why this document is needed)"
    )


class PreciseQueryGeneration(dspy.Signature):
    """Generate a highly specific query optimized for Wikipedia title/abstract matching.
    The query should target the exact document described, using precise terminology."""

    claim: str = dspy.InputField(desc="the original claim being verified")
    document_description: str = dspy.InputField(desc="description of the specific document to retrieve")
    query: str = dspy.OutputField(
        desc="a precise query optimized for Wikipedia title/abstract matching, max 10 words, targeting the specific entity or topic"
    )


class DocumentRelevanceScoring(dspy.Signature):
    """Score how well a retrieved document matches the list of required documents for verifying a claim.
    Consider whether the document title and content align with any of the required document descriptions."""

    claim: str = dspy.InputField(desc="the original claim being verified")
    required_documents_list: str = dspy.InputField(desc="formatted list of required document titles/topics")
    document_title: str = dspy.InputField(desc="the title of the retrieved document to score")
    document_text: str = dspy.InputField(desc="the text content of the retrieved document")
    relevance_score: int = dspy.OutputField(
        desc="relevance score from 0-10, where 10 means perfect match to a required document, 0 means completely irrelevant"
    )


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

        # Phase 1: Analyze claim to identify required documents
        self.document_analyzer = dspy.ChainOfThought(RequiredDocumentAnalysis)

        # Phase 2: Generate precise queries for each required document
        self.query_generator = dspy.Predict(PreciseQueryGeneration)

        # Phase 3: Score and rank retrieved documents
        self.document_scorer = dspy.Predict(DocumentRelevanceScoring)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Phase 1: Identify required documents
            analysis_result = self.document_analyzer(claim=claim)
            required_docs = analysis_result.required_documents

            # Phase 2: Generate precise queries (max 3 queries)
            queries = []
            for i, doc_info in enumerate(required_docs[:3]):  # Limit to max 3 queries
                doc_description = f"{doc_info.get('title', '')} - {doc_info.get('rationale', '')}"
                query_result = self.query_generator(
                    claim=claim,
                    document_description=doc_description
                )
                queries.append(query_result.query)

            # Retrieve k=25 documents per query
            all_retrieved_passages = []
            seen_passages = set()

            for query in queries:
                retrieved = self.rm(query, k=25)
                for passage in retrieved:
                    # Deduplicate based on passage content
                    passage_key = passage.split(" | ")[0] if " | " in passage else passage
                    if passage_key not in seen_passages:
                        seen_passages.add(passage_key)
                        all_retrieved_passages.append(passage)

            # Phase 3: Score and rank documents
            # Format required documents list for scoring
            required_docs_list = "\n".join([
                f"- {doc.get('title', 'N/A')}: {doc.get('rationale', 'N/A')}"
                for doc in required_docs
            ])

            scored_passages = []
            for passage in all_retrieved_passages:
                # Parse passage format: "title | text"
                parts = passage.split(" | ", 1)
                doc_title = parts[0] if len(parts) > 0 else ""
                doc_text = parts[1] if len(parts) > 1 else passage

                # Truncate document text to avoid excessive token usage
                doc_text_truncated = doc_text[:500]

                try:
                    score_result = self.document_scorer(
                        claim=claim,
                        required_documents_list=required_docs_list,
                        document_title=doc_title,
                        document_text=doc_text_truncated
                    )

                    # Extract numeric score
                    score = score_result.relevance_score
                    if isinstance(score, str):
                        # Try to extract number from string
                        import re
                        match = re.search(r'\d+', score)
                        score = int(match.group()) if match else 0
                    else:
                        score = int(score)

                    # Clamp score to 0-10 range
                    score = max(0, min(10, score))
                except (ValueError, AttributeError, TypeError):
                    # If scoring fails, assign a default score
                    score = 5

                scored_passages.append((passage, score))

            # Sort by score (descending) and select top 21
            scored_passages.sort(key=lambda x: x[1], reverse=True)
            top_passages = [passage for passage, score in scored_passages[:21]]

            return dspy.Prediction(retrieved_docs=top_passages)
