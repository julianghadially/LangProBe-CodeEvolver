import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class GenerateEntityQuery(dspy.Signature):
    """Generate an entity-focused search query from the claim that targets key entities, people, places, or organizations mentioned."""
    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="an entity-focused search query")


class GenerateRelationshipQuery(dspy.Signature):
    """Generate a relationship-focused search query from the claim that targets connections, events, or relationships between entities."""
    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a relationship-focused search query")


class AnalyzeRetrievalGaps(dspy.Signature):
    """Analyze the initial retrieved documents to identify coverage gaps. Determine which key entities, facts, or relationships from the claim are well-covered by the documents, which are missing or poorly covered, and generate a targeted query to fill the most critical gaps."""
    claim: str = dspy.InputField()
    documents: str = dspy.InputField(desc="initial retrieved documents")
    well_covered: str = dspy.OutputField(desc="key entities/facts from the claim that are well-covered by the documents")
    missing_or_poor: str = dspy.OutputField(desc="key entities/facts from the claim that are missing or poorly covered")
    gap_filling_query: str = dspy.OutputField(desc="a specific search query targeting the missing information")


class ScoreDocumentRelevance(dspy.Signature):
    """Score a single document's relevance to the claim on a scale of 0-10, where 10 is highly relevant and 0 is irrelevant. Consider how well the document covers key entities, facts, and relationships mentioned in the claim."""
    claim: str = dspy.InputField()
    document: str = dspy.InputField(desc="a single document to score")
    score: int = dspy.OutputField(desc="relevance score from 0-10")
    justification: str = dspy.OutputField(desc="brief explanation for the score")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.retrieve_25 = dspy.Retrieve(k=25)
        self.retrieve_10 = dspy.Retrieve(k=10)
        self.entity_query_gen = dspy.Predict(GenerateEntityQuery)
        self.relationship_query_gen = dspy.Predict(GenerateRelationshipQuery)
        self.gap_analyzer = dspy.ChainOfThought(AnalyzeRetrievalGaps)
        self.doc_scorer = dspy.Predict(ScoreDocumentRelevance)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Stage 1: Initial retrieval with diverse queries
            # Generate entity-focused query
            entity_query = self.entity_query_gen(claim=claim).query
            entity_docs = self.retrieve_25(entity_query).passages

            # Generate relationship-focused query
            relationship_query = self.relationship_query_gen(claim=claim).query
            relationship_docs = self.retrieve_25(relationship_query).passages

            # Combine initial 50 documents
            initial_docs = entity_docs + relationship_docs

            # Stage 2: Gap analysis
            # Format documents for gap analysis
            doc_list_str = "\n".join([f"{i}. {doc}" for i, doc in enumerate(initial_docs)])

            # Analyze coverage gaps with reasoning
            gap_analysis = self.gap_analyzer(claim=claim, documents=doc_list_str)
            gap_filling_query = gap_analysis.gap_filling_query

            # Stage 3: Targeted retrieval to fill gaps
            gap_docs = self.retrieve_10(gap_filling_query).passages

            # Combine all 60 documents
            all_docs = initial_docs + gap_docs

            # Stage 4: Deterministic deduplication by normalized title
            def normalize_title(doc):
                """Extract and normalize document title."""
                # Documents are in format "title | content"
                title = doc.split(" | ")[0] if " | " in doc else doc
                # Normalize: lowercase and strip whitespace
                return title.lower().strip()

            # Deduplicate by normalized title
            seen_titles = set()
            unique_docs = []
            for doc in all_docs:
                norm_title = normalize_title(doc)
                if norm_title not in seen_titles:
                    seen_titles.add(norm_title)
                    unique_docs.append(doc)

            # Stage 5: Hybrid scoring approach
            # Score all unique documents
            scored_docs = []
            for doc in unique_docs:
                try:
                    score_result = self.doc_scorer(claim=claim, document=doc)
                    score = score_result.score
                    # Ensure score is an integer between 0 and 10
                    if isinstance(score, str):
                        score = int(score)
                    score = max(0, min(10, score))
                    scored_docs.append((score, doc))
                except (ValueError, AttributeError):
                    # If scoring fails, assign a default middle score
                    scored_docs.append((5, doc))

            # Sort by score (descending) and select top 21
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_21_docs = [doc for score, doc in scored_docs[:21]]

            # Ensure we have exactly 21 documents
            # If we have fewer, pad with remaining unique docs
            if len(top_21_docs) < 21 and len(unique_docs) > len(top_21_docs):
                remaining_docs = [doc for score, doc in scored_docs[21:]]
                top_21_docs.extend(remaining_docs[:21 - len(top_21_docs)])

            return dspy.Prediction(retrieved_docs=top_21_docs)
