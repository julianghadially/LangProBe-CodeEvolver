import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop
from difflib import SequenceMatcher

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class WikipediaTitleExtractor(dspy.Signature):
    """Extract potential Wikipedia article titles from a claim. Focus on proper nouns, specific entities, works, institutions, and other named entities that would have Wikipedia articles."""

    claim: str = dspy.InputField(desc="the claim to analyze")
    entity_titles: list[str] = dspy.OutputField(desc="list of potential Wikipedia article titles extracted from the claim (proper nouns, entities, works, institutions)")


class EntitySearchQuery(dspy.Signature):
    """Generate a search query to find documents about specific entities."""

    claim: str = dspy.InputField(desc="the original claim")
    entities: str = dspy.InputField(desc="extracted entity titles")
    query: str = dspy.OutputField(desc="search query targeting specific entities")


class RelationshipQuery(dspy.Signature):
    """Generate a search query to find documents about relationships between entities."""

    claim: str = dspy.InputField(desc="the original claim")
    entities: str = dspy.InputField(desc="extracted entity titles")
    query: str = dspy.OutputField(desc="search query focusing on relationships and connections between the entities")


class AttributeQuery(dspy.Signature):
    """Generate a search query to find documents about attributes and properties of entities."""

    claim: str = dspy.InputField(desc="the original claim")
    entities: str = dspy.InputField(desc="extracted entity titles")
    query: str = dspy.OutputField(desc="search query focusing on descriptive facts, attributes, and properties")


class RelevanceScorer(dspy.Signature):
    """Score the relevance of a document to a claim for fact-checking."""

    claim: str = dspy.InputField(desc="the claim being fact-checked")
    document_title: str = dspy.InputField(desc="the title of the document")
    document_text: str = dspy.InputField(desc="the text content of the document")
    relevance_score: float = dspy.OutputField(desc="relevance score from 0.0 to 1.0, where 1.0 is highly relevant")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

        # Initialize new components for entity-based retrieval
        self.title_extractor = dspy.Predict(WikipediaTitleExtractor)
        self.entity_query_gen = dspy.Predict(EntitySearchQuery)
        self.relationship_query_gen = dspy.Predict(RelationshipQuery)
        self.attribute_query_gen = dspy.Predict(AttributeQuery)
        self.relevance_scorer = dspy.Predict(RelevanceScorer)

        self.k_per_query = 50
        self.fuzzy_threshold = 0.85
        self.top_candidates = 50
        self.final_k = 21

    def _fuzzy_match_score(self, str1: str, str2: str) -> float:
        """Compute fuzzy string matching score between two strings."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _extract_title(self, passage: str) -> str:
        """Extract the title from a passage string (format: 'Title | content')."""
        return passage.split(" | ")[0] if " | " in passage else passage

    def _boost_entity_matches(self, documents: list[str], entity_titles: list[str]) -> list[tuple[str, float]]:
        """Boost documents whose titles match extracted entities using fuzzy matching."""
        scored_docs = []

        for doc in documents:
            doc_title = self._extract_title(doc)
            max_score = 0.0

            # Compare with each extracted entity
            for entity in entity_titles:
                match_score = self._fuzzy_match_score(doc_title, entity)
                max_score = max(max_score, match_score)

            # Boost score if fuzzy match exceeds threshold
            boost = 1.0
            if max_score >= self.fuzzy_threshold:
                boost = 2.0  # Strong boost for matching entities
            elif max_score >= 0.7:
                boost = 1.5  # Moderate boost for partial matches

            scored_docs.append((doc, max_score * boost))

        return scored_docs

    def _llm_rerank(self, claim: str, documents: list[str]) -> list[str]:
        """Apply LLM-based relevance scoring to rerank documents."""
        scored_docs = []

        for doc in documents:
            doc_title = self._extract_title(doc)
            doc_text = doc.split(" | ", 1)[1] if " | " in doc else doc

            try:
                result = self.relevance_scorer(
                    claim=claim,
                    document_title=doc_title,
                    document_text=doc_text[:500]  # Limit text length for efficiency
                )
                score = float(result.relevance_score) if hasattr(result, 'relevance_score') else 0.5
            except:
                score = 0.5  # Default score if scoring fails

            scored_docs.append((doc, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_docs]

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Extract entity titles from the claim
            extraction_result = self.title_extractor(claim=claim)
            entity_titles = extraction_result.entity_titles if hasattr(extraction_result, 'entity_titles') else []

            # Convert to string for query generation
            entities_str = ", ".join(entity_titles) if entity_titles else claim

            # Step 2: Generate three targeted queries
            query1_result = self.entity_query_gen(claim=claim, entities=entities_str)
            query1 = query1_result.query if hasattr(query1_result, 'query') else claim

            query2_result = self.relationship_query_gen(claim=claim, entities=entities_str)
            query2 = query2_result.query if hasattr(query2_result, 'query') else claim

            query3_result = self.attribute_query_gen(claim=claim, entities=entities_str)
            query3 = query3_result.query if hasattr(query3_result, 'query') else claim

            # Step 3: Retrieve k=50 documents per query
            retrieve_k50 = dspy.Retrieve(k=self.k_per_query)

            docs1 = retrieve_k50(query1).passages
            docs2 = retrieve_k50(query2).passages
            docs3 = retrieve_k50(query3).passages

            # Combine and deduplicate documents
            all_docs = []
            seen = set()
            for doc in docs1 + docs2 + docs3:
                if doc not in seen:
                    all_docs.append(doc)
                    seen.add(doc)

            # Step 4: Two-stage reranking

            # Stage 1: Boost documents with entity title matches
            scored_docs = self._boost_entity_matches(all_docs, entity_titles)

            # Sort by boosted scores and take top candidates
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [doc for doc, _ in scored_docs[:self.top_candidates]]

            # Stage 2: LLM-based relevance scoring on top candidates
            final_ranked_docs = self._llm_rerank(claim, top_candidates)

            # Return top 21 documents
            final_docs = final_ranked_docs[:self.final_k]

            return dspy.Prediction(retrieved_docs=final_docs)
