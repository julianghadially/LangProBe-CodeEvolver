import dspy
from difflib import SequenceMatcher
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityTitleExtractor(dspy.Signature):
    """Extract potential Wikipedia article titles from a claim. Focus on proper nouns, full names,
    specific entities, organizations, locations, events, and other named entities that would likely
    have their own Wikipedia articles."""

    claim: str = dspy.InputField(desc="The claim to extract entity titles from")
    entity_titles: list[str] = dspy.OutputField(desc="List of potential Wikipedia article titles extracted from the claim (proper nouns, full names, specific entities)")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()
        self.entity_extractor = dspy.Predict(EntityTitleExtractor)
        self.retrieve_k50 = dspy.Retrieve(k=50)

    def _extract_doc_title(self, doc):
        """Extract the title from a document string (format: 'title | content')"""
        return doc.split(" | ")[0] if " | " in doc else doc

    def _fuzzy_match_score(self, str1, str2):
        """Calculate fuzzy matching score between two strings (0-100)"""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio() * 100

    def _deduplicate_by_title(self, docs):
        """Deduplicate documents by their titles"""
        seen_titles = set()
        unique_docs = []
        for doc in docs:
            title = self._extract_doc_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
        return unique_docs

    def _rerank_by_entity_match(self, docs, entity_titles):
        """Rerank documents prioritizing exact/fuzzy matches with extracted entities"""
        scored_docs = []

        for doc in docs:
            doc_title = self._extract_doc_title(doc)
            max_match_score = 0

            # Check fuzzy match against all extracted entity titles
            for entity in entity_titles:
                match_score = self._fuzzy_match_score(doc_title, entity)
                max_match_score = max(max_match_score, match_score)

            # Score: (entity_match_score >= 85) gets priority boost
            # High entity match scores are prioritized over ColBERT relevance
            if max_match_score >= 85:
                priority_score = 1000 + max_match_score  # High priority
            else:
                priority_score = max_match_score  # Lower priority

            scored_docs.append((priority_score, doc))

        # Sort by priority score (descending)
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs]

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Extract entity titles from the claim using LM
            extraction_result = self.entity_extractor(claim=claim)
            entity_titles = extraction_result.entity_titles if hasattr(extraction_result, 'entity_titles') else []

            # Ensure entity_titles is a list
            if not isinstance(entity_titles, list):
                entity_titles = [entity_titles] if entity_titles else []

            # Search 1: Original claim query (k=50)
            search1_docs = self.retrieve_k50(claim).passages
            search1_titles = [self._extract_doc_title(doc) for doc in search1_docs]

            # Search 2: Reformulated query with extracted entity titles in quotes
            if entity_titles:
                quoted_entities = [f'"{entity}"' for entity in entity_titles]
                reformulated_query = " OR ".join(quoted_entities)
                search2_docs = self.retrieve_k50(reformulated_query).passages
            else:
                search2_docs = []

            # Search 3: Gap-focused query
            search_titles_combined = list(set(search1_titles + [self._extract_doc_title(doc) for doc in search2_docs]))
            if search_titles_combined:
                gap_query = f"Given the claim '{claim}' and these potentially relevant entities: {', '.join(search_titles_combined[:10])}, what additional entities or connections are needed to verify this claim?"
            else:
                gap_query = f"What additional entities or connections are needed to verify this claim: {claim}?"
            search3_docs = self.retrieve_k50(gap_query).passages

            # Combine all search results
            all_docs = search1_docs + search2_docs + search3_docs

            # Deduplicate by document title
            unique_docs = self._deduplicate_by_title(all_docs)

            # Rerank with entity-title matching priority
            reranked_docs = self._rerank_by_entity_match(unique_docs, entity_titles)

            # Select top 21 documents
            top_21_docs = reranked_docs[:21]

            return dspy.Prediction(retrieved_docs=top_21_docs)
