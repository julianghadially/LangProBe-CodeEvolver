import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


# ============ New DSPy Signature Classes for Query Decomposition Architecture ============

class ClaimDecomposition(dspy.Signature):
    """Decompose a complex multi-hop claim into 2-3 specific sub-questions that can be answered independently.
    Each sub-question should target a distinct piece of information needed to verify the claim."""

    claim: str = dspy.InputField(desc="the complex claim to verify")
    sub_questions: list[str] = dspy.OutputField(desc="2-3 specific sub-questions that break down the claim into answerable components")


class EntityExtractor(dspy.Signature):
    """Extract key entities and relationships from retrieved documents that are relevant to verifying the claim.
    Focus on concrete entities (people, places, organizations, dates, events) and their relationships."""

    claim: str = dspy.InputField(desc="the claim being verified")
    documents: str = dspy.InputField(desc="the retrieved documents to analyze")
    entities: list[str] = dspy.OutputField(desc="list of key entities discovered (e.g., 'Person: John Doe', 'Movie: The River Rat', 'Date: 1984')")
    relationships: list[str] = dspy.OutputField(desc="list of relationships between entities (e.g., 'John Doe directed The River Rat', 'The River Rat was released in 1984')")


class GapAnalysis(dspy.Signature):
    """Analyze what critical information is still missing to verify the claim, given what has been discovered so far.
    Identify specific entities, relationships, or facts that need to be retrieved in the next iteration."""

    claim: str = dspy.InputField(desc="the claim being verified")
    entities_found: str = dspy.InputField(desc="entities discovered so far")
    relationships_found: str = dspy.InputField(desc="relationships discovered so far")
    documents_retrieved: str = dspy.InputField(desc="summary of documents retrieved so far")
    missing_information: list[str] = dspy.OutputField(desc="specific pieces of missing information needed to verify the claim")
    targeted_queries: list[str] = dspy.OutputField(desc="2-3 specific search queries to find the missing information")


class EntityTitleExtractor(dspy.Signature):
    """Extract potential Wikipedia article titles (proper nouns) from the claim and retrieved context.
    Focus on specific entities that would have their own Wikipedia articles: people, places, events, organizations, works (books, films, albums, etc.).
    Extract the exact names as they would appear in Wikipedia article titles (e.g., 'Pat Ashton', 'SM Megamall', 'Planes Trains and Automobiles')."""

    claim: str = dspy.InputField(desc="the claim being verified")
    context: str = dspy.InputField(desc="retrieved documents providing context")
    entity_titles: list[str] = dspy.OutputField(desc="list of 3-5 potential Wikipedia article titles (proper nouns: people, places, events, works, organizations)")


class AttributeExtractor(dspy.Signature):
    """Identify specific attributes and qualifiers from the claim and retrieved context that can enhance search queries.
    Focus on three types: (1) temporal qualifiers (years, dates, decades, seasons), (2) relational qualifiers (films, albums, TV shows, books, songs), (3) other specific attributes (locations, roles, awards, events).
    Extract entities along with their associated attributes for creating precise attribute-enhanced queries."""

    claim: str = dspy.InputField(desc="the claim being verified")
    context: str = dspy.InputField(desc="retrieved documents providing context")
    temporal_attributes: list[str] = dspy.OutputField(desc="list of temporal qualifiers found (e.g., '1995', '1974 season', '2010s', 'summer 2005')")
    relational_attributes: list[str] = dspy.OutputField(desc="list of relational qualifiers found (e.g., 'debut film', 'Josie Pussycats movie', 'album release', 'TV series')")
    entity_attribute_pairs: list[str] = dspy.OutputField(desc="list of 3-4 entity-attribute combinations for enhanced queries (e.g., 'Rosario Dawson 1995 film debut', 'New York Islanders 1974 season', 'Rachael Leigh Cook Josie Pussycats')")


class DocumentRelevanceSignature(dspy.Signature):
    """Evaluate the relevance of a document to a claim. Score from 1-10 where 10 is highly relevant and provides critical evidence, and 1 is completely irrelevant."""

    claim: str = dspy.InputField(desc="the claim to verify")
    document: str = dspy.InputField(desc="the document to evaluate")
    reasoning: str = dspy.OutputField(desc="explanation of why this document is relevant or not relevant to the claim")
    score: int = dspy.OutputField(desc="relevance score from 1 (irrelevant) to 10 (highly relevant)")


class DocumentRelevanceScorer(dspy.Module):
    """Module that scores document relevance using chain-of-thought reasoning."""

    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(DocumentRelevanceSignature)

    def forward(self, claim, document):
        return self.scorer(claim=claim, document=document)


# ============ Main Pipeline with Iterative Entity Discovery ============

class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop system for retrieving documents for a provided claim using Query Decomposition with Iterative Entity Discovery.

    ARCHITECTURE:
    - Replaces linear 3-hop retrieval with iterative entity-driven approach
    - Uses structured reasoning to discover implicit entities and relationships
    - Each iteration builds on previous knowledge with gap analysis and self-correction

    EVALUATION:
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize new modules for iterative entity discovery
        self.claim_decomposer = dspy.ChainOfThought(ClaimDecomposition)
        self.entity_extractor = dspy.ChainOfThought(EntityExtractor)
        self.gap_analyzer = dspy.ChainOfThought(GapAnalysis)
        self.entity_title_extractor = dspy.ChainOfThought(EntityTitleExtractor)
        self.attribute_extractor = dspy.ChainOfThought(AttributeExtractor)
        self.scorer = DocumentRelevanceScorer()

        # Retrieval modules with different k values
        self.retrieve_semantic = dspy.Retrieve(k=5)  # For semantic queries
        self.retrieve_entity = dspy.Retrieve(k=3)    # For entity-title queries
        self.retrieve_attribute = dspy.Retrieve(k=3)  # For attribute-enhanced queries

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Storage for iterative discovery
            all_retrieved_docs = []
            entities_store = []
            relationships_store = []
            seen_titles = set()  # Track seen titles for deduplication

            def deduplicate_docs(docs):
                """Helper to deduplicate documents by title."""
                unique_docs = []
                for doc in docs:
                    title = doc.split(" | ")[0]
                    if title not in seen_titles:
                        seen_titles.add(title)
                        unique_docs.append(doc)
                return unique_docs

            def normalize_list(value):
                """Helper to normalize DSPy outputs to lists."""
                if not isinstance(value, list):
                    if isinstance(value, str):
                        items = [q.strip() for q in value.split('\n') if q.strip()]
                        items = [q.lstrip('0123456789.-)> ').strip() for q in items if q.strip()]
                        return items
                    return [str(value)] if value else []
                return value

            # ========== ITERATION 1: Triple-Query Architecture (Semantic + Entity + Attribute) ==========

            # STEP 1A: Decompose claim into semantic sub-questions
            try:
                decomposition_result = self.claim_decomposer(claim=claim)
                semantic_queries = normalize_list(decomposition_result.sub_questions)[:3]  # Max 3
            except Exception:
                semantic_queries = [claim]

            # STEP 1B: Extract entity titles from claim
            try:
                entity_title_result = self.entity_title_extractor(claim=claim, context=claim)
                entity_queries = normalize_list(entity_title_result.entity_titles)[:3]  # Max 3 (reduced from 5)
            except Exception:
                entity_queries = []

            # STEP 1C: Extract attributes and generate attribute-enhanced queries
            try:
                attribute_result = self.attribute_extractor(claim=claim, context=claim)
                attribute_queries = normalize_list(attribute_result.entity_attribute_pairs)[:4]  # Max 4
            except Exception:
                attribute_queries = []

            # STEP 1D: Retrieve documents in parallel for all three query types
            iteration1_docs = []

            # Semantic queries: k=5 per query
            for query in semantic_queries:
                try:
                    docs = self.retrieve_semantic(query).passages
                    iteration1_docs.extend(docs)
                except Exception:
                    pass

            # Entity-title queries: k=3 per query
            for entity in entity_queries:
                try:
                    docs = self.retrieve_entity(entity).passages
                    iteration1_docs.extend(docs)
                except Exception:
                    pass

            # Attribute-enhanced queries: k=3 per query
            for attr_query in attribute_queries:
                try:
                    docs = self.retrieve_attribute(attr_query).passages
                    iteration1_docs.extend(docs)
                except Exception:
                    pass

            # STEP 1E: Deduplicate after iteration 1
            iteration1_unique = deduplicate_docs(iteration1_docs)
            all_retrieved_docs.extend(iteration1_unique)

            # STEP 1F: Extract entities and relationships from iteration 1 documents
            if iteration1_unique:
                docs_text = "\n\n".join(iteration1_unique[:15])  # Limit context size
                try:
                    extraction_result = self.entity_extractor(claim=claim, documents=docs_text)
                    entities_store.extend(normalize_list(extraction_result.entities))
                    relationships_store.extend(normalize_list(extraction_result.relationships))
                except Exception:
                    pass

            # ========== ITERATION 2: Triple-Query Architecture (Gap Analysis + Entity + Attribute) ==========

            # STEP 2A: Perform gap analysis for semantic queries
            entities_summary = "\n".join(entities_store) if entities_store else "None found yet"
            relationships_summary = "\n".join(relationships_store) if relationships_store else "None found yet"
            docs_summary = f"Retrieved {len(all_retrieved_docs)} unique documents from iteration 1"

            try:
                gap_result = self.gap_analyzer(
                    claim=claim,
                    entities_found=entities_summary,
                    relationships_found=relationships_summary,
                    documents_retrieved=docs_summary
                )
                semantic_queries_iter2 = normalize_list(gap_result.targeted_queries)[:3]  # Max 3
            except Exception:
                semantic_queries_iter2 = [claim]

            # STEP 2B: Extract entity titles from current context
            context_for_entities = "\n\n".join(all_retrieved_docs[:20]) if all_retrieved_docs else claim
            try:
                entity_title_result2 = self.entity_title_extractor(claim=claim, context=context_for_entities)
                entity_queries_iter2 = normalize_list(entity_title_result2.entity_titles)[:3]  # Max 3 (reduced from 5)
            except Exception:
                entity_queries_iter2 = []

            # STEP 2C: Extract attributes from current context and generate attribute-enhanced queries
            try:
                attribute_result2 = self.attribute_extractor(claim=claim, context=context_for_entities)
                attribute_queries_iter2 = normalize_list(attribute_result2.entity_attribute_pairs)[:4]  # Max 4
            except Exception:
                attribute_queries_iter2 = []

            # STEP 2D: Retrieve documents in parallel for all three query types
            iteration2_docs = []

            # Semantic queries: k=5 per query
            for query in semantic_queries_iter2:
                try:
                    docs = self.retrieve_semantic(query).passages
                    iteration2_docs.extend(docs)
                except Exception:
                    pass

            # Entity-title queries: k=3 per query
            for entity in entity_queries_iter2:
                try:
                    docs = self.retrieve_entity(entity).passages
                    iteration2_docs.extend(docs)
                except Exception:
                    pass

            # Attribute-enhanced queries: k=3 per query
            for attr_query in attribute_queries_iter2:
                try:
                    docs = self.retrieve_attribute(attr_query).passages
                    iteration2_docs.extend(docs)
                except Exception:
                    pass

            # STEP 2E: Deduplicate after iteration 2
            iteration2_unique = deduplicate_docs(iteration2_docs)
            all_retrieved_docs.extend(iteration2_unique)

            # STEP 2F: Extract entities from iteration 2 documents
            if iteration2_unique:
                docs_text = "\n\n".join(iteration2_unique[:15])
                try:
                    extraction_result = self.entity_extractor(claim=claim, documents=docs_text)
                    entities_store.extend(normalize_list(extraction_result.entities))
                    relationships_store.extend(normalize_list(extraction_result.relationships))
                except Exception:
                    pass

            # ========== ITERATION 3: Final Triple-Query Architecture (Gap Analysis + Entity + Attribute) ==========

            # STEP 3A: Final gap analysis for semantic queries
            entities_summary = "\n".join(entities_store) if entities_store else "None found yet"
            relationships_summary = "\n".join(relationships_store) if relationships_store else "None found yet"
            docs_summary = f"Retrieved {len(all_retrieved_docs)} unique documents total across iterations 1-2"

            try:
                final_gap_result = self.gap_analyzer(
                    claim=claim,
                    entities_found=entities_summary,
                    relationships_found=relationships_summary,
                    documents_retrieved=docs_summary
                )
                semantic_queries_iter3 = normalize_list(final_gap_result.targeted_queries)[:3]  # Max 3
            except Exception:
                semantic_queries_iter3 = [claim]

            # STEP 3B: Extract entity titles from accumulated context
            context_for_entities_final = "\n\n".join(all_retrieved_docs[:30]) if all_retrieved_docs else claim
            try:
                entity_title_result3 = self.entity_title_extractor(claim=claim, context=context_for_entities_final)
                entity_queries_iter3 = normalize_list(entity_title_result3.entity_titles)[:3]  # Max 3 (reduced from 5)
            except Exception:
                entity_queries_iter3 = []

            # STEP 3C: Extract attributes from accumulated context and generate attribute-enhanced queries
            try:
                attribute_result3 = self.attribute_extractor(claim=claim, context=context_for_entities_final)
                attribute_queries_iter3 = normalize_list(attribute_result3.entity_attribute_pairs)[:4]  # Max 4
            except Exception:
                attribute_queries_iter3 = []

            # STEP 3D: Retrieve documents in parallel for all three query types
            iteration3_docs = []

            # Semantic queries: k=5 per query
            for query in semantic_queries_iter3:
                try:
                    docs = self.retrieve_semantic(query).passages
                    iteration3_docs.extend(docs)
                except Exception:
                    pass

            # Entity-title queries: k=3 per query
            for entity in entity_queries_iter3:
                try:
                    docs = self.retrieve_entity(entity).passages
                    iteration3_docs.extend(docs)
                except Exception:
                    pass

            # Attribute-enhanced queries: k=3 per query
            for attr_query in attribute_queries_iter3:
                try:
                    docs = self.retrieve_attribute(attr_query).passages
                    iteration3_docs.extend(docs)
                except Exception:
                    pass

            # STEP 3E: Deduplicate after iteration 3
            iteration3_unique = deduplicate_docs(iteration3_docs)
            all_retrieved_docs.extend(iteration3_unique)

            # ========== POST-ITERATION: Scoring and Selection ==========
            # Score each unique document with DocumentRelevanceScorer
            scored_docs = []
            for doc in all_retrieved_docs:
                try:
                    score_result = self.scorer(claim=claim, document=doc)
                    try:
                        score = int(score_result.score)
                    except (ValueError, TypeError):
                        score = 5
                    scored_docs.append((doc, score))
                except Exception:
                    scored_docs.append((doc, 5))

            # Sort by score descending and take top 21
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_docs)
