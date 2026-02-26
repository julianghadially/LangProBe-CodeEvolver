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


class BridgingEntityIdentifier(dspy.Signature):
    """Identify specific bridging entities (people, organizations, events) mentioned in retrieved documents that are not explicitly in the original claim but are crucial for verification.
    These entities appear in the documents as important intermediate connections and need their own dedicated document retrieval."""

    claim: str = dspy.InputField(desc="the original claim being verified")
    documents: str = dspy.InputField(desc="the retrieved documents to analyze for bridging entities")
    bridging_entities: list[str] = dspy.OutputField(desc="3-5 specific entity names (people, organizations, events) that appear in documents but need standalone retrieval (e.g., 'Lisa Raymond', 'Ellis Ferreira', 'Wimbledon Championships')")


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
        self.bridging_identifier = dspy.ChainOfThought(BridgingEntityIdentifier)
        self.scorer = DocumentRelevanceScorer()

        # Retrieval module
        self.retrieve_k = dspy.Retrieve(k=12)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Storage for iterative discovery
            all_retrieved_docs = []
            entities_store = []
            relationships_store = []

            # ========== ITERATION 1: Decompose and Initial Retrieval ==========
            # Decompose claim into 2-3 sub-questions
            decomposition_result = self.claim_decomposer(claim=claim)
            sub_questions = decomposition_result.sub_questions

            # Ensure we have a list of sub-questions (handle different DSPy output formats)
            if not isinstance(sub_questions, list):
                # If it's a string, try to parse it as a list
                if isinstance(sub_questions, str):
                    # Try to split by newlines or common delimiters
                    sub_questions = [q.strip() for q in sub_questions.split('\n') if q.strip()]
                    # Remove numbered prefixes like "1.", "2.", etc.
                    sub_questions = [q.lstrip('0123456789.-)> ').strip() for q in sub_questions if q.strip()]
                else:
                    # Fallback: use original claim
                    sub_questions = [claim]

            # Limit to first 3 sub-questions to control retrieval volume
            sub_questions = sub_questions[:3]

            # Retrieve documents for each sub-question (k=5 per sub-question)
            iteration1_docs = []
            for sub_q in sub_questions:
                try:
                    docs = self.retrieve_k(sub_q).passages
                    iteration1_docs.extend(docs)
                except Exception:
                    # If retrieval fails, continue with other sub-questions
                    pass

            all_retrieved_docs.extend(iteration1_docs)

            # Extract entities and relationships from iteration 1 documents
            if iteration1_docs:
                docs_text = "\n\n".join(iteration1_docs[:15])  # Limit context size
                try:
                    extraction_result = self.entity_extractor(claim=claim, documents=docs_text)

                    # Handle extraction results (ensure they are lists)
                    entities = extraction_result.entities
                    relationships = extraction_result.relationships

                    if not isinstance(entities, list):
                        entities = [str(entities)] if entities else []
                    if not isinstance(relationships, list):
                        relationships = [str(relationships)] if relationships else []

                    entities_store.extend(entities)
                    relationships_store.extend(relationships)
                except Exception:
                    # If extraction fails, continue without entities
                    pass

            # ========== BRIDGING ENTITY DISCOVERY: Identify and Retrieve Implicit Entities ==========
            # After iteration 1, identify bridging entities that appear in documents but need dedicated retrieval
            if iteration1_docs:
                docs_text_for_bridging = "\n\n".join(iteration1_docs[:15])  # Limit context size
                try:
                    bridging_result = self.bridging_identifier(claim=claim, documents=docs_text_for_bridging)

                    # Get bridging entities
                    bridging_entities = bridging_result.bridging_entities
                    if not isinstance(bridging_entities, list):
                        if isinstance(bridging_entities, str):
                            # Parse string into list
                            bridging_entities = [e.strip() for e in bridging_entities.split('\n') if e.strip()]
                            # Remove numbered prefixes
                            bridging_entities = [e.lstrip('0123456789.-)> ').strip() for e in bridging_entities if e.strip()]
                        else:
                            bridging_entities = []

                    # Limit to 3-5 entities to control retrieval volume
                    bridging_entities = bridging_entities[:5]

                    # Retrieve documents for each bridging entity using just the entity name
                    bridging_docs = []
                    for entity_name in bridging_entities:
                        try:
                            # Direct retrieve using entity name as query (e.g., 'Lisa Raymond', 'Ellis Ferreira')
                            docs = self.retrieve_k(entity_name).passages
                            bridging_docs.extend(docs)
                        except Exception:
                            # If retrieval fails for this entity, continue with others
                            pass

                    # Add bridging documents to all_retrieved_docs before iteration 2
                    all_retrieved_docs.extend(bridging_docs)

                except Exception:
                    # If bridging entity identification fails, continue without it
                    pass

            # ========== ITERATION 2: Gap Analysis and Targeted Retrieval ==========
            # Perform gap analysis to identify missing information
            entities_summary = "\n".join(entities_store) if entities_store else "None found yet"
            relationships_summary = "\n".join(relationships_store) if relationships_store else "None found yet"
            docs_summary = f"Retrieved {len(iteration1_docs)} documents from sub-questions"

            try:
                gap_result = self.gap_analyzer(
                    claim=claim,
                    entities_found=entities_summary,
                    relationships_found=relationships_summary,
                    documents_retrieved=docs_summary
                )

                # Get targeted queries from gap analysis
                targeted_queries = gap_result.targeted_queries
                if not isinstance(targeted_queries, list):
                    if isinstance(targeted_queries, str):
                        targeted_queries = [q.strip() for q in targeted_queries.split('\n') if q.strip()]
                        targeted_queries = [q.lstrip('0123456789.-)> ').strip() for q in targeted_queries if q.strip()]
                    else:
                        targeted_queries = []

                # Limit to 3 queries to control retrieval volume
                targeted_queries = targeted_queries[:3]

            except Exception:
                # If gap analysis fails, use fallback queries
                targeted_queries = [claim]

            # Retrieve documents for each targeted query (k=5 per query)
            iteration2_docs = []
            for query in targeted_queries:
                try:
                    docs = self.retrieve_k(query).passages
                    iteration2_docs.extend(docs)
                except Exception:
                    pass

            all_retrieved_docs.extend(iteration2_docs)

            # Extract entities from iteration 2 documents
            if iteration2_docs:
                docs_text = "\n\n".join(iteration2_docs[:15])
                try:
                    extraction_result = self.entity_extractor(claim=claim, documents=docs_text)

                    entities = extraction_result.entities
                    relationships = extraction_result.relationships

                    if not isinstance(entities, list):
                        entities = [str(entities)] if entities else []
                    if not isinstance(relationships, list):
                        relationships = [str(relationships)] if relationships else []

                    entities_store.extend(entities)
                    relationships_store.extend(relationships)
                except Exception:
                    pass

            # ========== ITERATION 3: Final Gap Analysis and Specific Retrieval ==========
            # Update summaries with iteration 2 findings
            entities_summary = "\n".join(entities_store) if entities_store else "None found yet"
            relationships_summary = "\n".join(relationships_store) if relationships_store else "None found yet"
            docs_summary = f"Retrieved {len(all_retrieved_docs)} documents total across iterations 1-2"

            try:
                final_gap_result = self.gap_analyzer(
                    claim=claim,
                    entities_found=entities_summary,
                    relationships_found=relationships_summary,
                    documents_retrieved=docs_summary
                )

                # Get final targeted queries
                final_queries = final_gap_result.targeted_queries
                if not isinstance(final_queries, list):
                    if isinstance(final_queries, str):
                        final_queries = [q.strip() for q in final_queries.split('\n') if q.strip()]
                        final_queries = [q.lstrip('0123456789.-)> ').strip() for q in final_queries if q.strip()]
                    else:
                        final_queries = []

                # Limit to 3 queries
                final_queries = final_queries[:3]

            except Exception:
                # Fallback: use claim-based query
                final_queries = [claim]

            # Retrieve documents for final queries (k=5 per query)
            iteration3_docs = []
            for query in final_queries:
                try:
                    docs = self.retrieve_k(query).passages
                    iteration3_docs.extend(docs)
                except Exception:
                    pass

            all_retrieved_docs.extend(iteration3_docs)

            # ========== POST-ITERATION: Deduplication, Scoring, and Selection ==========
            # Deduplicate documents based on title (before " | ")
            unique_docs = []
            seen_titles = set()
            for doc in all_retrieved_docs:
                title = doc.split(" | ")[0]
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_docs.append(doc)

            # Score each unique document with DocumentRelevanceScorer
            scored_docs = []
            for doc in unique_docs:
                try:
                    score_result = self.scorer(claim=claim, document=doc)
                    # Parse score as integer, default to 5 if parsing fails
                    try:
                        score = int(score_result.score)
                    except (ValueError, TypeError):
                        score = 5
                    scored_docs.append((doc, score))
                except Exception:
                    # If scoring fails, assign neutral score
                    scored_docs.append((doc, 5))

            # Sort by score descending and take top 21
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_docs)
