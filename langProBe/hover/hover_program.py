import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityExtractorSignature(dspy.Signature):
    """Extract 2-3 key entities or concepts from a claim that need verification.
    Identify distinct entities that would require separate evidence (e.g., person names,
    organizations, locations, specific attributes like 'director', 'birthplace').
    For the claim 'The director of Titanic was born in Canada', extract:
    - entity1: 'Titanic' (the film)
    - entity2: 'Titanic director' (the person who directed it)
    - entity3: 'Canada birthplace' (birthplace information)"""

    claim = dspy.InputField(desc="The claim to analyze for key entities")
    entity1 = dspy.OutputField(desc="First key entity or concept requiring evidence")
    entity2 = dspy.OutputField(desc="Second key entity or concept requiring evidence")
    entity3 = dspy.OutputField(desc="Third key entity or concept requiring evidence (may be empty if only 2 entities)")


class EntityExtractor(dspy.Module):
    """DSPy module that extracts 2-3 key entities from a claim."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EntityExtractorSignature)

    def forward(self, claim):
        result = self.extract(claim=claim)
        # Filter out empty entities
        entities = [result.entity1, result.entity2]
        if result.entity3 and result.entity3.strip():
            entities.append(result.entity3)
        return dspy.Prediction(entities=entities)


class EntityQueryGeneratorSignature(dspy.Signature):
    """Generate a focused search query to find evidence for a specific entity/concept
    in the context of verifying the given claim. The query should be optimized to
    retrieve documents containing information about this entity."""

    claim = dspy.InputField(desc="The claim being verified")
    entity = dspy.InputField(desc="The specific entity or concept to search for")
    search_query = dspy.OutputField(desc="A targeted search query for this entity")


class EntityQueryGenerator(dspy.Module):
    """DSPy module that generates search query for a specific entity."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(EntityQueryGeneratorSignature)

    def forward(self, claim, entity):
        result = self.generate(claim=claim, entity=entity)
        return dspy.Prediction(query=result.search_query)


class EntityCoverageAnalyzerSignature(dspy.Signature):
    """Analyze which target entities have insufficient coverage in the retrieved documents.
    Return entities that need more evidence, prioritized by lack of coverage."""

    claim = dspy.InputField(desc="The claim being verified")
    target_entities = dspy.InputField(desc="List of entities that need coverage (comma-separated)")
    retrieved_docs = dspy.InputField(desc="Documents retrieved so far (may be truncated)")
    underrepresented_entity = dspy.OutputField(desc="The entity most lacking coverage, or 'NONE' if all covered")
    reasoning = dspy.OutputField(desc="Explanation of coverage analysis")


class EntityCoverageAnalyzer(dspy.Module):
    """DSPy module that analyzes entity coverage in retrieved documents."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(EntityCoverageAnalyzerSignature)

    def forward(self, claim, target_entities, retrieved_docs):
        # Truncate docs to avoid token limits (send summaries)
        doc_summary = self._summarize_docs(retrieved_docs)
        entities_str = ", ".join(target_entities)

        result = self.analyze(
            claim=claim,
            target_entities=entities_str,
            retrieved_docs=doc_summary
        )
        return dspy.Prediction(
            underrepresented_entity=result.underrepresented_entity,
            reasoning=result.reasoning
        )

    def _summarize_docs(self, docs, max_chars=2000):
        """Summarize docs to avoid token limits - just use titles."""
        titles = [doc.split(" | ")[0] for doc in docs]
        summary = "Retrieved document titles:\n" + "\n".join(f"- {t}" for t in titles[:50])
        return summary[:max_chars]


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop system for retrieving documents using sequential entity-aware retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide exactly 21 documents at the end of the program.

    ARCHITECTURE
    - Uses entity extraction to identify 2-3 key entities/concepts from the claim
    - Sequential 3-hop retrieval (21 docs per hop = 63 docs total):
      * Hop 1: Retrieve for primary entity
      * Hop 2: Analyze coverage, retrieve for underrepresented entity
      * Hop 3: Analyze coverage, retrieve for remaining gaps
    - Diversity-based selection: Deduplicate, ensure ≥1 doc per entity, fill by ColBERT scores
    - Returns exactly 21 documents (no LLM-based scoring)'''

    def __init__(self):
        super().__init__()
        self.k = 21  # Retrieve 21 documents per hop

        # Entity-aware modules
        self.entity_extractor = EntityExtractor()
        self.query_generator = EntityQueryGenerator()
        self.coverage_analyzer = EntityCoverageAnalyzer()

    def forward(self, claim):
        # Step 1: Extract 2-3 key entities from claim
        entity_result = self.entity_extractor(claim=claim)
        target_entities = entity_result.entities  # List of 2-3 entities

        # Data structure to track all retrieved documents with scores
        all_docs_with_scores = []  # List of {doc: str, score: float, hop: int, entity: str}

        # Step 2: Sequential 3-Hop Retrieval
        for hop in range(3):
            if hop == 0:
                # Hop 1: Retrieve for first/primary entity
                current_entity = target_entities[0]
            else:
                # Hop 2 & 3: Analyze coverage and retrieve for underrepresented entity
                retrieved_docs_so_far = [item['doc'] for item in all_docs_with_scores]

                coverage_result = self.coverage_analyzer(
                    claim=claim,
                    target_entities=target_entities,
                    retrieved_docs=retrieved_docs_so_far
                )

                underrep_entity = coverage_result.underrepresented_entity

                # If no underrepresented entity, use next entity in list
                if underrep_entity == "NONE" or hop >= len(target_entities):
                    if hop < len(target_entities):
                        current_entity = target_entities[hop]
                    else:
                        # All entities covered, use first entity again
                        current_entity = target_entities[0]
                else:
                    current_entity = underrep_entity

            # Generate query for current entity
            query_result = self.query_generator(claim=claim, entity=current_entity)
            query = query_result.query

            # Access dspy.settings.rm directly to get scores
            raw_results = dspy.settings.rm(query, k=self.k)

            # Extract documents and scores
            for result in raw_results:
                doc_text = result.long_text  # "Title | Content" format
                score = getattr(result, 'score', 1.0)  # Use score if available, else 1.0

                all_docs_with_scores.append({
                    'doc': doc_text,
                    'score': float(score),
                    'hop': hop + 1,
                    'entity': current_entity
                })

        # Step 3: Diversity-Based Selection (deduplicate + ensure entity coverage)
        final_docs = self._select_diverse_documents(
            all_docs_with_scores,
            target_entities,
            max_docs=21
        )

        return dspy.Prediction(retrieved_docs=final_docs)

    def _select_diverse_documents(self, docs_with_scores, target_entities, max_docs=21):
        """
        Select top documents ensuring diversity and entity coverage.

        Algorithm:
        1. Deduplicate documents (keep highest score for each doc)
        2. Ensure at least 1 document per entity
        3. Fill remaining slots with highest ColBERT scores

        Args:
            docs_with_scores: List of dicts with keys: doc, score, hop, entity
            target_entities: List of entities that need coverage
            max_docs: Maximum documents to return (21)

        Returns:
            List of document strings (max 21)
        """
        # Step 1: Deduplicate - keep highest score for each unique document
        doc_dict = {}  # doc_text -> {score, hop, entity}
        for item in docs_with_scores:
            doc = item['doc']
            if doc not in doc_dict or item['score'] > doc_dict[doc]['score']:
                doc_dict[doc] = {
                    'score': item['score'],
                    'hop': item['hop'],
                    'entity': item['entity']
                }

        # Step 2: Extract document titles for entity matching
        doc_items = []
        for doc, metadata in doc_dict.items():
            title = doc.split(" | ")[0].lower()
            doc_items.append({
                'doc': doc,
                'score': metadata['score'],
                'title': title,
                'entity': metadata['entity']
            })

        # Step 3: Ensure at least 1 document per entity (diversity constraint)
        selected_docs = []
        selected_set = set()

        for entity in target_entities:
            # Find best matching document for this entity (not yet selected)
            entity_lower = entity.lower()
            candidates = [
                item for item in doc_items
                if item['doc'] not in selected_set and entity_lower in item['title']
            ]

            if not candidates:
                # If no title match, use documents retrieved for this entity
                candidates = [
                    item for item in doc_items
                    if item['doc'] not in selected_set and item['entity'] == entity
                ]

            if candidates:
                # Select highest scoring document for this entity
                best = max(candidates, key=lambda x: x['score'])
                selected_docs.append(best['doc'])
                selected_set.add(best['doc'])

        # Step 4: Fill remaining slots with highest ColBERT scores
        remaining_items = [
            item for item in doc_items
            if item['doc'] not in selected_set
        ]
        remaining_items.sort(key=lambda x: x['score'], reverse=True)

        for item in remaining_items:
            if len(selected_docs) >= max_docs:
                break
            selected_docs.append(item['doc'])

        # Step 5: Ensure exactly max_docs (pad if needed, though shouldn't happen)
        return selected_docs[:max_docs]
