import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ExtractKeyEntities(dspy.Signature):
    """Extract 3-5 key named entities (people, places, works, events) from a claim that would be most helpful for finding supporting documents."""

    claim: str = dspy.InputField(desc="the claim to extract entities from")
    entities: list[str] = dspy.OutputField(desc="list of 3-5 key named entities (people, places, works, events)")


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


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.entity_extractor = dspy.Predict(ExtractKeyEntities)
        self.retrieve_k35 = dspy.Retrieve(k=35)
        self.scorer = DocumentRelevanceScorer()

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Extract key entities from the claim
            entity_result = self.entity_extractor(claim=claim)
            entities = entity_result.entities

            # Ensure entities is a list and limit to first 3 entities for retrieval
            if isinstance(entities, str):
                # If entities is a string, try to parse it as a list
                import ast
                try:
                    entities = ast.literal_eval(entities)
                except:
                    # If parsing fails, split by common delimiters
                    entities = [e.strip() for e in entities.replace('\n', ',').split(',') if e.strip()]

            # Limit to 3 entities for retrieval (as per constraint)
            entities = entities[:3] if isinstance(entities, list) else []

            # Step 2: Perform entity-based retrieval
            all_docs = []
            for entity in entities:
                if entity:
                    # Retrieve k=35 documents per entity query
                    entity_docs = self.retrieve_k35(entity).passages
                    all_docs.extend(entity_docs)

            # Step 3: Deduplicate documents based on title
            unique_docs = []
            seen_titles = set()
            for doc in all_docs:
                title = doc.split(" | ")[0]
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_docs.append(doc)

            # Step 4: Score and rerank documents
            # Extract claim keywords for scoring
            claim_lower = claim.lower()
            claim_words = set(word.strip('.,!?;:()[]{}\"\'') for word in claim_lower.split())

            scored_docs = []
            for doc in unique_docs:
                # Parse document title and content
                parts = doc.split(" | ", 1)
                title = parts[0] if len(parts) > 0 else ""
                content = parts[1] if len(parts) > 1 else ""

                title_lower = title.lower()
                content_lower = content.lower()

                # Scoring mechanism:
                # 1. Exact entity name match in title (highest priority)
                entity_match_score = 0
                for entity in entities:
                    if entity and entity.lower() in title_lower:
                        entity_match_score += 100

                # 2. Claim keyword overlap (medium priority)
                title_words = set(word.strip('.,!?;:()[]{}\"\'') for word in title_lower.split())
                content_words = set(word.strip('.,!?;:()[]{}\"\'') for word in content_lower.split())

                title_overlap = len(claim_words.intersection(title_words))
                content_overlap = len(claim_words.intersection(content_words))
                keyword_score = title_overlap * 10 + content_overlap * 2

                # 3. Document relevance score (baseline priority)
                relevance_score = 0
                try:
                    score_result = self.scorer(claim=claim, document=doc)
                    try:
                        relevance_score = int(score_result.score)
                    except (ValueError, TypeError):
                        relevance_score = 5
                except Exception:
                    relevance_score = 5

                # Combined score: entity match >> keyword overlap >> relevance
                total_score = entity_match_score + keyword_score + relevance_score
                scored_docs.append((doc, total_score))

            # Sort by score descending and take top 21
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_docs)
