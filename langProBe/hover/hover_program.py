import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class ExtractNamedEntities(dspy.Signature):
    """Extract 2-4 specific named entities (people, bands, organizations, titles) directly from the claim text.
    Focus on concrete entities that can be searched, not abstract concepts."""

    claim = dspy.InputField(desc="The claim to analyze")

    entities = dspy.OutputField(desc="A comma-separated list of 2-4 specific named entities from the claim")
    reasoning = dspy.OutputField(desc="Explanation of why these entities are key to verifying the claim")

class ExtractQueryAspects(dspy.Signature):
    """Identify a distinct queryable aspect or entity from the claim that hasn't been covered yet.
    The aspect should be a specific entity, person, place, event, or concept that can be searched for.
    Ensure the aspect is different from the previous aspects already covered."""

    claim = dspy.InputField(desc="The claim to analyze")
    previous_aspects = dspy.InputField(desc="List of aspects already covered in previous hops")

    aspect = dspy.OutputField(desc="A distinct queryable aspect, entity, or concept from the claim")
    reasoning = dspy.OutputField(desc="Explanation of why this aspect is important and different from previous aspects")

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.extract_entities = dspy.ChainOfThought(ExtractNamedEntities)
        self.extract_aspect = dspy.ChainOfThought(ExtractQueryAspects)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # PHASE 1: Extract and retrieve documents for named entities
        entity_result = self.extract_entities(claim=claim)
        entities_str = entity_result.entities

        # Parse entities from comma-separated string
        entities = [e.strip() for e in entities_str.split(',')]

        # Retrieve k=10 documents for EACH entity in parallel (stores all results)
        all_entity_docs = []
        entity_retriever = dspy.Retrieve(k=10)
        for entity in entities:
            entity_docs = entity_retriever(entity).passages
            all_entity_docs.extend(entity_docs)

        # PHASE 2: Use aspect-based approach for ONE connecting hop
        # Extract a single connecting aspect with k=15
        covered_aspects = entities  # Track entities as covered aspects
        aspect_result = self.extract_aspect(
            claim=claim,
            previous_aspects=covered_aspects
        )
        connecting_aspect = aspect_result.aspect
        connecting_retriever = dspy.Retrieve(k=15)
        connecting_docs = connecting_retriever(connecting_aspect).passages

        # Combine all retrieved passages
        all_passages = all_entity_docs + connecting_docs

        # PHASE 3: Score-based deduplication to keep most relevant unique documents
        # Create a dictionary with passage text as key and (passage, score) as value
        unique_passages = {}
        for idx, passage in enumerate(all_passages):
            # Use the passage text as the key for deduplication
            passage_text = passage if isinstance(passage, str) else str(passage)

            # Assign a score: earlier documents from entity search get higher base scores
            # Entity docs: indices 0 to len(all_entity_docs)-1
            # Connecting docs: indices len(all_entity_docs) onwards
            if idx < len(all_entity_docs):
                # Entity docs: higher score (prioritize entity-based retrieval)
                # Within entity docs, earlier = higher score
                score = 1000 - idx
            else:
                # Connecting docs: lower base score but still ordered
                score = 500 - (idx - len(all_entity_docs))

            # Keep the passage with the highest score for each unique text
            if passage_text not in unique_passages or score > unique_passages[passage_text][1]:
                unique_passages[passage_text] = (passage, score)

        # Sort by score (descending) and take top 21
        sorted_passages = sorted(unique_passages.values(), key=lambda x: x[1], reverse=True)
        top_21_passages = [p[0] for p in sorted_passages[:21]]

        return dspy.Prediction(retrieved_docs=top_21_passages)
