import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        # Entity extraction module
        self.extract_entities = dspy.ChainOfThought("claim -> entities")
        # Retrieve module for entity-based search
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # Step 1: Extract 2-4 key entities/concepts from the claim
        entity_extraction = self.extract_entities(claim=claim)
        entities_text = entity_extraction.entities

        # Parse entities from the output (expecting comma-separated or list format)
        # Handle various formats the LLM might return
        if isinstance(entities_text, str):
            # Split by common delimiters
            entities = [e.strip() for e in entities_text.replace('\n', ',').split(',') if e.strip()]
            # Remove numbering (e.g., "1. Entity" -> "Entity")
            entities = [e.split('.', 1)[-1].strip() if e[0].isdigit() else e for e in entities]
            # Filter out empty strings and limit to 2-4 entities
            entities = [e for e in entities if e][:4]
            if len(entities) < 2:
                entities = [claim]  # Fallback to using claim if extraction fails
        else:
            entities = [claim]

        num_entities = len(entities)
        all_docs = []
        seen_docs = set()  # For deduplication

        # Step 2 & 3: Allocate hops strategically based on number of entities
        if num_entities == 2:
            # 2 entities: 1 hop each for primary entities + 1 connecting query
            # Hop 1: First entity
            hop1_docs = self.retrieve_k(entities[0]).passages
            for doc in hop1_docs:
                if doc not in seen_docs:
                    all_docs.append(doc)
                    seen_docs.add(doc)

            # Hop 2: Second entity
            hop2_docs = self.retrieve_k(entities[1]).passages
            for doc in hop2_docs:
                if doc not in seen_docs:
                    all_docs.append(doc)
                    seen_docs.add(doc)

            # Hop 3: Connecting/verification query combining both entities
            connecting_query = f"{entities[0]} {entities[1]} {claim}"
            hop3_docs = self.retrieve_k(connecting_query).passages
            for doc in hop3_docs:
                if doc not in seen_docs:
                    all_docs.append(doc)
                    seen_docs.add(doc)

        elif num_entities >= 3:
            # 3+ entities: 1 hop per top entity (up to 3 hops)
            top_entities = entities[:3]
            for entity in top_entities:
                hop_docs = self.retrieve_k(entity).passages
                for doc in hop_docs:
                    if doc not in seen_docs:
                        all_docs.append(doc)
                        seen_docs.add(doc)

        else:
            # 1 entity (fallback): use sequential approach with claim
            hop1_docs = self.retrieve_k(claim).passages
            for doc in hop1_docs:
                if doc not in seen_docs:
                    all_docs.append(doc)
                    seen_docs.add(doc)

            hop2_docs = self.retrieve_k(f"{claim} evidence").passages
            for doc in hop2_docs:
                if doc not in seen_docs:
                    all_docs.append(doc)
                    seen_docs.add(doc)

            hop3_docs = self.retrieve_k(f"{claim} verification").passages
            for doc in hop3_docs:
                if doc not in seen_docs:
                    all_docs.append(doc)
                    seen_docs.add(doc)

        # Step 4: Return final 21 documents (deduplicated and combined)
        # Limit to 21 documents total
        final_docs = all_docs[:21]

        return dspy.Prediction(retrieved_docs=final_docs)


