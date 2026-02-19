import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ExtractEntities(dspy.Signature):
    """Extract named entities (people, places, organizations) from retrieved documents.

    Identify and list all named entities mentioned in the passages, including their types.
    Focus on entities that could be relevant to connecting different pieces of information."""

    claim: str = dspy.InputField(desc="The claim being verified")
    passages: list[str] = dspy.InputField(desc="Retrieved document passages")
    entities: list[str] = dspy.OutputField(desc="List of named entities with their types (format: 'Entity Name (TYPE)'). Include people, places, organizations, dates, events.")


class IdentifyBridgeEntities(dspy.Signature):
    """Identify entities that likely bridge to missing information for the claim.

    Analyze the claim and currently extracted entities to find entities that are mentioned
    but not fully explored. These bridge entities connect different facts and can lead to
    additional relevant documents."""

    claim: str = dspy.InputField(desc="The claim being verified")
    extracted_entities: list[str] = dspy.InputField(desc="Entities already extracted from retrieved documents")
    passages: list[str] = dspy.InputField(desc="Currently retrieved passages")
    bridge_entities: list[str] = dspy.OutputField(desc="List of 3-5 bridge entities that likely connect to missing information, ranked by importance")


class GenerateBridgedQuery(dspy.Signature):
    """Generate a targeted search query based on a bridge entity and claim context.

    Create a search query that explores the bridge entity in the context of the claim,
    aiming to find documents that connect different pieces of information."""

    claim: str = dspy.InputField(desc="The claim being verified")
    bridge_entity: str = dspy.InputField(desc="The bridge entity to explore")
    context: str = dspy.InputField(desc="Summary of what has been discovered so far")
    query: str = dspy.OutputField(desc="A targeted search query for the bridge entity")


class RerankDocuments(dspy.Signature):
    """Rerank all retrieved documents to select the most relevant ones for claim verification.

    Prioritize documents that: (1) contain multiple claim-relevant entities, (2) provide
    connections between different facts, (3) directly support or refute the claim."""

    claim: str = dspy.InputField(desc="The claim being verified")
    all_passages: list[str] = dspy.InputField(desc="All retrieved passages from multiple hops")
    extracted_entities: list[str] = dspy.InputField(desc="All extracted entities")
    top_indices: list[int] = dspy.OutputField(desc="Indices of the top 21 most relevant passages (0-indexed)")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Entity-bridge retrieval components
        self.extract_entities = dspy.ChainOfThought(ExtractEntities)
        self.identify_bridges = dspy.ChainOfThought(IdentifyBridgeEntities)
        self.generate_bridged_query = dspy.ChainOfThought(GenerateBridgedQuery)
        self.rerank_documents = dspy.ChainOfThought(RerankDocuments)

        # Retrieval with k=50 for each hop
        self.retrieve_k50 = dspy.Retrieve(k=50)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # HOP 1: Retrieve k=50 docs based on the claim
            hop1_docs = self.retrieve_k50(claim).passages

            # Extract entities from hop 1 documents
            entities_hop1 = self.extract_entities(
                claim=claim,
                passages=hop1_docs
            ).entities

            # Identify bridge entities that likely connect to missing information
            bridge_entities_hop1 = self.identify_bridges(
                claim=claim,
                extracted_entities=entities_hop1,
                passages=hop1_docs
            ).bridge_entities

            # HOP 2: Generate query targeting the top bridge entity (k=50)
            if bridge_entities_hop1 and len(bridge_entities_hop1) > 0:
                top_bridge_entity = bridge_entities_hop1[0] if isinstance(bridge_entities_hop1[0], str) else str(bridge_entities_hop1[0])
                context_hop1 = f"Extracted entities: {', '.join(map(str, entities_hop1[:10]))}"

                hop2_query = self.generate_bridged_query(
                    claim=claim,
                    bridge_entity=top_bridge_entity,
                    context=context_hop1
                ).query
            else:
                # Fallback if no bridge entities identified
                hop2_query = claim

            hop2_docs = self.retrieve_k50(hop2_query).passages

            # Extract entities from hop 2 documents
            combined_entities_hop2 = list(set(list(entities_hop1) + list(self.extract_entities(
                claim=claim,
                passages=hop2_docs
            ).entities)))

            # Identify new bridge entities from combined information
            bridge_entities_hop2 = self.identify_bridges(
                claim=claim,
                extracted_entities=combined_entities_hop2,
                passages=hop1_docs + hop2_docs
            ).bridge_entities

            # HOP 3: Generate query for the next most promising bridge entity (k=50)
            if bridge_entities_hop2 and len(bridge_entities_hop2) > 1:
                # Use the second bridge entity (first might be already explored)
                next_bridge_entity = bridge_entities_hop2[1] if isinstance(bridge_entities_hop2[1], str) else str(bridge_entities_hop2[1])
                context_hop2 = f"Explored: {top_bridge_entity if 'top_bridge_entity' in locals() else 'claim'}. Entities: {', '.join(map(str, combined_entities_hop2[:15]))}"

                hop3_query = self.generate_bridged_query(
                    claim=claim,
                    bridge_entity=next_bridge_entity,
                    context=context_hop2
                ).query
            elif bridge_entities_hop2 and len(bridge_entities_hop2) > 0:
                # Use the first if only one available
                next_bridge_entity = bridge_entities_hop2[0] if isinstance(bridge_entities_hop2[0], str) else str(bridge_entities_hop2[0])
                context_hop2 = f"Entities: {', '.join(map(str, combined_entities_hop2[:15]))}"

                hop3_query = self.generate_bridged_query(
                    claim=claim,
                    bridge_entity=next_bridge_entity,
                    context=context_hop2
                ).query
            else:
                # Fallback
                hop3_query = claim

            hop3_docs = self.retrieve_k50(hop3_query).passages

            # Extract all entities from hop 3
            all_entities = list(set(list(combined_entities_hop2) + list(self.extract_entities(
                claim=claim,
                passages=hop3_docs
            ).entities)))

            # Combine all 150 retrieved documents
            all_docs = hop1_docs + hop2_docs + hop3_docs

            # Apply LLM-based listwise reranking to select final 21 documents
            # Prioritize documents containing multiple claim-relevant entities
            top_indices = self.rerank_documents(
                claim=claim,
                all_passages=all_docs,
                extracted_entities=all_entities
            ).top_indices

            # Select the top 21 documents based on reranking
            # Ensure we handle the indices correctly
            final_docs = []
            for idx in top_indices[:21]:  # Take at most 21
                if isinstance(idx, int) and 0 <= idx < len(all_docs):
                    final_docs.append(all_docs[idx])

            # If reranking didn't return enough valid indices, fill with original docs
            if len(final_docs) < 21:
                for i, doc in enumerate(all_docs):
                    if len(final_docs) >= 21:
                        break
                    if i not in top_indices and doc not in final_docs:
                        final_docs.append(doc)

            return dspy.Prediction(retrieved_docs=final_docs[:21])
