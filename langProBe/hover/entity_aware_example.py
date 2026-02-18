"""
Example usage of the entity-aware gap analysis retrieval pipeline.

This demonstrates how to use the HoverEntityAwareMultiHop class for
sophisticated claim verification with entity tracking and gap analysis.
"""

import dspy
from langProBe.hover.hover_program import HoverEntityAwareMultiHop

# Configure DSPy with your LM and retrieval model
# Example configuration (update with your actual endpoints)
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    rm=dspy.ColBERTv2(url="https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search")
)

# Initialize the entity-aware multi-hop retrieval system
entity_aware_system = HoverEntityAwareMultiHop()

# Example claim to verify
claim = "Barack Obama was the 44th President of the United States and was born in Hawaii."

# Run the retrieval pipeline
result = entity_aware_system(claim=claim)

# Access results
print(f"Number of retrieved documents: {len(result.retrieved_docs)}")
print(f"\nExtracted entities: {result.entities}")
print(f"\nUncovered entities (gaps): {result.uncovered_entities}")

print("\n" + "="*80)
print("RETRIEVED DOCUMENTS:")
print("="*80)
for i, doc in enumerate(result.retrieved_docs, 1):
    print(f"\n[{i}] {doc[:200]}..." if len(doc) > 200 else f"\n[{i}] {doc}")

# The pipeline workflow:
# 1. Extracts entities: ["Barack Obama", "44th President", "United States", "Hawaii"]
# 2. Retrieves 15 documents with full claim query
# 3. Identifies which entities lack coverage (e.g., "Hawaii" might be uncovered)
# 4. Retrieves 10 documents targeting first uncovered entity
# 5. Retrieves 10 documents targeting second uncovered entity
# 6. Combines and deduplicates all documents
# 7. Reranks by entity coverage and relevance
# 8. Returns top 21 documents
