import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


# Signature classes for the three parallel query generators
class EntityFocusedQuerySignature(dspy.Signature):
    """Extract key entities from the claim and generate a focused query to find documents about these specific entities."""
    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a query focusing on key entities mentioned in the claim")


class ComparativeQuerySignature(dspy.Signature):
    """Identify comparative or contrasting elements in the claim and generate a query to find documents for comparison."""
    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a query focusing on comparative/contrasting elements in the claim")


class ContextualQuerySignature(dspy.Signature):
    """Generate a broader contextual query to find background information and context related to the claim."""
    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a broader contextual query for background information")


class DocumentScoringSignature(dspy.Signature):
    """Score a document based on its relevance to the claim and its diversity compared to already-selected documents.
    Consider: (a) how relevant the document is to verifying the claim, and (b) how unique/diverse the information is compared to already-selected documents."""
    claim: str = dspy.InputField()
    document: str = dspy.InputField(desc="the document to score")
    already_selected: str = dspy.InputField(desc="titles of documents already selected, for diversity assessment")
    relevance_score: float = dspy.OutputField(desc="relevance score from 0.0 to 1.0")
    diversity_score: float = dspy.OutputField(desc="diversity/uniqueness score from 0.0 to 1.0")
    combined_score: float = dspy.OutputField(desc="combined score balancing relevance and diversity from 0.0 to 1.0")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using Parallel Multi-Query Diversified Retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize the three parallel query generators
        self.entity_query_gen = dspy.ChainOfThought(EntityFocusedQuerySignature)
        self.comparative_query_gen = dspy.ChainOfThought(ComparativeQuerySignature)
        self.contextual_query_gen = dspy.ChainOfThought(ContextualQuerySignature)

        # Retrieve k=21 documents for each query
        self.retrieve_k = dspy.Retrieve(k=21)

        # Diversity-based reranking module
        self.document_scorer = dspy.ChainOfThought(DocumentScoringSignature)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Generate three different queries in parallel (conceptually)
            entity_query = self.entity_query_gen(claim=claim).query
            comparative_query = self.comparative_query_gen(claim=claim).query
            contextual_query = self.contextual_query_gen(claim=claim).query

            # Retrieve documents for each query (k=21 each, total 63)
            entity_docs = self.retrieve_k(entity_query).passages
            comparative_docs = self.retrieve_k(comparative_query).passages
            contextual_docs = self.retrieve_k(contextual_query).passages

            # Combine all retrieved documents (63 total)
            all_docs = entity_docs + comparative_docs + contextual_docs

            # Deduplicate by document title (first part before " | ")
            seen_titles = set()
            unique_docs = []
            for doc in all_docs:
                title = doc.split(" | ")[0] if " | " in doc else doc
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_docs.append(doc)

            # Diversity-based reranking: select exactly 21 diverse, high-quality documents
            selected_docs = []
            remaining_docs = unique_docs.copy()

            # Select documents iteratively based on relevance and diversity
            for i in range(min(21, len(remaining_docs))):
                best_doc = None
                best_score = -1.0
                best_idx = -1

                # Get titles of already selected documents for diversity assessment
                selected_titles = ", ".join([d.split(" | ")[0] for d in selected_docs]) if selected_docs else "none"

                # Score each remaining document
                for idx, doc in enumerate(remaining_docs):
                    try:
                        # Use the scoring module to evaluate relevance and diversity
                        scoring_result = self.document_scorer(
                            claim=claim,
                            document=doc,
                            already_selected=selected_titles
                        )

                        # Extract combined score
                        combined_score = float(scoring_result.combined_score)

                        if combined_score > best_score:
                            best_score = combined_score
                            best_doc = doc
                            best_idx = idx
                    except (ValueError, AttributeError):
                        # If scoring fails, use a simple heuristic (position-based score)
                        # Earlier documents in the list are assumed more relevant
                        heuristic_score = 1.0 / (idx + 1) * (1.0 / (i + 1))
                        if heuristic_score > best_score:
                            best_score = heuristic_score
                            best_doc = doc
                            best_idx = idx

                # Add the best document to selected set
                if best_doc is not None:
                    selected_docs.append(best_doc)
                    remaining_docs.pop(best_idx)
                else:
                    break

            # If we still don't have 21 docs, fill with remaining docs
            while len(selected_docs) < 21 and remaining_docs:
                selected_docs.append(remaining_docs.pop(0))

            return dspy.Prediction(retrieved_docs=selected_docs)
