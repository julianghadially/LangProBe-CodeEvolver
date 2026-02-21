import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


# ====== Stage 1: Claim Decomposition Signatures ======

class ClaimDecomposer(dspy.Signature):
    """Analyze the claim and decompose it into 2-3 sequential sub-questions representing the logical reasoning hops needed for multi-hop fact verification. Each sub-question should explicitly identify what bridge entities (people, places, organizations, works) connect the reasoning hops."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    sub_questions: list[str] = dspy.OutputField(desc="List of 2-3 sequential sub-questions, each identifying bridge entities that connect to the next hop (e.g., 'Who starred in Film X?', 'What other films did that person star in?')")
    bridge_entities: list[str] = dspy.OutputField(desc="List of expected bridge entity types for each hop (e.g., 'actor name', 'company name', 'location')")


# ====== Stage 2: Bridge Entity Extraction Signatures ======

class BridgeEntityExtractor(dspy.Signature):
    """Extract the specific entity that serves as a bridge to connect to the next reasoning hop from the retrieved documents."""
    sub_question: str = dspy.InputField(desc="The current sub-question being answered")
    documents: str = dspy.InputField(desc="Retrieved documents for the sub-question")
    expected_entity_type: str = dspy.InputField(desc="The type of entity expected as a bridge (e.g., 'actor name', 'location', 'organization')")
    bridge_entity: str = dspy.OutputField(desc="The specific entity extracted that will connect to the next reasoning hop")
    supporting_context: str = dspy.OutputField(desc="Brief context from documents explaining why this entity is the bridge")


# ====== Stage 3: Intermediate Representation Reranking Signatures ======

class MultiHopRelevanceScorer(dspy.Signature):
    """Score documents based on multi-hop retrieval criteria: (a) direct relevance to sub-question, (b) presence of bridge entities, and (c) how documents connect across reasoning hops."""
    sub_question: str = dspy.InputField(desc="The current sub-question")
    documents: str = dspy.InputField(desc="Documents to score, each prefixed with an index like [0], [1], etc.")
    bridge_entities: str = dspy.InputField(desc="Bridge entities identified so far in the reasoning chain")
    top_indices: list[int] = dspy.OutputField(desc="List of top 7-10 document indices ranked by multi-hop relevance score")
    reasoning: str = dspy.OutputField(desc="Explanation of scoring based on sub-question relevance, bridge entity presence, and cross-hop connectivity")


# ====== Stage 4: Final Fusion Signatures ======

class CrossHopConnectivityRanker(dspy.Signature):
    """Score documents based on how well they form a complete reasoning chain across all hops, selecting documents that best support multi-hop verification."""
    claim: str = dspy.InputField(desc="The original claim to be verified")
    all_sub_questions: str = dspy.InputField(desc="All sub-questions representing the reasoning hops")
    documents: str = dspy.InputField(desc="All documents from all hops, each prefixed with an index")
    bridge_entities: str = dspy.InputField(desc="All bridge entities connecting the reasoning chain")
    top_indices: list[int] = dspy.OutputField(desc="List of top 21 document indices that best form a complete reasoning chain")
    reasoning: str = dspy.OutputField(desc="Explanation of how selected documents connect across hops to verify the claim")


# ====== Legacy Signatures (kept for compatibility) ======

class EntityExtraction(dspy.Signature):
    """Extract key entities (people, places, works, organizations) from retrieved documents that are relevant to verifying the claim."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    documents: str = dspy.InputField(desc="Retrieved documents to extract entities from")
    entities: list[str] = dspy.OutputField(desc="List of 1-5 key entities (people, places, works, organizations) most relevant to the claim")


class EntityQueryGenerator(dspy.Signature):
    """Generate a focused search query for a specific entity in the context of the claim."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    entity: str = dspy.InputField(desc="The entity to generate a query for")
    query: str = dspy.OutputField(desc="A focused search query for the entity")


class ListwiseReranker(dspy.Signature):
    """Score and rank documents based on their relevance to the multi-hop reasoning chain needed to verify the claim. Consider how documents connect to support multi-hop reasoning."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    documents: str = dspy.InputField(desc="Documents to rank, each prefixed with an index like [0], [1], etc.")
    top_indices: list[int] = dspy.OutputField(desc="List of document indices ranked by relevance (most relevant first), selecting the top 21 documents")
    reasoning: str = dspy.OutputField(desc="Explanation of the multi-hop reasoning chain and why these documents are most relevant")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Stage 1: Claim Decomposition Module
        self.claim_decomposer = dspy.ChainOfThought(ClaimDecomposer)

        # Stage 2: Bridge Entity Extraction and Sequential Retrieval Modules
        self.bridge_extractor = dspy.ChainOfThought(BridgeEntityExtractor)
        self.sub_question_retrieve = dspy.Retrieve(k=50)

        # Stage 3: Intermediate Representation Reranking Module
        self.multi_hop_scorer = dspy.ChainOfThought(MultiHopRelevanceScorer)

        # Stage 4: Final Fusion Module
        self.cross_hop_ranker = dspy.ChainOfThought(CrossHopConnectivityRanker)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # ====== Stage 1: Claim Decomposition ======
            try:
                decomposition_result = self.claim_decomposer(claim=claim)
                sub_questions = decomposition_result.sub_questions
                expected_bridge_types = decomposition_result.bridge_entities

                # Ensure we have 2-3 sub-questions
                if isinstance(sub_questions, list) and len(sub_questions) > 0:
                    sub_questions = sub_questions[:3]  # Limit to 3 hops max
                else:
                    # Fallback: use claim as single question
                    sub_questions = [claim]
                    expected_bridge_types = ["entity"]

                # Ensure bridge types match sub-questions count
                if not isinstance(expected_bridge_types, list):
                    expected_bridge_types = ["entity"] * len(sub_questions)
                while len(expected_bridge_types) < len(sub_questions):
                    expected_bridge_types.append("entity")

            except Exception:
                # Fallback to single-hop retrieval
                sub_questions = [claim]
                expected_bridge_types = ["entity"]

            # ====== Stage 2: Sequential Bridge Retrieval ======
            all_hop_docs = []  # List of (hop_index, documents) tuples
            bridge_entities_found = []  # Track extracted bridge entities
            current_query = claim  # Start with original claim

            for hop_idx, (sub_question, expected_type) in enumerate(zip(sub_questions, expected_bridge_types)):
                try:
                    # Retrieve k=50 documents for this sub-question
                    # Use the current query which may be enhanced with bridge entity
                    if hop_idx == 0:
                        query = sub_question
                    else:
                        # Incorporate bridge entity from previous hop
                        query = f"{sub_question} {current_query}"

                    hop_docs = self.sub_question_retrieve(query).passages

                    # Store documents for this hop
                    all_hop_docs.append((hop_idx, hop_docs))

                    # Extract bridge entity if not the last hop
                    if hop_idx < len(sub_questions) - 1:
                        try:
                            docs_text = "\n\n".join([f"[{i}] {doc}" for i, doc in enumerate(hop_docs[:20])])  # Use top 20 for extraction

                            bridge_result = self.bridge_extractor(
                                sub_question=sub_question,
                                documents=docs_text,
                                expected_entity_type=expected_type
                            )

                            bridge_entity = bridge_result.bridge_entity
                            bridge_entities_found.append(bridge_entity)

                            # Update current query with bridge entity for next hop
                            current_query = bridge_entity

                        except Exception:
                            # If extraction fails, continue without bridge entity
                            bridge_entities_found.append("")
                            current_query = sub_question

                except Exception:
                    # If retrieval fails for this hop, continue with next
                    all_hop_docs.append((hop_idx, []))
                    continue

            # ====== Stage 3: Intermediate Representation Reranking ======
            hop_selected_docs = []  # Documents selected from each hop

            for hop_idx, hop_docs in all_hop_docs:
                if len(hop_docs) == 0:
                    continue

                try:
                    # Get corresponding sub-question
                    sub_question = sub_questions[hop_idx] if hop_idx < len(sub_questions) else claim

                    # Format documents for scoring
                    indexed_docs = "\n\n".join([f"[{i}] {doc}" for i, doc in enumerate(hop_docs)])

                    # Format bridge entities found so far
                    bridge_context = ", ".join(bridge_entities_found[:hop_idx+1]) if bridge_entities_found else "none yet"

                    # Score documents based on multi-hop criteria
                    score_result = self.multi_hop_scorer(
                        sub_question=sub_question,
                        documents=indexed_docs,
                        bridge_entities=bridge_context
                    )

                    top_indices = score_result.top_indices

                    # Select top 7-10 documents from this hop
                    valid_indices = []
                    for idx in top_indices:
                        if isinstance(idx, int) and 0 <= idx < len(hop_docs):
                            valid_indices.append(idx)
                        if len(valid_indices) >= 10:  # Max 10 per hop
                            break

                    # Add selected documents from this hop
                    selected = [hop_docs[idx] for idx in valid_indices]
                    hop_selected_docs.extend(selected)

                except Exception:
                    # If scoring fails, take top 7 documents from this hop
                    hop_selected_docs.extend(hop_docs[:7])

            # ====== Stage 4: Final Fusion with Cross-Hop Connectivity Ranking ======

            # Deduplicate documents by normalized title
            seen_titles = set()
            unique_docs = []
            for doc in hop_selected_docs:
                title = doc.split(" | ")[0]
                normalized_title = title.lower().strip()
                if normalized_title not in seen_titles:
                    seen_titles.add(normalized_title)
                    unique_docs.append(doc)

            # Apply final cross-hop connectivity ranker
            try:
                # Format all sub-questions
                all_sub_questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(sub_questions)])

                # Format all bridge entities
                all_bridge_entities_text = ", ".join(bridge_entities_found) if bridge_entities_found else "none identified"

                # Format all unique documents
                indexed_unique_docs = "\n\n".join([f"[{i}] {doc}" for i, doc in enumerate(unique_docs)])

                # Rank based on cross-hop connectivity
                final_rank_result = self.cross_hop_ranker(
                    claim=claim,
                    all_sub_questions=all_sub_questions_text,
                    documents=indexed_unique_docs,
                    bridge_entities=all_bridge_entities_text
                )

                final_indices = final_rank_result.top_indices

                # Select final 21 documents
                valid_final_indices = []
                for idx in final_indices:
                    if isinstance(idx, int) and 0 <= idx < len(unique_docs):
                        valid_final_indices.append(idx)
                    if len(valid_final_indices) >= 21:
                        break

                final_docs = [unique_docs[idx] for idx in valid_final_indices]

            except Exception:
                # If final ranking fails, use first 21 unique docs
                final_docs = unique_docs[:21]

            # Ensure we have at most 21 documents
            final_docs = final_docs[:21]

            return dspy.Prediction(retrieved_docs=final_docs)
