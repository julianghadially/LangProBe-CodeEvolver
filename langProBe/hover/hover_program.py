import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.decompose_claim = dspy.ChainOfThought("claim -> sub_questions: list[str], reasoning: str")
        self.decompose_claim.__doc__ = "Decompose the claim into multiple targeted sub-questions that ask about specific entities, relationships, and facts needed for verification. Each sub-question should focus on retrieving a specific Wikipedia article (e.g., 'Who wrote The Broken Tower?', 'What is The Greatest Game Ever Played?')."
        self.rerank_passages = dspy.ChainOfThought("sub_question, passages -> selected_passages: list[str], reasoning: str")
        self.rerank_passages.__doc__ = "Select the top 1-2 most relevant passages for the given sub-question. Focus on passages that directly answer the question or provide critical information about the entities mentioned."
        self.select_docs_hop1 = dspy.ChainOfThought("claim, passages -> selected_passages: list[str], reasoning: str")
        self.select_docs_hop2 = dspy.ChainOfThought("claim, passages -> selected_passages: list[str], reasoning: str")
        self.create_query_hop2 = dspy.ChainOfThought("claim, selected_passages -> query")
        self.create_query_hop3 = dspy.ChainOfThought("claim, selected_passages -> query")
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # QUERY DECOMPOSITION
        decomposition_result = self.decompose_claim(claim=claim)
        sub_questions = decomposition_result.sub_questions

        # Retrieve k=3 documents per sub-question
        sub_question_retrieve = dspy.Retrieve(k=3)
        all_retrieved_docs = []

        for sub_question in sub_questions:
            # Retrieve 3 documents for this sub-question
            retrieved_result = sub_question_retrieve(sub_question)
            retrieved_passages = retrieved_result.passages

            # Apply reranking to select top 1-2 most relevant docs
            rerank_result = self.rerank_passages(
                sub_question=sub_question,
                passages=retrieved_passages
            )
            selected_passages = rerank_result.selected_passages

            # Keep only top 1-2 passages (limit the selection)
            selected_passages = selected_passages[:2]
            all_retrieved_docs.extend(selected_passages)

        # Deduplicate selected documents from query decomposition
        decomposition_docs = list(set(all_retrieved_docs))

        # Calculate dynamic k for remaining hops to maintain total of 21 docs
        dynamic_k = max(1, (21 - len(decomposition_docs)) // 3)
        retrieve_dynamic = dspy.Retrieve(k=dynamic_k)

        # HOP 1
        hop1_docs = retrieve_dynamic(claim).passages
        selection_1 = self.select_docs_hop1(
            claim=claim, passages=hop1_docs
        )
        selected_passages_1 = selection_1.selected_passages

        # HOP 2
        hop2_query = self.create_query_hop2(
            claim=claim, selected_passages=selected_passages_1
        ).query
        hop2_docs = retrieve_dynamic(hop2_query).passages
        selection_2 = self.select_docs_hop2(
            claim=claim, passages=hop2_docs
        )
        selected_passages_2 = selection_2.selected_passages

        # HOP 3
        # Combine selected passages from hop1 and hop2 for context
        combined_selected = selected_passages_1 + selected_passages_2
        hop3_query = self.create_query_hop3(
            claim=claim, selected_passages=combined_selected
        ).query
        hop3_docs = retrieve_dynamic(hop3_query).passages

        # Deduplicate: remove hop docs that are already in decomposition_docs
        decomposition_docs_set = set(decomposition_docs)
        hop1_docs_dedup = [doc for doc in hop1_docs if doc not in decomposition_docs_set]
        hop2_docs_dedup = [doc for doc in hop2_docs if doc not in decomposition_docs_set]
        hop3_docs_dedup = [doc for doc in hop3_docs if doc not in decomposition_docs_set]

        return dspy.Prediction(retrieved_docs=decomposition_docs + hop1_docs_dedup + hop2_docs_dedup + hop3_docs_dedup)


