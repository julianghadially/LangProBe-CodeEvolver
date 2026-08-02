import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

MAX_RETRIEVED_DOCS = 21


class QueryExpansion(dspy.Signature):
    """Given a claim and a summary of the Wikipedia documents retrieved so far, generate
    diverse search queries to find the remaining documents needed to verify the claim.

    Each query must target a DIFFERENT entity, attribute, or relationship that is mentioned
    or implied in the claim but not yet covered by retrieved documents. Use the most specific
    entity names available (people, places, organizations, works, dates). The queries must be
    non-redundant with one another so that different queries retrieve different documents.
    """
    claim = dspy.InputField(desc="The claim to find supporting documents for")
    context = dspy.InputField(desc="Summary of the documents retrieved so far")
    queries: list[str] = dspy.OutputField(desc="A list of diverse, non-redundant search queries")


class DocReranker(dspy.Signature):
    """Below is a numbered list of candidate Wikipedia documents (title + snippet) retrieved
    for a claim. Select the documents most relevant to verifying the claim — i.e. documents
    about the entities mentioned or implied in the claim that could supply supporting facts.

    Prefer documents whose titles correspond to entities named in the claim, and ensure the
    selected set covers the distinct entities in the claim rather than many documents about a
    single entity. Return the 1-based index numbers of the most relevant documents, most
    relevant first.
    """
    claim = dspy.InputField(desc="The claim to find supporting documents for")
    candidate_passages = dspy.InputField(desc="Numbered list of candidate documents (1-based index, then title)")
    selected_indices: list[int] = dspy.OutputField(desc="1-based index numbers of the most relevant documents, most relevant first")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 10
        self.num_queries = 3
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.expand_queries = dspy.ChainOfThought(QueryExpansion)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")
        self.reranker = dspy.ChainOfThought(DocReranker)

    @staticmethod
    def _title(doc):
        return doc.split(" | ", 1)[0].strip().lower()

    @staticmethod
    def _round_robin_dedup(lists):
        seen = set()
        out = []
        max_len = max((len(lst) for lst in lists), default=0)
        for rank in range(max_len):
            for lst in lists:
                if rank < len(lst):
                    doc = lst[rank]
                    t = HoverMultiHop._title(doc)
                    if t not in seen:
                        seen.add(t)
                        out.append(doc)
        return out

    def _retrieve_many(self, queries):
        return [self.retrieve_k(q).passages for q in queries]

    def _rerank(self, claim, candidates):
        n = len(candidates)
        pool = candidates[:50] if n > 50 else candidates
        npool = len(pool)
        numbered = [f"{i + 1}. {pool[i].split(' | ', 1)[0]}" for i in range(npool)]
        try:
            res = self.reranker(claim=claim, candidate_passages="\n".join(numbered))
            indices = getattr(res, "selected_indices", None) or []
        except Exception:
            indices = []

        seen_idx = set()
        ordered = []
        for idx in indices:
            try:
                i = int(idx) - 1
            except (ValueError, TypeError):
                continue
            if 0 <= i < npool and i not in seen_idx:
                seen_idx.add(i)
                ordered.append(pool[i])

        ordered += [pool[i] for i in range(npool) if i not in seen_idx]
        ordered += candidates[npool:]

        seen_t = set()
        out = []
        for d in ordered:
            t = self._title(d)
            if t not in seen_t:
                seen_t.add(t)
                out.append(d)
        return out

    def forward(self, claim):
        hop1 = self.retrieve_k(claim).passages

        summary_1 = self.summarize1(claim=claim, passages=hop1).summary

        queries2 = self.expand_queries(claim=claim, context=summary_1).queries
        queries2 = [q for q in (queries2 or []) if q and q.strip()][: self.num_queries]
        if not queries2:
            queries2 = [claim]
        hop2_lists = self._retrieve_many(queries2)

        all_hop2 = [d for lst in hop2_lists for d in lst]
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=all_hop2
        ).summary

        queries3 = self.expand_queries(claim=claim, context=summary_2).queries
        queries3 = [q for q in (queries3 or []) if q and q.strip()][: self.num_queries]
        if not queries3:
            queries3 = [claim]
        hop3_lists = self._retrieve_many(queries3)

        retrieval_lists = [hop1] + hop2_lists + hop3_lists
        candidates = self._round_robin_dedup(retrieval_lists)

        if len(candidates) <= MAX_RETRIEVED_DOCS:
            final = candidates
        else:
            final = self._rerank(claim, candidates)[:MAX_RETRIEVED_DOCS]

        return dspy.Prediction(retrieved_docs=final)
