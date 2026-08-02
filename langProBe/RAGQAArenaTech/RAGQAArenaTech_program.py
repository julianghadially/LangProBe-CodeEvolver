import re

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate
from .RAGQAArenaTech_utils import GenerateSearchQuery

# Leaked template placeholders the reasoning model occasionally echoes verbatim
# instead of producing real content (seen as full-credit losses in traces):
# e.g. "{response}", "{reasoning}", "(actual answer)", "[response]".
_CURLY_PLACEHOLDER_RE = re.compile(r"^\s*\{\s*[a-zA-Z_][\w]*\s*\}\s*$")
_BRACKETED_RE = re.compile(r"^\s*[\[(<]\s*[a-zA-Z_][\w\s]*\s*[\])>]\s*$")
_SLOT_WORD_RE = re.compile(
    r"\b(answer|response|output|result|reasoning|placeholder|insert)\b", re.I
)


def _is_degenerate(text) -> bool:
    """True if the LM emitted an empty string or a leaked template placeholder."""
    if text is None:
        return True
    s = str(text).strip()
    if not s:
        return True
    # Unambiguous template slot: a single {identifier} such as "{response}".
    if _CURLY_PLACEHOLDER_RE.match(s):
        return True
    # A short bracketed/parenthetical/angle phrase naming a slot, e.g.
    # "(actual answer)", "[response]", "<answer>".
    if len(s) <= 40 and _BRACKETED_RE.match(s) and _SLOT_WORD_RE.search(s):
        return True
    return False


class GenerateAnswer(dspy.Signature):
    """Answer the user's question using the retrieved context.

    - Answer the SPECIFIC question being asked, using the question's own framing
      and scope. Do not reinterpret a narrow question as a general one (e.g. if it
      asks about a particular feature, icon, or behavior, answer about that,
      not the broad topic it happens to mention).
    - Give a clear, direct answer. State the correct answer plainly; only present
      competing views when the topic is genuinely contested, rather than hedging
      with "there is no single definitive answer".
    - Be complete: cover every relevant fact, method, alternative, and caveat
      that bears on the question. When several approaches or details are
      relevant, mention all of them rather than just one.
    - Treat the retrieved context as your primary evidence and do not contradict
      it. When it contains relevant passages, ground your answer in them and
      avoid unsupported claims. When a directly relevant passage is missing, do
      NOT refuse or say "the context does not contain..." -- answer the question
      as accurately as you can.
    - Write a clear, self-contained answer in natural prose, using short lists or
      commands only when the question calls for them.
    - Output the actual answer text. Never output a placeholder such as
      "{response}", "(actual answer)", or an empty answer.
    """

    context = dspy.InputField(desc="retrieved passages that may help answer the question")
    question = dspy.InputField()
    response = dspy.OutputField(
        desc="a complete, faithful answer grounded in the retrieved context"
    )


class RewriteResponse(dspy.Signature):
    """Produce the final answer to the question from the retrieved context and the draft reasoning.

    Output ONLY the final answer text directly. Do not output placeholders or
    template strings (e.g. "{response}"); output the real answer.
    """

    context = dspy.InputField(desc="retrieved passages that may help answer the question")
    question = dspy.InputField()
    reasoning = dspy.InputField(desc="draft reasoning toward the answer (may be empty)")
    response = dspy.OutputField(
        desc="the complete, faithful final answer grounded in the context"
    )


class SimplifiedBaleen(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi-hop RAG program for RAGQAArenaTech.

    Retrieval is injected (see ``RAGQAArenaTech_retrieval.HTTPEmbeddingRetriever``):
    the program holds no corpus/index and performs no IO -- it calls out to a
    separate warm retriever server. This keeps the program pure logic the optimizer
    can freely evolve (hops, query generation, answer synthesis), while the heavy
    3.7GB embedding index lives in a process loaded once, outside the eval.
    """

    def __init__(self, retriever, num_docs=8, max_hops=2):
        super().__init__()
        self.retriever = retriever
        self.max_hops = max_hops
        self.num_docs = num_docs
        self.respond = dspy.ChainOfThought(GenerateAnswer)
        # Repair predictor used only when the main answer leaks a placeholder or
        # comes back empty -- it rewrites the response from the (usually valid)
        # reasoning + context. No extra LM cost on the common (valid) path.
        self.rewrite = dspy.Predict(RewriteResponse)
        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(self.max_hops)
        ]

    def search(self, query, k=5):
        return self.retriever.search(query, k=k)

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.search(query, k=self.num_docs)
            context = deduplicate(context + passages)
        pred = self.respond(context=context, question=question)
        # Guard against degenerate LM outputs (an empty response or a leaked
        # template placeholder such as "{response}"): recover the answer by
        # rewriting it from the (usually valid) reasoning + context. These
        # failures otherwise score as full-credit losses.
        if _is_degenerate(getattr(pred, "response", None)):
            reasoning = getattr(pred, "reasoning", "")
            if _is_degenerate(reasoning):
                reasoning = ""
            rewritten = self.rewrite(
                context=context, question=question, reasoning=reasoning or ""
            )
            pred.response = rewritten.response
        # Carry the retrieved passages on the prediction so downstream metrics
        # (e.g. faithfulness/groundedness) can see the evidence the answer was
        # generated from. Mirrors hover_program.py's dspy.Prediction(retrieved_docs=...).
        pred.context = context
        return pred
