import re

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate
from .RAGQAArenaTech_utils import GenerateSearchQuery

# Leaked template placeholders the reasoning model occasionally echoes verbatim
# instead of producing real content (seen as full-credit losses in traces):
# e.g. "{response}", "{reasoning}", "(actual answer)", "[response]", "(contents)".
_CURLY_PLACEHOLDER_RE = re.compile(r"^\s*\{\s*[a-zA-Z_][\w]*\s*\}\s*$")
_BRACKETED_RE = re.compile(r"^\s*[\[(<]\s*[a-zA-Z_][\w\s]*\s*[\])>]\s*$")
_SLOT_WORD_RE = re.compile(
    r"\b(answer|answers|response|responses|output|result|results|reasoning|"
    r"placeholder|insert|contents|content|summary|details|description|snippet|"
    r"text|body|value)\b",
    re.I,
)

# Vocabulary of short context-referral / placeholder fragments the reasoning
# model sometimes emits INSTEAD of the real answer -- pointing back at the
# retrieved context rather than stating it (seen as full-credit losses in
# traces): e.g. "Above", "See above", "As noted in the passage", "N/A",
# "None". A fragment is degenerate only when EVERY one of its tokens is in this
# set, so genuine short answers ("Use cron.", "Yes.") are never matched --
# they carry real-content tokens not listed here.
_ECHO_VOCAB = {
    "above", "below", "see", "refer", "to", "back", "per", "cf", "c.f",
    "as", "noted", "stated", "described", "shown", "mentioned", "seen",
    "the", "a", "an", "in", "from", "of", "at", "on", "for", "and", "or",
    "passage", "passages", "context", "document", "doc", "source", "sources",
    "reference", "references", "link", "links", "n/a", "na", "none", "null",
    "nil", "todo", "tbd", "placeholder", "example", "idk", "unknown",
    "result", "answer", "response", "output", "text",
}
_WORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'/-]*|[0-9]+")


def _is_echo_or_referral(s: str) -> bool:
    """True for a short fragment that only refers to/echoes the context.

    Applied only to short strings (real answers are full sentences), this
    catches "Above", "See above", "As noted in the passage", "N/A", "None",
    "...", etc. -- bare referrals and placeholders with no actual content.
    """
    if len(s) > 40:
        return False
    toks = _WORD_TOKEN_RE.findall(s)
    if not toks:
        # Pure punctuation/symbols, e.g. "..." "-" "??" -- no content at all.
        return True
    return all(t.lower() in _ECHO_VOCAB for t in toks)


def _is_degenerate(text) -> bool:
    """True if the LM emitted an empty string, a leaked template placeholder,
    or a bare context-referral fragment instead of a real answer."""
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
    # A bare context-referral / placeholder fragment with no real content
    # (e.g. "Above", "See above", "N/A", "None"). These score as full-credit
    # losses; recover them via the rewrite path.
    if _is_echo_or_referral(s):
        return True
    return False


# Leaked DSPy output delimiters (e.g. "[[ ## completed ## ]]") that sometimes
# bleed into the response field, and passage-citation references the model
# emits despite the prompt's prohibition (e.g. "passage [9]", "passage 3",
# "snippet [1]", "Passage 1", "passages [10], [12]"). Stripping these is pure
# cleanup -- they never belong in the scored response.
_DSPY_MARKER_RE = re.compile(r"\[\[ ## \w+ ## \]\]")

# Parenthesized passage/snippet citations: "(passage 3)", "(see passage [9])",
# "(as discussed in passage [8])", "(passages [10], [12])".
_CITATION_PAREN_RE = re.compile(
    r"\s*\((?:see |as discussed in |as per |in )?"
    r"(?:passages?|snippets?|sources?) "
    r"(?:\[\d+\](?:, ?\[\d+\])*|\d+(?:, ?\d+)*)\)",
    re.I,
)

# Inline passage/snippet citations (not in parentheses): "passage [9]",
# "passage 3", "Passage 1 mentions", "see passage [9]".
_CITATION_INLINE_RE = re.compile(
    r"\s*(?:see |as discussed in |as per )?"
    r"(?:passages?|snippets?) "
    r"(?:\[\d+\](?:, ?\[\d+\])*|\d+)",
    re.I,
)


def _clean_response(text):
    """Strip leaked DSPy delimiters and passage-citation references."""
    if not text:
        return text
    s = str(text)
    s = _DSPY_MARKER_RE.sub("", s)
    s = _CITATION_PAREN_RE.sub("", s)
    s = _CITATION_INLINE_RE.sub("", s)
    return s.strip()


class GenerateAnswer(dspy.Signature):
    """Answer the user's question using the retrieved context.

    - Answer the SPECIFIC question being asked, using the question's own framing
      and scope. Do not reinterpret a narrow question as a general one (e.g. if it
      asks about a particular feature, icon, or behavior, answer about that,
      not the broad topic it happens to mention). If a term in the question has
      several senses across domains, use the retrieved context to identify which
      sense the context actually addresses and answer THAT sense -- do not silently
      substitute a different but related question from another domain.
    - Give a clear, direct answer. State the correct answer plainly; only present
      competing views when the topic is genuinely contested, rather than hedging
      with "there is no single definitive answer".
    - For conditional questions (e.g. "can I...", "is it possible to..."), if the
      retrieved context states that the answer depends on a specific condition
      (certain hardware, drivers, OS/app versions) or that it is sometimes or
      frequently impossible, include that condition as part of the answer. Do not
      give an unqualified "yes"/"no" when the context qualifies it. Include ONLY
      conditions the context actually states; never invent limitations or caveats.
    - Prefer a FOCUSED, reliable answer over an exhaustive one. Lead with the most
      direct, well-supported method or answer to the question as stated; a single
      canonical command or method is better than a longer list. Only enumerate
      multiple distinct methods/tools/options when BOTH are true: the question
      explicitly asks for options or "all ways", AND the retrieved context clearly
      and reliably supports each one. Do NOT pad the answer with every vaguely
      related tool, obscure web service, or tangential workaround the context
      happens to mention -- omit marginal, dubious, or low-quality extras even when
      they appear in the context, because an untruthful or unreliable specific is
      judged worse than a missing one. When a method has exact command syntax the
      context gives, state it verbatim. Truthfulness and reliability always take
      priority over completeness.
    - Ground every claim in the retrieved context and do not contradict it. Do
      NOT invent or extrapolate specifics the context does not state -- exact
      version numbers, file paths, configuration mechanisms, absolute claims such as
      "impossible"/"always", OR plausible-but-unsupported workarounds, third-party
      tools, and extra steps the context does not mention. Be SELECTIVE about which
      context content to include: not everything the passages mention belongs in the
      answer -- prefer the canonical, mainstream, well-corroborated method and drop
      peripheral, one-off, or low-quality suggestions even if the context cites them.
      When unsure whether a detail is supported, OMIT it: a focused, faithful answer
      is preferred over one padded with marginal possibilities, and an untruthful
      specific is penalized more harshly than a missing one. Match the STRENGTH of
      your claim to the context: if it says "superseded"/"preferred"/"may", do not
      escalate to
      "deprecated"/"must"/"will". If the context ties a command or setting to a
      specific OS/app version and the question is about a different version, do not
      assert it applies -- say the context does not confirm it for that version. Do
      not mirror a one-sided framing in the retrieved passages as established fact --
      give the balanced, mainstream-correct answer. When a directly relevant
      passage is missing, do NOT refuse or say "the context does not contain..."
      -- answer as accurately as you can using only what the context supports plus
      well-established general knowledge.
    - For requests that solicit methods to attack, hijack, break, or otherwise harm
      systems or people, give a brief, high-level account of the general principles
      and the defensive context (why it matters, how it is mitigated) -- do NOT give
      a detailed step-by-step actionable playbook. Describe what is conceptually
      possible without enumerating concrete exploit steps.
    - Write a clear, self-contained answer in natural prose, using short lists or
      commands only when the question calls for them. Do not add citation markers
      or passage references (e.g. [1], "(source 1)", "passage [9]", "snippet [1]",
      "Passage 1"); write plain prose with no source references at all.
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

    def _safe_query(self, hop, context, question):
        """Generate a search query, tolerating an empty/unparseable LM output.

        The reasoning LM occasionally emits an empty body (e.g. "{}") that
        DSPy's JSON adapter cannot parse, raising AdapterParseError before a
        query is produced. Falling back to the raw question keeps retrieval
        working for this hop instead of crashing the whole example (a crash
        scores 0 on a row that retrieval could otherwise have answered).
        """
        try:
            q = self.generate_query[hop](context=context, question=question).query
        except Exception:
            q = question
        # Strip leaked DSPy delimiters (e.g. "[[ ## completed ## ]]") that the
        # reasoning model occasionally appends to the query field; these
        # pollute the dense-retrieval embedding and degrade result relevance.
        return _clean_response(q)

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self._safe_query(hop, context, question)
            passages = self.search(query, k=self.num_docs)
            # First hop: also search with the raw question so the original
            # terms (proper nouns, product names, exact error strings) always
            # reach the retriever.  The generated query may rephrase or drop
            # critical terms, causing retrieval misses; the raw-question
            # supplementary search is a safety net that costs one extra
            # retrieval call but substantially improves recall on questions
            # whose own wording matches the corpus best.
            if hop == 0 and query.strip() != question.strip():
                raw_passages = self.search(question, k=self.num_docs)
                passages = deduplicate(passages + raw_passages)
            context = deduplicate(context + passages)
        # The main answer predictor (ChainOfThought) can raise AdapterParseError
        # when the LM returns an empty/unparseable body (e.g. "{}"), which aborts
        # the example and scores 0. Retry once (caching is disabled, so this is a
        # fresh LM call); if it still fails, synthesize an empty prediction so the
        # degenerate-output guard below rewrites a best-effort answer from the
        # context rather than losing the whole row.
        try:
            pred = self.respond(context=context, question=question)
        except Exception:
            try:
                pred = self.respond(context=context, question=question)
            except Exception:
                pred = dspy.Prediction(reasoning="", response="")
        # Guard against degenerate LM outputs (an empty response or a leaked
        # template placeholder such as "{response}"): recover the answer by
        # rewriting it from the (usually valid) reasoning + context. These
        # failures otherwise score as full-credit losses.
        if _is_degenerate(getattr(pred, "response", None)):
            reasoning = getattr(pred, "reasoning", "")
            if _is_degenerate(reasoning):
                reasoning = ""
            try:
                rewritten = self.rewrite(
                    context=context, question=question, reasoning=reasoning or ""
                )
                pred.response = rewritten.response
            except Exception:
                # The repair call itself failed to parse: leave whatever the
                # predictor produced (possibly empty). We never let a parse
                # exception escape -- a low-scored row is still better than a
                # crash that aborts scoring, and the rewrite path rarely fails
                # because it uses the simpler dspy.Predict adapter.
                if _is_degenerate(getattr(pred, "response", None)):
                    pred.response = ""
        # Strip leaked DSPy delimiters and passage-citation references from the
        # final response (pure cleanup -- they never belong in the scored text).
        resp = getattr(pred, "response", None)
        if resp:
            pred.response = _clean_response(resp)
        # Carry the retrieved passages on the prediction so downstream metrics
        # (e.g. faithfulness/groundedness) can see the evidence the answer was
        # generated from. Mirrors hover_program.py's dspy.Prediction(retrieved_docs=...).
        pred.context = context
        return pred
