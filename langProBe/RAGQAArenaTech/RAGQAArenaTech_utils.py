import dspy


class GenerateSearchQuery(dspy.Signature):
    """Write a concise search query (4-12 words) to find passages that help
    answer the question.

    Extract the MOST specific technical terms from the question: exact command
    names, tool names, API names, file paths, error strings, protocol names, and
    version numbers. Quote error messages or code verbatim when present. Prefer
    concrete terms over generic phrases (e.g., "rsync --checksum behavior" rather
    than "file sync tool"; "ext4 journal mode" rather than "linux filesystem").

    If context passages are already provided, formulate the query to find
    information that is still MISSING -- the angle or detail not yet covered by
    those passages.

    Output a single search query string, NOT a question."""

    context = dspy.InputField(desc="may contain relevant facts; empty on the first hop")
    question = dspy.InputField()
    query = dspy.OutputField()


class GenerateAnswer(dspy.Signature):
    """Answer the user's question with a thorough, well-organized long-form answer.

    The retrieved context passages are the primary evidence, but you may also draw
    on your own knowledge to make the answer complete and accurate.
    - Truthfulness first: every claim must be correct. Never fabricate statistics,
      specifics, or facts you are not confident about.
    - Do NOT refuse merely because the context is incomplete -- give the best
      correct, complete answer you can, using the context where it helps.
    - Stay tightly scoped to the user's actual scenario. If the question describes
      a concrete situation (a specific device, "my apps", a single symptom), answer
      THAT scenario directly; do not reframe a focused question as a generic
      cross-platform or cross-domain survey.
    - Use only the parts of the retrieved context that DIRECTLY answer the
      user's literal question. Some passages elaborate a dimension ADJACENT to
      but not the same as what was asked (e.g. a topic's historical evolution
      when the question is why it is vulnerable today, or the most extreme
      cosmic/physical energy scale when the question is what is out of reach
      for humanity). Do NOT let such adjacent or more "extreme" retrieved
      context reframe the answer or supply its framing; stay anchored to the
      question's actual scope. When the question has multiple relevant levels,
      address each relevant level rather than collapsing to a single extreme
      framing pulled in from deeper context.
    - Lead with the concrete, named specifics present in the context -- exact
      tools, app names, commands, file paths, version numbers, and symptoms --
      before any generalization. Be complete within the question's scope: cover
      every relevant option, cause, or method, not just the first one.
    - Do NOT invent speculative "special requirements", "alternative methods", or
      extra rules that are not stated in the context or confidently known -- such
      plausible-sounding but unverified extras read as dubious specifics and lose
      to a concise truthful human answer. When uncertain, give the verified core
      rather than padding with guesses.
    - When a question asks about a single short label/abbreviation shown in a
      tool's output (e.g. a column name or flag), answer what it represents IN
      THAT tool's output, not a plausible-sounding alternate expansion of the
      same letters. Do not invent an alternate meaning just to produce a
      longer answer; the context's definition is authoritative.
    - Write the answer as a self-contained explanation to the user. Do NOT quote
      or cite passage numbers, and do NOT refer to "the context", "the passages",
      "passage [N]", or "the retrieved information" inside the answer; synthesize
      the evidence into your own prose, since the user never sees the passages.
    - Write a clear long-form answer with no preamble and no meta-commentary about
      the context or what you did.
    - Always write the answer in the SAME natural language as the user's question.
      If the question is in English, the answer MUST be in English -- do not
      switch to Chinese or any other language mid-answer, even for terms that
      have a well-known translation. The user reads English, so an answer in
      another language is unreadable to them regardless of technical accuracy.
    """

    context = dspy.InputField(desc="retrieved passages that may contain the answer")
    question = dspy.InputField()
    response = dspy.OutputField(
        desc="a complete, accurate, well-organized long-form answer"
    )
