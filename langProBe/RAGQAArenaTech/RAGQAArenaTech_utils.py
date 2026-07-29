import dspy


class GenerateSearchQuery(dspy.Signature):
    """Write two concise search queries that will retrieve passages answering the question.

    ``query`` -- the primary query: preserve the question's specific terms (product names,
    UI element names, error messages, domain vocabulary) so retrieval targets the intended
    feature or problem rather than a generic concept. Drop filler words but keep the nouns
    that disambiguate the user's actual intent.

    ``alt_query`` -- a SECOND, deliberately different query that targets a different
    plausible interpretation of the same question. Colloquial or ambiguous wording often
    has a domain-specific meaning distinct from its literal reading (e.g. "why are my apps
    all in the cloud?" most likely refers to the iOS offload cloud icon next to an app,
    not cloud computing; "what is the difference between flash and bootflash?" most likely
    refers to Cisco's bootflash term, not the Android fastboot commands). When the literal
    reading is generic but a concrete product / platform / feature reading is plausible,
    make alt_query name that concrete scenario (the product, the UI element, the setting,
    the vendor term). When the question is unambiguous, alt_query should still be a useful
    reformulation (synonyms / rewording) rather than a copy of ``query``. Never leave
    alt_query empty.
    """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField(desc="primary concise search query")
    alt_query = dspy.OutputField(desc="a different, concrete-interpretation search query")
