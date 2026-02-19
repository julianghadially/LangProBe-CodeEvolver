import dspy


def count_unique_docs(example):
    return len(set([fact["key"] for fact in example["supporting_facts"]]))


# Constraint: Do NOT return more than 21 documents for evaluation.
MAX_RETRIEVED_DOCS = 21


def discrete_retrieval_eval(example, pred, trace=None):
    gold_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [doc["key"] for doc in example["supporting_facts"]],
        )
    )
    found_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [c.split(" | ")[0] for c in pred.retrieved_docs[:MAX_RETRIEVED_DOCS]],
        )
    )
    return gold_titles.issubset(found_titles)


def verification_eval(example, pred, trace=None):
    """Evaluate verification quality on HoVer benchmark.

    Checks verdict accuracy (SUPPORTS vs REFUTES) by comparing the
    verification result's overall verdict with the gold label.

    Args:
        example: Dataset example with 'label' field (0=REFUTES, 1=SUPPORTS)
        pred: Prediction with 'verification' attribute
        trace: Optional trace (unused)

    Returns:
        Boolean indicating whether the prediction is correct
    """
    # Import here to avoid circular dependency
    from .hover_verifier_models import OverallVerdict

    if not hasattr(pred, "verification"):
        return False

    verification = pred.verification
    gold_label = example.label  # 0=REFUTES, 1=SUPPORTS

    # Map overall verdict to label
    if verification.overall_verdict == OverallVerdict.SUPPORTS:
        pred_label = 1
    elif verification.overall_verdict == OverallVerdict.REFUTES:
        pred_label = 0
    else:  # NOT_ENOUGH_INFO
        return False

    return pred_label == gold_label


def verification_detailed_eval(example, pred, trace=None):
    """Detailed evaluation with partial credit for verification quality.

    Scoring breakdown:
    - 50% for correct overall verdict
    - 25% for evidence quality (overlap with gold supporting facts)
    - 25% for sub-claim coverage (having 2-4 sub-claims)

    Args:
        example: Dataset example with 'label' and 'supporting_facts' fields
        pred: Prediction with 'verification' attribute
        trace: Optional trace (unused)

    Returns:
        Float score between 0.0 and 1.0
    """
    # Import here to avoid circular dependency
    from .hover_verifier_models import OverallVerdict

    if not hasattr(pred, "verification"):
        return 0.0

    verification = pred.verification
    gold_label = example.label
    score = 0.0

    # 1. Verdict accuracy (50%)
    if verification.overall_verdict == OverallVerdict.SUPPORTS and gold_label == 1:
        score += 0.5
    elif verification.overall_verdict == OverallVerdict.REFUTES and gold_label == 0:
        score += 0.5

    # 2. Evidence quality (25%) - check document overlap
    gold_doc_keys = set([fact["key"] for fact in example.supporting_facts])
    found_doc_keys = set()

    for evidence_list in verification.evidence_mapping.values():
        for evidence in evidence_list:
            # Normalize document ID for comparison
            found_doc_keys.add(dspy.evaluate.normalize_text(evidence.document_id))

    # Normalize gold keys too
    gold_doc_keys_normalized = set(
        dspy.evaluate.normalize_text(key) for key in gold_doc_keys
    )

    if gold_doc_keys_normalized:
        evidence_overlap = (
            len(gold_doc_keys_normalized & found_doc_keys)
            / len(gold_doc_keys_normalized)
        )
        score += 0.25 * evidence_overlap

    # 3. Sub-claim coverage (25%)
    num_subclaims = len(verification.sub_claims)
    if 2 <= num_subclaims <= 4:
        score += 0.25
    elif num_subclaims > 0:
        score += 0.125

    return score
