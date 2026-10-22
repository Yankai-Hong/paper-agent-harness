"""Score saved task judgments without an LLM call.

Correctness/support judgments must be independently supplied by reviewers.
These functions NEVER infer semantic correctness from citation presence.
"""


def rate(correct, total):
    return {"correct": correct, "total": total, "rate": correct / total if total else None}


def classification(gold, predicted):
    if not gold:
        raise ValueError("Empty classification gold")
    if set(predicted) - set(gold):
        raise ValueError("Predictions contain unknown IDs")
    labels = sorted(set(gold.values()))
    confusion = {label: {p: 0 for p in labels + ["missing_or_invalid"]} for label in labels}
    for identifier, label in gold.items():
        p = predicted.get(identifier)
        confusion[label][p if p in labels else "missing_or_invalid"] += 1
    result = rate(sum(predicted.get(i) == g for i, g in gold.items()), len(gold))
    result["confusion"] = confusion
    return result


def boundary(gold, predicted):
    if not set(gold.values()) <= {"answer", "refuse", "clarify"}:
        raise ValueError("Boundary labels must be answer/refuse/clarify")
    overall = classification(gold, predicted)
    true_refuse = sum(g == "refuse" and predicted.get(i) == "refuse" for i, g in gold.items())
    issued_refuse = sum(predicted.get(i) == "refuse" for i in gold)
    required_refuse = sum(g == "refuse" for g in gold.values())
    false_refuse = sum(g == "answer" and predicted.get(i) == "refuse" for i, g in gold.items())
    return {"decision_accuracy": overall, "refusal_precision": rate(true_refuse, issued_refuse),
            "refusal_recall": rate(true_refuse, required_refuse),
            "false_refusal_rate": rate(false_refuse, sum(g == "answer" for g in gold.values()))}


def pipeline_success(gold_ids, judgments):
    if not gold_ids or len(set(gold_ids)) != len(gold_ids):
        raise ValueError("Need unique gold turn IDs")
    required = ("behavior_correct", "answer_correct", "citations_correct", "no_unrecovered_error")
    for row in judgments.values():
        if any(key not in row or not isinstance(row[key], bool) for key in required):
            raise ValueError("Explicit independent boolean judgments required")
    passed = sum(i in judgments and all(judgments[i][k] for k in required) for i in gold_ids)
    return rate(passed, len(gold_ids))


def memory_consistency(expected, observed):
    if not expected:
        raise ValueError("No memory checkpoints")
    return rate(sum(observed.get(i) == value for i, value in expected.items()), len(expected))


def token_reduction(baseline, compressed):
    if not baseline or set(baseline) != set(compressed):
        raise ValueError("Token comparison must use the same nonempty session IDs")
    # Anthropic input_tokens exclude cache input fields: include each category once.
    fields = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    def total(rows):
        for row in rows.values():
            if any(not isinstance(row.get(k, 0), int) or row.get(k, 0) < 0 for k in fields):
                raise ValueError("Actual nonnegative token counts required")
        return sum(sum(row.get(k, 0) for k in fields) for row in rows.values())
    before, after = total(baseline), total(compressed)
    return {"baseline_tokens": before, "compressed_tokens": after,
            "reduction": 1-after/before if before else None,
            "note": "Include every NLU/reflect/generate/summary/retry call in input rows."}
