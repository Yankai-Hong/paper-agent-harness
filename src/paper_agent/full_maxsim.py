"""Deterministic ranking for complete page-level MaxSim scores."""
import math


def rank_pages(page_ids, scores):
    if len(page_ids) != len(scores) or len(page_ids) != len(set(page_ids)):
        raise ValueError('Page and score identities must be complete and unique')
    if any(not isinstance(page, str) or not page for page in page_ids) or any(not math.isfinite(float(x)) for x in scores):
        raise ValueError('Invalid page or MaxSim score')
    return sorted(range(len(page_ids)), key=lambda i: (-float(scores[i]), page_ids[i]))
