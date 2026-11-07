"""Fuse already deduplicated evidence rankings at the page identity level."""


def fuse_page_rankings(rankings, weights, *, k=10):
    """Return pages ordered by weighted RRF, counting each page once per lane."""
    if len(rankings) != len(weights) or not rankings or k <= 0:
        raise ValueError("Invalid page fusion configuration")
    scores = {}
    for ranking, weight in zip(rankings, weights):
        if weight < 0:
            raise ValueError("Page fusion weights must be nonnegative")
        if not weight:
            continue
        seen = set()
        for page in ranking:
            if not isinstance(page, str) or not page:
                raise ValueError("Page IDs must be nonempty strings")
            if page in seen:
                continue
            seen.add(page)
            scores[page] = scores.get(page, 0.0) + weight / (k + len(seen))
    return sorted(scores, key=lambda page: (-scores[page], page))
