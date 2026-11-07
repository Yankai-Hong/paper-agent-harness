"""Fuse already deduplicated evidence rankings at the page identity level."""

import math


def fuse_page_rankings(rankings, weights, *, k=10):
    """Return pages ordered by weighted RRF, counting each page once per lane."""
    if len(rankings) != len(weights) or not rankings or not math.isfinite(k) or k <= 0:
        raise ValueError("Invalid page fusion configuration")
    scores = {}
    for ranking, weight in zip(rankings, weights):
        if not math.isfinite(weight) or weight < 0:
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


def select_page_evidence(lanes, weights, *, k=10):
    """Rank pages while retaining the independently supplied evidence and ranks.

    Each lane is a list of dictionaries with paper_id, chunk_id, and an
    optional page_id. A page_id is usable only when its source mapping was
    verified by the caller (page_mapping == 'verified'). Unknown locations
    stay separate, even when they have the same paper or evidence ID. This
    function does not establish that items on a page support one another.
    """
    if len(lanes) != len(weights) or not lanes or not math.isfinite(k) or k <= 0:
        raise ValueError('Invalid page fusion configuration')
    groups = {}
    for lane_index, (lane, weight) in enumerate(zip(lanes, weights)):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Page fusion weights must be finite and nonnegative')
        if not weight:
            continue
        seen = set()
        for original_rank, item in enumerate(lane, 1):
            if not isinstance(item, dict) or not all(isinstance(item.get(x), str) and item[x] for x in ('paper_id', 'chunk_id')):
                raise ValueError('Evidence requires paper_id and chunk_id')
            page = item.get('page_id')
            verified = item.get('page_mapping') == 'verified' and isinstance(page, str) and bool(page)
            key = ('page', item['paper_id'], page) if verified else ('unknown', lane_index, original_rank)
            group = groups.setdefault(key, {'paper_id': item['paper_id'], 'page_id': page if verified else None,
                                             'page_mapping': 'verified' if verified else 'unknown',
                                             'score': 0.0, 'evidence': []})
            group['evidence'].append({'lane': lane_index, 'rank': original_rank, 'item': dict(item)})
            if key not in seen:
                seen.add(key)
                group['score'] += weight / (k + len(seen))
    return sorted(groups.values(), key=lambda group: -group['score'])
