"""Progressive candidate retrieval with explicit approximate and scope status."""


def search_grouped(store, vector, sources, *, limit, offset=0, paper_ids=None,
                   page_ids=None, initial_depth=100, max_depth=1000, exact_fallback=False):
    """Find distinct pages from ranked chunks, optionally constrained by source.

    `sources` maps every chunk ID to a (paper_id, page_id) pair. Even when enough
    results are returned, bounded HNSW ranking is approximate, not exhaustive.
    Offset is applied after paper/page filtering and grouping, never before.
    """
    if (type(limit) is not int or limit < 1 or type(offset) is not int or offset < 0
            or type(initial_depth) is not int or initial_depth < 1
            or type(max_depth) is not int or max_depth < 1):
        raise ValueError("Invalid search bounds")
    total = store.manifest["count"]
    if len(sources) != total:
        raise ValueError("Source mapping count differs from collection")
    allowed_papers = None if paper_ids is None else set(paper_ids)
    allowed_pages = None if page_ids is None else set(page_ids)
    ceiling = min(total, max_depth)
    depth = min(ceiling, initial_depth)
    if not total:
        return {"results": [], "satisfied": False, "reason": "empty_collection", "depths": [], "exhaustive": True}
    depths = []
    last = []
    while True:
        try:
            found = store.search(vector, depth)
        except RuntimeError as exc:
            if not exact_fallback or 'contigious 2D array' not in str(exc):
                raise
            found = store.search(vector, total)
            depth = total
        if len(found) != depth or len({key for key, _ in found}) != len(found):
            raise ValueError('Chroma returned an incomplete or duplicate candidate list')
        if any(found[i][1] + 1e-6 < found[i + 1][1] for i in range(len(found) - 1)):
            raise ValueError('Chroma returned candidates out of score order')
        depths.append(depth)
        seen = set()
        last = []
        for chunk_id, similarity in found:
            if chunk_id not in sources:
                raise ValueError("Retrieved vector lacks source mapping")
            paper_id, page_id = sources[chunk_id]
            if allowed_papers is not None and paper_id not in allowed_papers:
                continue
            if allowed_pages is not None and page_id not in allowed_pages:
                continue
            key = (paper_id, page_id)
            if key not in seen:
                seen.add(key)
                last.append({"paper_id": paper_id, "page_id": page_id, "chunk_id": chunk_id, "score": similarity})
        if len(last) >= offset + limit or depth == total:
            break
        if depth >= ceiling:
            if exact_fallback and depth < total:
                depth = total
                continue
            break
        depth = min(ceiling, depth * 2)
    satisfied = len(last) >= offset + limit
    exhaustive = depth == total
    return {"results": last[offset:offset + limit], "satisfied": satisfied,
            "reason": "enough_candidates" if satisfied else ("exhausted_collection" if exhaustive else "candidate_depth_limit"),
            "depths": depths, "exhaustive": exhaustive, "search_mode": store.last_search_mode,
            "ranking": "exhaustive" if exhaustive else "approximate", "eligible_pages_seen": len(last)}
