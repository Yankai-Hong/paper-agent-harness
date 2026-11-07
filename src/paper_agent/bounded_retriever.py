"""Opt-in bounded page search around an existing Chroma-backed Retriever."""
import time

from .bounded_search import search_grouped


class BoundedPageRetriever:
    """Retrieve pages without changing the existing chunk/citation pipeline."""

    def __init__(self, retriever):
        index = getattr(retriever, 'index', None)
        store = getattr(index, 'store', None)
        if store is None or retriever.encoder is None:
            raise ValueError('A Chroma-backed Retriever with an encoder is required')
        sources = {}
        for chunk in retriever.chunks:
            if not all(chunk.get(key) for key in ('chunk_id', 'paper_id', 'page_id')):
                raise ValueError('Every chunk must have an explicit paper and page ID')
            if chunk['chunk_id'] in sources:
                raise ValueError('Duplicate chunk ID')
            sources[chunk['chunk_id']] = (chunk['paper_id'], chunk['page_id'])
        if len(sources) != store.manifest['count']:
            raise ValueError('Source count differs from vector collection')
        self.retriever, self.store, self.sources = retriever, store, sources

    def search(self, query, *, limit=20, offset=0, paper_ids=None, page_ids=None,
               initial_depth=300, max_depth=1000, exact_fallback=False):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Query must be nonempty')
        started = time.perf_counter()
        vector = self.retriever.encoder.encode([query])
        if len(vector) != 1:
            raise ValueError('Encoder returned an unexpected query count')
        encoded = time.perf_counter()
        result = search_grouped(self.store, vector[0], self.sources, limit=limit, offset=offset,
                                paper_ids=paper_ids, page_ids=page_ids, initial_depth=initial_depth,
                                max_depth=max_depth, exact_fallback=exact_fallback)
        result['query'] = query
        result['timing_ms'] = {'encode': (encoded - started) * 1000, 'search': (time.perf_counter() - encoded) * 1000}
        result['evidence_scope'] = 'page candidates; resolve chunk citation from authoritative source separately'
        return result
