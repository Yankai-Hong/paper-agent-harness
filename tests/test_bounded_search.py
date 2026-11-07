import numpy as np
import pytest
from types import SimpleNamespace

from paper_agent.bounded_search import search_grouped
from paper_agent.bounded_retriever import BoundedPageRetriever
from paper_agent.vector_store import ChromaVectorStore


def test_real_chroma_group_scope_pagination_and_expansion(tmp_path):
    contract = {"collection": "paper_bounded_test", "dimension": 2, "normalization": "l2", "metric": "cosine"}
    store = ChromaVectorStore(tmp_path / "index", contract, create=True)
    values = np.array([[1, 0], [.99, .1], [.8, .6], [.6, .8], [0, 1]], dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    ids = ["a", "b", "c", "d", "e"]
    store.upsert(ids, values)
    store.seal(ids)
    mapping = {"a": ("paper1", "page1"), "b": ("paper1", "page1"),
               "c": ("paper1", "page2"), "d": ("paper2", "page3"), "e": ("paper2", "page4")}
    result = search_grouped(store, [1, 0], mapping, limit=1, offset=1, paper_ids=["paper1"],
                            initial_depth=2, max_depth=4)
    assert [item["page_id"] for item in result["results"]] == ["page2"]
    assert result["depths"] == [2, 4]
    assert result["satisfied"] and not result["exhaustive"] and result["ranking"] == "approximate"
    blocked = search_grouped(store, [1, 0], mapping, limit=1, paper_ids=["paper2"], initial_depth=2, max_depth=2)
    assert blocked["reason"] == "candidate_depth_limit" and not blocked["satisfied"]
    recovered = search_grouped(store, [1, 0], mapping, limit=1, paper_ids=["paper2"], initial_depth=2,
                               max_depth=2, exact_fallback=True)
    assert recovered["exhaustive"] and recovered["results"][0]["page_id"] == "page3"
    filtered = search_grouped(store, [1, 0], mapping, limit=1, page_ids=["page4"], initial_depth=5, max_depth=5)
    assert filtered["results"][0]["page_id"] == "page4"
    chunks = [{'chunk_id': key, 'paper_id': value[0], 'page_id': value[1]} for key, value in mapping.items()]
    encoder = SimpleNamespace(encode=lambda texts: np.array([[1., 0.]], dtype=np.float32))
    wrapper = BoundedPageRetriever(SimpleNamespace(index=SimpleNamespace(store=store), encoder=encoder, chunks=chunks))
    response = wrapper.search('actual query', limit=1, paper_ids=['paper2'], initial_depth=2, max_depth=2, exact_fallback=True)
    assert response['results'][0]['page_id'] == 'page3'
    assert response['evidence_scope'].startswith('page candidates') and response['exhaustive']
    with pytest.raises(ValueError, match="mapping count"):
        search_grouped(store, [1, 0], {"a": ("paper1", "page1")}, limit=1)


def test_incomplete_candidate_list_is_not_silently_accepted():
    class IncompleteStore:
        manifest = {'count': 3}
        def search(self, vector, depth):
            return [('a', .9)]
    sources = {'a': ('p', '1'), 'b': ('p', '2'), 'c': ('p', '3')}
    with pytest.raises(ValueError, match='incomplete'):
        search_grouped(IncompleteStore(), [1, 0], sources, limit=1, initial_depth=2, max_depth=2)
