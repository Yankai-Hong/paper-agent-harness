import numpy as np
import pytest

from paper_agent.colqwen_store import ColQwenPageStore, maxsim


def test_maxsim_uses_each_query_token_and_rejects_padding():
    query = np.eye(2, dtype=np.float32)
    document = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    assert maxsim(query, document) == pytest.approx(2.0)
    assert maxsim(query, [[-(2 ** -0.5), -(2 ** -0.5)]]) == pytest.approx(-(2 ** 0.5))
    with pytest.raises(ValueError, match="unpadded"):
        maxsim(query, [[0, 0]])


def test_real_chroma_page_tokens_roundtrip_and_candidate_scope(tmp_path):
    path = tmp_path / "tokens"
    contract = {"model_revision": "example-revision", "source_hash": "source"}
    store = ColQwenPageStore(path, create=True, contract=contract)
    doc_a = np.eye(2, dtype=np.float32)
    doc_b = np.array([[2 ** -0.5, 2 ** -0.5]], dtype=np.float32)
    store.add_page("page-a", doc_a, "a" * 64)
    store.add_page("page-b", doc_b, "b" * 64)
    with pytest.raises(ValueError, match="duplicate"):
        store.add_page("page-a", doc_a, "a" * 64)
    store.seal()
    with pytest.raises(ValueError, match="immutable"):
        store.add_page("page-c", doc_a, "c" * 64)
    reopened = ColQwenPageStore(path, contract={**contract, "dimension": 2})
    np.testing.assert_allclose(reopened.get_page("page-a"), doc_a)
    ordered = reopened.rerank(doc_a, ["page-b", "page-a"])
    assert [page for page, _ in ordered] == ["page-a", "page-b"]
    assert ordered[0][1] == pytest.approx(2)
    with pytest.raises(ValueError, match="unavailable"):
        reopened.rerank(doc_a, ["not-indexed"])
    with pytest.raises(ValueError, match="unique"):
        reopened.rerank(doc_a, ["page-a", "page-a"])
    with pytest.raises(ValueError, match="contract"):
        ColQwenPageStore(path, contract={"model_revision": "other"})


def test_unsealed_and_tampered_vectors_fail_closed(tmp_path):
    path = tmp_path / "tokens"
    store = ColQwenPageStore(path, create=True, contract={"model_revision": "r", "source_hash": "s"})
    store.add_page("page", [[1.0, 0.0]], "a" * 64)
    with pytest.raises(ValueError, match="not sealed"):
        ColQwenPageStore(path)
    store.seal()
    store.collection.update(ids=["page:000000"], embeddings=[[0.0, 1.0]])
    with pytest.raises(ValueError, match="content mismatch"):
        ColQwenPageStore(path).get_page("page")
