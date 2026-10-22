"""FAISS integration with known vectors; these are NOT BGE effect measurements."""
import numpy as np
import pytest

from paper_agent.corpus import Chunker
from paper_agent.retrieval import build_index, Retriever
from paper_agent.storage import write_jsonl


class KnownVectorEncoder:
    model_id = "test-only-known-vectors"
    max_length = 768

    def encode(self, texts):
        values = np.array([[1, 0] if "apple" in t else [0, 1] for t in texts], dtype=np.float32)
        return values


def test_exact_dense_filter_fusion_and_rerank(tmp_path):
    papers = [{"paper_id": pid, "title": pid, "paragraphs": [{"text": text}]}
              for pid, text in [("p1", "apple fruit"), ("p2", "blue sky"), ("p3", "cloud white")]]
    chunks = [c for p in papers for c in Chunker().split(p)]
    write_jsonl(tmp_path/"chunks.jsonl", chunks)
    encoder = KnownVectorEncoder()
    build_index(tmp_path/"chunks.jsonl", tmp_path/"index", encoder)
    retriever = Retriever(tmp_path/"index", encoder)
    assert retriever.search("apple", "dense")["hits"][0]["paper_id"] == "p1"
    assert retriever.search("apple", "hybrid")["hits"][0]["paper_id"] == "p1"
    filtered = retriever.search("apple", "dense", paper_ids=["p2"])
    assert [h["paper_id"] for h in filtered["hits"]] == ["p2"]
    class KnownScores:
        def score(self, query, texts):
            return [100 if "sky" in t else 0 for t in texts]
    retriever.reranker = KnownScores()
    assert retriever.search("apple", "rerank")["hits"][0]["paper_id"] == "p2"
    encoder.model_id = "different"
    with pytest.raises(ValueError, match="match"):
        Retriever(tmp_path/"index", encoder)


def test_index_detects_corpus_corruption(tmp_path):
    chunks = Chunker().split({"paper_id": "p", "title": "p", "paragraphs": [{"text": "apple fruit"}]})
    write_jsonl(tmp_path/"chunks.jsonl", chunks)
    build_index(tmp_path/"chunks.jsonl", tmp_path/"index")
    chunks[0]["text"] = "tampered"
    write_jsonl(tmp_path/"index"/"chunks.jsonl", chunks)
    with pytest.raises(ValueError, match="hash mismatch"):
        Retriever(tmp_path/"index")
