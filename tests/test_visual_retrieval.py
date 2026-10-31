import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from paper_agent.storage import write_jsonl
from paper_agent.visual_evidence import validate_image
from paper_agent.visual_retrieval import VisualRetriever, build_visual_index


class Encoder:
    provenance = {"image": "fixture-image", "text": "fixture-text"}

    def encode_images(self, paths):
        return np.array([[1.0, 0.0] if "one" in str(p) else [0.0, 1.0] for p in paths])

    def encode_texts(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


def item(root, name, paper, number):
    path = root / name
    path.write_bytes(name.encode())
    return {"paper_id": paper, "chunk_id": name, "title": "test", "text": f"{number} caption", "caption": f"{number} caption",
            "evidence_type": "figure", "page": 1, "bbox": [1, 2, 10, 20], "page_size": [100, 200],
            "coordinate_system": "top-left-points", "page_rotation": 0, "figure_number": number,
            "image_path": name, "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_image_geometry_path_and_hash(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    e = item(root, "one.png", "p1", "Figure 1")
    assert validate_image(e, root) == root / "one.png"
    with pytest.raises(ValueError):
        validate_image({**e, "bbox": [1, 2, 101, 20]}, root)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError):
        validate_image({**e, "image_path": "../outside.png"}, root)
    with pytest.raises(ValueError):
        validate_image({**e, "image_sha256": "0" * 64}, root)


def test_visual_index_filter_tampering_and_no_overwrite(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    records = [item(root, "one.png", "p1", "Figure 1"), item(root, "two.png", "p2", "Figure 2")]
    evidence = tmp_path / "evidence.jsonl"
    write_jsonl(evidence, records)
    index = tmp_path / "index"
    encoder = Encoder()
    build_visual_index(evidence, index, root, encoder)
    with pytest.raises(ValueError, match="overwrite"):
        build_visual_index(evidence, index, root, encoder)
    retriever = VisualRetriever(index, root, encoder)
    assert retriever.search_text("测试")['hits'][0]["paper_id"] == "p1"
    assert retriever.search_image(root / "two.png")['hits'][0]["paper_id"] == "p2"
    assert retriever.search_text("x", paper_ids=["p2"])['hits'][0]["paper_id"] == "p2"
    assert retriever.search_text("x", figure_number="Fig. 1")['hits'][0]["paper_id"] == "p1"
    assert retriever.search_caption("caption", paper_ids=["p1"])['hits'][0]["paper_id"] == "p1"
    assert retriever.search_caption("架构流程")['hits'] == []
    assert [h['paper_id'] for h in retriever.search_caption("1")['hits']] == ['p1']
    retriever.evidence[0]['text'] += ' secretcell'
    assert retriever.search_caption('secretcell')['hits'] == []
    (index / "visual.faiss").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        VisualRetriever(index, root, encoder)


def test_model_provenance_and_dimension_mismatch(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    evidence = tmp_path / "evidence.jsonl"
    write_jsonl(evidence, [item(root, "one.png", "p1", "Figure 1")])
    index = tmp_path / "index"
    build_visual_index(evidence, index, root, Encoder())
    different = Encoder()
    different.provenance = {"image": "other", "text": "fixture-text"}
    with pytest.raises(ValueError, match="provenance"):
        VisualRetriever(index, root, different)
    retriever = VisualRetriever(index, root, Encoder())
    retriever.encoder.encode_texts = lambda _: np.ones((1, 3))
    with pytest.raises(ValueError, match="dimension"):
        retriever.search_text("test")


def test_parse_failure_trace_is_metadata_only(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "docling_ingest.py"
    trace_dir = tmp_path / "trace"
    result = subprocess.run([sys.executable, str(script), "private-paper.pdf", "--paper-id", "../invalid",
                             "--image-root", str(tmp_path / "images"), "--output", str(tmp_path / "out.jsonl"),
                             "--trace-dir", str(trace_dir)], capture_output=True, text=True)
    assert result.returncode != 0
    traces = list(trace_dir.glob("*.json"))
    assert len(traces) == 1
    content = traces[0].read_text(encoding="utf-8")
    record = json.loads(content)
    assert record["stages"][0]["stage"] == "Parse"
    assert record["stages"][0]["status"] == "error"
    assert record["stages"][0]["error_type"] == "ValueError"
    assert "private-paper" not in content and "../invalid" not in content


def test_standalone_text_reranker_never_scores_table_cells():
    from paper_agent.retrieval import Retriever
    class Sparse:
        def get_scores(self, tokens): return [1.0, 0.5]
    class Dense:
        def search(self, vector, count): return np.array([[1.0, 0.5]]), np.array([[0, 1]])
        ntotal = 2
    class TextEncoder:
        def encode(self, texts): return np.array([[1.0, 0.0]], dtype=np.float32)
    class Ranker:
        def score(self, query, texts):
            self.texts = texts
            return [1.0] * len(texts)
    chunks = [{'paper_id':'p', 'chunk_id':'body', 'text':'Paper body'},
              {'paper_id':'p', 'chunk_id':'table', 'evidence_type':'table',
               'caption':'Table 1: Results', 'text':'Table 1: Results\n| SECRET_CELL |'}]
    r = Retriever.__new__(Retriever)
    r.chunks, r.by_id = chunks, {c['chunk_id']:c for c in chunks}
    r.bm25, r.index, r.encoder, r.reranker = Sparse(), Dense(), TextEncoder(), Ranker()
    r.metadata = {'corpus_hash':'fixture'}
    r.search('results', mode='rerank', top_k=2, candidate_k=2)
    assert 'Table 1: Results' in r.reranker.texts
    assert all('SECRET_CELL' not in text for text in r.reranker.texts)
