"""Separate CLIP image index; no vectors from text retrieval enter this index."""
import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .storage import digest, read_jsonl, write_json, write_jsonl
from .visual_evidence import validate_image

IMAGE_MODEL = "sentence-transformers/clip-ViT-B-32"
TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


def _file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _number(value):
    value = re.sub(r"\s+", "", str(value or "").casefold())
    value = re.sub(r"^fig(?:ure)?\.?", "figure", value)
    return re.sub(r"^tab(?:le)?\.?", "table", value)


def _model_manifest(path):
    path = Path(path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("CLIP model must be a local snapshot directory")
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.name != "provenance.json" and ".cache" not in item.relative_to(path).parts:
            files.append((item.relative_to(path).as_posix(), _file_sha256(item)))
    if not files:
        raise ValueError("Empty CLIP snapshot")
    provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    complete = path / "snapshot.complete.json"
    if not complete.is_file() or json.loads(complete.read_text(encoding="utf-8")) != provenance:
        raise ValueError("CLIP snapshot download is incomplete")
    return {"model_id": provenance["model_id"], "revision": provenance["revision"], "files_hash": digest(files)}


def _normalize(vectors):
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid CLIP vectors")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero CLIP vector")
    return np.ascontiguousarray(values / norms)


class SubprocessEncoder:
    """Load only one model at a time in the isolated interpreter."""
    def __init__(self, python, image_model_path, text_model_path, device="cpu"):
        self.python = str(python)
        self.image_model_path, self.text_model_path = str(Path(image_model_path).resolve()), str(Path(text_model_path).resolve())
        self.device = device
        self.provenance = {"image": _model_manifest(self.image_model_path), "text": _model_manifest(self.text_model_path)}
        if self.provenance["image"]["model_id"] != IMAGE_MODEL or self.provenance["text"]["model_id"] != TEXT_MODEL:
            raise ValueError("CLIP snapshot model IDs do not match the required aligned pair")

    def _encode(self, kind, inputs):
        current = {"image": _model_manifest(self.image_model_path), "text": _model_manifest(self.text_model_path)}
        if current != self.provenance:
            raise ValueError("CLIP model snapshot changed after initialization")
        worker = Path(__file__).resolve().parents[2] / "scripts" / "clip_worker.py"
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "request.json"
            output = Path(tmp) / "vectors.npy"
            request.write_text(json.dumps({"kind": kind, "inputs": list(map(str, inputs)), "output": str(output),
                                           "model": self.image_model_path if kind == "images" else self.text_model_path,
                                           "device": self.device}, ensure_ascii=False), encoding="utf-8")
            subprocess.run([self.python, str(worker), "encode", str(request)], check=True, timeout=1800)
            return _normalize(np.load(output, allow_pickle=False))

    def encode_images(self, paths):
        return self._encode("images", paths)

    def encode_texts(self, texts):
        return self._encode("texts", texts)


def build_visual_index(evidence_path, directory, image_root, encoder):
    import faiss
    directory = Path(directory)
    if directory.exists():
        raise ValueError("Refusing to overwrite an existing visual index directory")
    evidence = [e for e in read_jsonl(evidence_path) if e.get("evidence_type") in {"figure", "table"}]
    keys = [(e.get("paper_id"), e.get("chunk_id")) for e in evidence]
    if not evidence or len(keys) != len(set(keys)):
        raise ValueError("Visual corpus must be nonempty with unique evidence keys")
    images = [validate_image(e, image_root) for e in evidence]
    vectors = _normalize(encoder.encode_images(images))
    if len(vectors) != len(evidence):
        raise ValueError("CLIP image count mismatch")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="visual-index-", dir=directory.parent) as temp:
        staging = Path(temp) / "index"
        staging.mkdir()
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(staging / "visual.faiss"))
        metadata = {"schema": 1, "corpus_hash": digest(evidence), "count": len(evidence), "dimension": vectors.shape[1],
                    "index_sha256": hashlib.sha256((staging / "visual.faiss").read_bytes()).hexdigest(),
                    "model_provenance": encoder.provenance, "normalization": "l2", "metric": "inner_product"}
        write_jsonl(staging / "evidence.jsonl", evidence)
        write_json(staging / "metadata.json", metadata)
        staging.rename(directory)
    return metadata


class VisualRetriever:
    def __init__(self, directory, image_root, encoder):
        import faiss
        self.directory, self.image_root, self.encoder = Path(directory), Path(image_root), encoder
        self.evidence = read_jsonl(self.directory / "evidence.jsonl")
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema") != 1 or self.metadata.get("corpus_hash") != digest(self.evidence):
            raise ValueError("Visual corpus version or hash mismatch")
        if encoder.provenance != self.metadata.get("model_provenance"):
            raise ValueError("CLIP model provenance differs from index")
        index_path = self.directory / "visual.faiss"
        if hashlib.sha256(index_path.read_bytes()).hexdigest() != self.metadata.get("index_sha256"):
            raise ValueError("Visual index checksum mismatch")
        self.index = faiss.read_index(str(index_path))
        if self.index.ntotal != len(self.evidence) or self.index.ntotal != self.metadata.get("count") or self.index.d != self.metadata.get("dimension"):
            raise ValueError("Visual index dimensions or count mismatch")

    def _search(self, vector, top_k, paper_ids, figure_number, start):
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be positive")
        vector = _normalize(vector)
        if vector.shape != (1, self.index.d):
            raise ValueError("CLIP query dimension mismatch")
        scores, positions = self.index.search(vector, self.index.ntotal)
        allowed = set(paper_ids) if paper_ids is not None else None
        hits = []
        for pos, score in zip(positions[0], scores[0]):
            e = self.evidence[int(pos)]
            if allowed is not None and e["paper_id"] not in allowed:
                continue
            if figure_number is not None and _number(e.get("figure_number")) != _number(figure_number):
                continue
            validate_image(e, self.image_root)
            hits.append({**e, "rank": len(hits) + 1, "visual_score": float(score)})
            if len(hits) == top_k:
                break
        return {"hits": hits, "timing_ms": {"visual_retrieve": (time.perf_counter() - start) * 1000},
                "corpus_hash": self.metadata["corpus_hash"]}

    def search_text(self, query, top_k=5, paper_ids=None, figure_number=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Empty text query")
        start = time.perf_counter()
        return self._search(self.encoder.encode_texts([query]), top_k, paper_ids, figure_number, start)

    def search_texts(self, queries, top_k=20, paper_ids=None, figure_number=None):
        """Encode original and rewritten text in one isolated CLIP process."""
        if not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError('Nonempty text queries required')
        start = time.perf_counter()
        vectors = _normalize(self.encoder.encode_texts(queries))
        if len(vectors) != len(queries):
            raise ValueError('CLIP text query count mismatch')
        return [self._search(vector.reshape(1, -1), top_k, paper_ids, figure_number, start) for vector in vectors]

    def search_image(self, image_path, top_k=5, paper_ids=None, figure_number=None):
        path = Path(image_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("Image query must be a file")
        start = time.perf_counter()
        return self._search(self.encoder.encode_images([path]), top_k, paper_ids, figure_number, start)

    def search_caption(self, query, top_k=5, paper_ids=None, figure_number=None):
        from rank_bm25 import BM25Okapi
        from .corpus import lexical_tokens
        if not isinstance(query, str) or not query.strip() or top_k < 1:
            raise ValueError("Nonempty query and positive top_k required")
        candidates = [e for e in self.evidence if isinstance(e.get("caption"), str) and e["caption"].strip()
                      and (paper_ids is None or e["paper_id"] in set(paper_ids))
                      and (figure_number is None or _number(e.get("figure_number")) == _number(figure_number))]
        if not candidates:
            return {"hits": []}
        query_tokens = lexical_tokens(query)
        tokenized = [lexical_tokens(e["caption"]) or ["__empty__"] for e in candidates]
        scores = BM25Okapi(tokenized).get_scores(query_tokens)
        # BM25 returns zero for every document if no query token appears. Such
        # ties are not caption matches and must not inject arbitrary entities
        # into RRF. Do not filter by score: BM25 IDF can legitimately be negative.
        query_terms = set(query_tokens)
        matching = [(e, score) for e, score, words in zip(candidates, scores, tokenized)
                    if query_terms.intersection(words)]
        ranked = sorted(matching, key=lambda pair: (-float(pair[1]), pair[0]["chunk_id"]))[:top_k]
        return {"hits": [{**e, "rank": i + 1, "caption_score": float(s)} for i, (e, s) in enumerate(ranked)]}
