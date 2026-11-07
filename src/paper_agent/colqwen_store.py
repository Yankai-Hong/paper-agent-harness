"""Store ColQwen page token vectors separately and score candidate pages by MaxSim.

The caller must remove padding according to the model's attention mask before
adding vectors. Page tokens include any non-padding prompt tokens produced by
the encoder, not merely image patches. This module does not run a model.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


def _matrix(value, dimension=None):
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("Expected a nonempty two-dimensional token matrix")
    if dimension is not None and matrix.shape[1] != dimension:
        raise ValueError("Token dimension mismatch")
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite token vector")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms < 1e-6) or not np.allclose(norms, 1.0, atol=2e-3):
        raise ValueError("Expected normalized, unpadded token vectors")
    return np.ascontiguousarray(matrix, dtype="<f4")


def _digest(matrix):
    return hashlib.sha256(matrix.tobytes(order="C")).hexdigest()


def maxsim(query, document):
    """Sum each query token's maximum dot product over document tokens."""
    q = _matrix(query)
    d = _matrix(document, q.shape[1])
    return float(np.max(q @ d.T, axis=1).sum(dtype=np.float64))


class ColQwenPageStore:
    """Append-only page tokens until sealed; a sealed snapshot is read-only."""

    def __init__(self, directory, *, create=False, contract=None):
        import chromadb
        from chromadb.config import Settings

        self.root = Path(directory)
        self.manifest = self.root / "pages.json"
        self.closed = not create
        if create:
            if self.root.exists():
                raise ValueError("Refusing to overwrite a page token index")
            if not isinstance(contract, dict) or not contract.get("model_revision") or not contract.get("source_hash"):
                raise ValueError("Model revision and source hash are required")
            self.root.mkdir(parents=True)
            self.contract = dict(contract)
            self.pages = {}
            self.client = chromadb.PersistentClient(path=str(self.root / "chroma"), settings=Settings(anonymized_telemetry=False, allow_reset=False))
            self.collection = self.client.create_collection("paper_colqwen_page_tokens", metadata={"hnsw:space": "cosine"})
        else:
            if not self.manifest.is_file():
                raise ValueError("Page token index is not sealed")
            data = json.loads(self.manifest.read_text(encoding="utf-8"))
            if data.get("schema") != 1 or not isinstance(data.get("pages"), dict):
                raise ValueError("Invalid page token manifest")
            self.contract, self.pages = data["contract"], data["pages"]
            if contract is not None and contract != self.contract:
                raise ValueError("Page token index contract mismatch")
            self.client = chromadb.PersistentClient(path=str(self.root / "chroma"), settings=Settings(anonymized_telemetry=False, allow_reset=False))
            self.collection = self.client.get_collection("paper_colqwen_page_tokens")
            if self.collection.count() != sum(p["count"] for p in self.pages.values()):
                raise ValueError("Page token count mismatch")

    def add_page(self, page_id, vectors, image_sha256):
        if self.closed:
            raise ValueError("Sealed page token index is immutable")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", page_id) or page_id in self.pages:
            raise ValueError("Invalid or duplicate page ID")
        if not re.fullmatch(r"[0-9a-f]{64}", image_sha256):
            raise ValueError("A valid source image SHA-256 is required")
        matrix = _matrix(vectors, self.contract.get("dimension"))
        self.contract.setdefault("dimension", matrix.shape[1])
        for offset in range(0, len(matrix), 1000):
            part = matrix[offset:offset + 1000]
            indices = range(offset, offset + len(part))
            self.collection.add(ids=[f"{page_id}:{i:06d}" for i in indices], embeddings=part.tolist(),
                                metadatas=[{"page_id": page_id, "token_index": i, "image_sha256": image_sha256} for i in indices])
        self.pages[page_id] = {"count": len(matrix), "image_sha256": image_sha256, "vectors_sha256": _digest(matrix)}

    def seal(self):
        if self.closed or not self.pages:
            raise ValueError("Cannot seal empty or closed token index")
        data = {"schema": 1, "contract": self.contract, "pages": self.pages}
        with self.manifest.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2)
        self.closed = True

    def get_page(self, page_id):
        if not self.closed or page_id not in self.pages:
            raise ValueError("Page token index is unsealed or page is unavailable")
        meta = self.pages[page_id]
        ids = [f"{page_id}:{i:06d}" for i in range(meta["count"])]
        raw = self.collection.get(ids=ids, include=["embeddings", "metadatas"])
        if len(raw["ids"]) != len(ids):
            raise ValueError("Page token count mismatch")
        by_id = dict(zip(raw["ids"], zip(raw["embeddings"], raw["metadatas"])))
        if set(by_id) != set(ids):
            raise ValueError("Page token IDs mismatch")
        vectors = []
        for i, key in enumerate(ids):
            vector, stored = by_id[key]
            if stored != {"page_id": page_id, "token_index": i, "image_sha256": meta["image_sha256"]}:
                raise ValueError("Page token metadata mismatch")
            vectors.append(vector)
        result = _matrix(vectors, self.contract["dimension"])
        if _digest(result) != meta["vectors_sha256"]:
            raise ValueError("Page token content mismatch")
        return result

    def rerank(self, query, candidates):
        if len(set(candidates)) != len(candidates):
            raise ValueError("Candidate page IDs must be unique")
        query = _matrix(query, self.contract["dimension"])
        scored = [(page, maxsim(query, self.get_page(page))) for page in candidates]
        return sorted(scored, key=lambda item: (-item[1], item[0]))
