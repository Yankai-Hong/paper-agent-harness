"""Official model/index dependencies with explicit, reproducible ranking.

BGE: FlagOpen/FlagEmbedding; BM25: dorianbrown/rank_bm25.
RRF formula reference: castorini/pyserini/pyserini/fusion/_base.py.
"""
import json
import hashlib
import time
from pathlib import Path

import numpy as np

from .corpus import lexical_tokens
from .storage import digest, read_jsonl, write_json, write_jsonl


def rrf(rankings, k=60):
    if k < 0:
        raise ValueError("RRF k must be nonnegative")
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class BGEEncoder:
    def __init__(self, model="BAAI/bge-m3", device="cpu", batch_size=2, max_length=768):
        from FlagEmbedding import BGEM3FlagModel
        self.model_id, self.batch_size, self.max_length = model, batch_size, max_length
        self.model = BGEM3FlagModel(model, devices=device, use_fp16=device.startswith("cuda"))

    def encode(self, texts):
        values = self.model.encode(texts, batch_size=self.batch_size, max_length=self.max_length,
                                   return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"]
        values = np.asarray(values, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.all(np.isfinite(values)):
            raise ValueError("Invalid embedding")
        return np.ascontiguousarray(values / norms)


class BGEReranker:
    def __init__(self, model="BAAI/bge-reranker-v2-m3", device="cpu", batch_size=2, max_length=768):
        from FlagEmbedding import FlagReranker
        self.model_id, self.batch_size, self.max_length = model, batch_size, max_length
        self.model = FlagReranker(model, devices=device, use_fp16=device.startswith("cuda"))

    def score(self, query, texts):
        scores = self.model.compute_score([[query, text] for text in texts], batch_size=self.batch_size,
                                          max_length=self.max_length, normalize=False)
        return np.atleast_1d(scores).tolist()


def build_index(chunks_path, directory, encoder=None):
    chunks = read_jsonl(chunks_path)
    ids = [c["chunk_id"] for c in chunks]
    if not chunks or len(ids) != len(set(ids)):
        raise ValueError("Corpus must be nonempty with unique chunk IDs")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"schema": 1, "corpus_hash": digest(chunks), "chunks": len(chunks),
                "bm25_tokenizer": "lexical-v1", "chunker": chunks[0].get("chunker"),
                "dense": False, "encoder": None}
    if encoder is not None:
        import faiss
        values = encoder.encode([c["text"] for c in chunks])
        index = faiss.IndexFlatIP(values.shape[1])
        index.add(values)
        faiss.write_index(index, str(directory / "dense.faiss"))
        metadata.update(dense=True, dimension=values.shape[1], encoder=encoder.model_id,
                        max_length=encoder.max_length,
                        index_sha256=hashlib.sha256((directory/"dense.faiss").read_bytes()).hexdigest())
        provenance = Path(encoder.model_id)/"provenance.json"
        metadata["model_provenance"] = json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else None
    write_jsonl(directory / "chunks.jsonl", chunks)
    write_json(directory / "metadata.json", metadata)
    return metadata


class Retriever:
    def __init__(self, directory, encoder=None, reranker=None):
        from rank_bm25 import BM25Okapi
        self.directory = Path(directory)
        self.chunks = read_jsonl(self.directory / "chunks.jsonl")
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        if digest(self.chunks) != self.metadata["corpus_hash"]:
            raise ValueError("Index corpus hash mismatch; rebuild index")
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([lexical_tokens(c["text"]) or ["__empty__"] for c in self.chunks])
        self.encoder, self.reranker, self.index = encoder, reranker, None
        if encoder is not None:
            if not self.metadata["dense"] or encoder.model_id != self.metadata["encoder"]:
                raise ValueError("Query encoder must match dense index")
            if encoder.max_length != self.metadata["max_length"]:
                raise ValueError("Query max_length must match index configuration")
            provenance = Path(encoder.model_id)/"provenance.json"
            if self.metadata.get("model_provenance") and (
                not provenance.exists() or json.loads(provenance.read_text(encoding="utf-8")) != self.metadata["model_provenance"]
            ):
                raise ValueError("Query model revision differs from index provenance")
            import faiss
            if hashlib.sha256((self.directory/"dense.faiss").read_bytes()).hexdigest() != self.metadata.get("index_sha256"):
                raise ValueError("Dense index checksum mismatch")
            self.index = faiss.read_index(str(self.directory / "dense.faiss"))
            if self.index.ntotal != len(self.chunks):
                raise ValueError("Dense index length mismatch")

    def retrieve_chunk(self, chunk_id, neighbor_window=0):
        if not 0 <= neighbor_window <= 2:
            raise ValueError("neighbor_window must be 0..2")
        if chunk_id not in self.by_id:
            raise KeyError(f"Unknown chunk: {chunk_id}")
        chunk = self.by_id[chunk_id]
        same_paper = [c for c in self.chunks if c["paper_id"] == chunk["paper_id"]]
        position = next(i for i, c in enumerate(same_paper) if c["chunk_id"] == chunk_id)
        return same_paper[max(0, position-neighbor_window):position+neighbor_window+1]

    def search(self, query, mode="hybrid", top_k=5, candidate_k=30, paper_ids=None):
        if mode not in {"bm25", "dense", "hybrid", "rerank"}:
            raise ValueError("Unknown retrieval mode")
        if not query.strip() or not 1 <= top_k <= candidate_k:
            raise ValueError("Nonempty query and 1 <= top_k <= candidate_k required")
        if mode != "bm25" and self.index is None:
            raise ValueError("Dense mode requires a real encoder/index; no silent fallback")
        if mode == "rerank" and self.reranker is None:
            raise ValueError("Rerank mode requires a real cross-encoder")
        start = time.perf_counter()
        allowed = set(paper_ids) if paper_ids is not None else None
        positions = [i for i, c in enumerate(self.chunks) if allowed is None or c["paper_id"] in allowed]
        sparse, dense = [], []
        if mode != "dense":
            scores = self.bm25.get_scores(lexical_tokens(query))
            sparse = sorted(positions, key=lambda i: (-float(scores[i]), self.chunks[i]["chunk_id"]))[:candidate_k]
        if mode != "bm25" and positions:
            vector = self.encoder.encode([query])
            # Small corpus: exact search all then filter BEFORE top-k selection.
            scores, indices = self.index.search(vector, self.index.ntotal)
            allowed_positions = set(positions)
            # 过滤
            pairs = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if int(i) in allowed_positions]
            dense = [i for i, _ in sorted(pairs, key=lambda x: (-x[1], self.chunks[x[0]]["chunk_id"]))[:candidate_k]]
        sid = [self.chunks[i]["chunk_id"] for i in sparse]
        did = [self.chunks[i]["chunk_id"] for i in dense]
        # RRF 排名融合
        ranking = sid if mode == "bm25" else did if mode == "dense" else [i for i, _ in rrf([sid, did])]
        ranking = ranking[:candidate_k]
        retrieval_ms = (time.perf_counter()-start)*1000
        fusion_ranks = {cid: i+1 for i, cid in enumerate(ranking)}
        rerank_scores = {}
        rerank_start = time.perf_counter()

        # cross-encoder 重排
        if mode == "rerank" and ranking:
            values = self.reranker.score(query, [
                (self.by_id[i].get('caption') or '') if self.by_id[i].get('evidence_type') in {'figure','table'}
                else self.by_id[i]['text'] for i in ranking])
            if len(values) != len(ranking) or not np.all(np.isfinite(values)):
                raise ValueError("Invalid reranker output")
            rerank_scores = dict(zip(ranking, map(float, values)))
            ranking.sort(key=lambda i: (-rerank_scores[i], i))
        hits = [{**self.by_id[cid], "rank": rank+1, "bm25_rank": sid.index(cid)+1 if cid in sid else None,
                 "dense_rank": did.index(cid)+1 if cid in did else None,
                 "fusion_rank": fusion_ranks[cid], "rerank_score": rerank_scores.get(cid)}
                for rank, cid in enumerate(ranking[:top_k])]
        return {"query": query, "mode": mode, "hits": hits, "candidate_ids": list(fusion_ranks),
                "channel_refs": {"bm25":[{"paper_id":self.by_id[cid]['paper_id'], "chunk_id":cid} for cid in sid],
                                 "dense":[{"paper_id":self.by_id[cid]['paper_id'], "chunk_id":cid} for cid in did]},
                "timing_ms": {"retrieve": retrieval_ms, "rerank": (time.perf_counter()-rerank_start)*1000},
                "corpus_hash": self.metadata["corpus_hash"]}
