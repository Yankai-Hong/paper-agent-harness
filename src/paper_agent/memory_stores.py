"""Durable memory and replaceable projections. No implicit embedding downloads."""
from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import time
import unicodedata
import re
import math
import threading

from .storage import digest, safe_id


def canonical_key(value):
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = re.sub(r"[\W_]+", " ", value).strip()
    if not value:
        raise ValueError("Empty canonical fact key")
    return value


def paper_ids(value):
    """Read both legacy scalar/JSON-array scope and new list scope, fail on corruption."""
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value) if value.lstrip().startswith("[") else [value]
    if not isinstance(value, list) or any(not isinstance(p, str) or not p.strip() for p in value):
        raise ValueError("Invalid memory paper scope")
    return sorted(set(value))


class RedisWorkingStore:
    def __init__(self, url, namespace="paper-agent:working:v1", ttl=86400, client=None):
        if ttl < 0:
            raise ValueError("TTL must be nonnegative")
        self.namespace, self.ttl = namespace, ttl
        if client is None:
            import redis
            from redis.backoff import NoBackoff
            from redis.retry import Retry
            client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2,
                                          socket_connect_timeout=2, retry=Retry(NoBackoff(), 0))
        self.client = client

    def key(self, session_id):
        return f"{self.namespace}:{safe_id(session_id)}"

    def load(self, session_id):
        raw = self.client.get(self.key(session_id))
        value = json.loads(raw) if raw else None
        if value and value.get("session_id") != session_id:
            raise ValueError("Working memory session mismatch")
        return value

    def save(self, snapshot):
        # Durable SQLite commit always precedes this expiring projection.
        self.client.set(self.key(snapshot["session_id"]), json.dumps(snapshot, ensure_ascii=False),
                        ex=self.ttl or None)


class ChromaEpisodes:
    # Chroma 0.6 registers clients before migrations finish. Serialize lazy
    # initialization across instances in this process, not normal queries.
    _initialization_lock = threading.RLock()
    def __init__(self, path, encoder, collection="paper_agent_episodes_v1", client=None, min_similarity=.35):
        if encoder is None:
            raise ValueError("Episodic memory requires the existing BGE encoder")
        self.encoder, self.path, self.name, self.client = encoder, Path(path).resolve().as_posix(), collection, client
        self._collection = None
        if not math.isfinite(min_similarity) or not 0 <= min_similarity <= 1:
            raise ValueError("Memory minimum similarity must be in [0,1]")
        self.min_similarity = min_similarity

    @property
    def collection(self):
        with self._initialization_lock:
            return self._initialize_collection()

    def _initialize_collection(self):
        if self._collection is None:
            if self.client is None:
                import chromadb
                from chromadb.config import Settings
                Path(self.path).mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(path=self.path, settings=Settings(anonymized_telemetry=False))
            provenance = Path(self.encoder.model_id)/"provenance.json"
            signature = digest([self.encoder.model_id, self.encoder.max_length,
                                provenance.read_text(encoding="utf-8") if provenance.exists() else None])
            self._collection = self.client.get_or_create_collection(
                self.name, embedding_function=None, metadata={"encoder_signature": signature, "hnsw:space": "l2"})
            if self._collection.metadata.get("encoder_signature") != signature:
                self._collection = None
                raise ValueError("Episodic encoder differs; rebuild a new collection from SQLite originals")
        return self._collection

    def vectors(self, texts):
        import numpy as np
        values = np.asarray(self.encoder.encode(texts), dtype=np.float32)
        if values.ndim != 2 or len(values) != len(texts) or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError("Invalid episodic embedding")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Invalid episodic embedding norm")
        return (values / norms).tolist()

    def fingerprints(self, session_id, ids):
        if not ids:
            return {}
        rows = self.collection.get(ids=ids, where={"session_id": session_id}, include=["metadatas"])
        return {i: (m or {}).get("payload_hash") for i, m in zip(rows["ids"], rows["metadatas"])}

    def write(self, episode):
        metadata = {k: episode[k] for k in ("session_id", "turn_id", "event_type", "created_at", "source_type")}
        metadata.update(paper_ids=json.dumps(episode["paper_ids"]), evidence_ids=json.dumps(episode["evidence_ids"]))
        metadata["payload_hash"] = digest(episode)
        for paper in episode["paper_ids"]:
            metadata["paper_" + digest(paper)] = True
        vectors = self.vectors([episode["content"]])
        if self.collection.get(ids=[episode["episode_id"]], include=[])["ids"]:
            # Chroma merges metadata on upsert; remove stale paper flags on version change.
            self.collection.delete(ids=[episode["episode_id"]])
        self.collection.upsert(ids=[episode["episode_id"]], documents=[episode["content"]],
                               embeddings=vectors, metadatas=[metadata])

    def search(self, session_id, query, paper_ids=None, event_type=None, limit=6):
        if not query.strip() or limit <= 0:
            return []
        filters = [{"session_id": session_id}]
        if event_type:
            filters.append({"event_type": event_type})
        if paper_ids:
            papers = [{"paper_" + digest(p): True} for p in paper_ids]
            filters.append(papers[0] if len(papers) == 1 else {"$or": papers})
        where = filters[0] if len(filters) == 1 else {"$and": filters}
        count = self.collection.count()
        if not count:
            return []
        metric = self.collection.metadata.get("hnsw:space", "l2")
        if metric not in {"l2", "cosine"}:
            raise ValueError("Unsupported episodic distance metric")
        result = self.collection.query(query_embeddings=self.vectors([query]),
                                       where=where, n_results=min(limit, count), include=["distances", "metadatas"])
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0] or []
        metadata = (result.get("metadatas") or [[]])[0] or []
        if len(ids) != len(distances) or len(ids) != len(metadata):
            raise ValueError("Missing episodic distance or provenance")
        candidates = []
        for identifier, distance, meta in zip(ids, distances, metadata):
            if not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < -1e-5:
                raise ValueError("Invalid episodic distance")
            # Chroma l2 is SQUARED Euclidean distance. Both vectors are unit normalized.
            similarity = 1 - distance / 2 if metric == "l2" else 1 - distance
            if similarity < -1.00001 or similarity > 1.00001:
                raise ValueError("Distance inconsistent with normalized embeddings")
            similarity = max(-1., min(1., similarity))
            if similarity >= self.min_similarity:
                candidates.append({"episode_id": identifier, "distance": distance, "metric": metric,
                                   "relevance_score": similarity, "score_kind": "cosine_similarity",
                                   "payload_hash": (meta or {}).get("payload_hash")})
        return candidates
