"""Real Redis/SQLite/Chroma/BGE/FAISS smoke test; scripted reasoner, no LLM API."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import uuid

from paper_agent.memory_manager import MemoryManager
from paper_agent.memory_stores import RedisWorkingStore, ChromaEpisodes
from paper_agent.pipeline import PaperAgentState, PaperPipeline
from paper_agent.retrieval import BGEEncoder, Retriever
from paper_agent.storage import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--index", default="indexes/bge")
    parser.add_argument("--encoder", default="models/bge-m3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    result = {"status":"running", "llm":"scripted; no API calls", "checks":[]}
    started = time.monotonic()
    cache = None
    sid = "smoke_" + uuid.uuid4().hex
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        cache = RedisWorkingStore(args.redis_url, namespace="paper-agent:test:"+sid, ttl=1)
        assert cache.client.ping()
        encoder = BGEEncoder(args.encoder, args.device)
        retriever = Retriever(args.index, encoder)
        episodes = ChromaEpisodes(root/"chroma", encoder)
        manager = MemoryManager(root/"memory", cache, episodes, fallback=False)

        class Reasoner:
            def structured(self, stage, payload):
                if stage == "nlu":
                    return {"intent":"fact", "standalone_query":"attention mechanism", "resolved_entities":{}}
                if stage == "reflect":
                    return {"decision":"answer"}
                hit = payload["evidence"][0]
                return {"claims":[{"key":"verbatim excerpt", "text":hit["text"],
                                   "citations":[{"paper_id":hit["paper_id"], "chunk_id":hit["chunk_id"]}]}]}

        pipeline = PaperPipeline(retriever, Reasoner(), root/"traces", root/"memory",
                                 mode="hybrid", memory_manager=manager)
        state = PaperAgentState(sid)
        for i in range(6):
            pipeline.run(f"研究决策: attention mechanism discussion {i}", state, f"turn{i}")
        assert len(state.messages) == 8 and state.turn_count == 6
        assert len(manager.facts.facts(sid, "active")) == 1
        assert len(manager.facts.episodes(sid)) == 6 and episodes.collection.count() == 6
        assert not manager.facts.episodes(sid, pending=True)
        result["checks"].append("six pipeline turns: real FAISS/BGE/Redis/SQLite/Chroma, compacted history")
        assert episodes.search(sid, "attention mechanism")
        paper = next(iter(manager.facts.episodes(sid)))["paper_ids"][0]
        assert episodes.search(sid, "attention", [paper], "research_decision")
        assert episodes.search(sid, "attention", event_type="user_decision") == []
        assert episodes.search("other", "attention mechanism") == []
        assert manager.retrieve_memories(state, "attention mechanism")
        result["checks"].append("semantic recall and session/paper/event filters")
        time.sleep(1.2)
        assert cache.load(sid) is None
        restored = PaperAgentState(sid)
        assert pipeline.run("研究决策: attention mechanism discussion 5", restored, "turn5")["replayed_commit"]
        assert restored.turn_count == 6 and episodes.collection.count() == 6
        assert cache.load(sid) is not None
        result["checks"].append("real Redis TTL expiry, durable restore, idempotent replay")
        traces = [json.loads(p.read_text(encoding="utf-8")) for p in (root/"traces").glob("*.json")]
        assert len(traces) == 6 and not any(t["degraded"] for t in traces)
        result.update(status="passed", turns=6, active_facts=1, episodes=6, corpus_hash=retriever.metadata["corpus_hash"],
                      versions={p:importlib.metadata.version(p) for p in ("redis", "chromadb", "FlagEmbedding", "faiss-cpu")})
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(root/"result.json", result)
        if cache:
            try:
                cache.client.delete(cache.key(sid))
            except Exception:
                pass


if __name__ == "__main__":
    main()
