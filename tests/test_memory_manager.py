"""Small targeted tests; scripted LLMs and protocol doubles are not live API evals."""
import json
from types import SimpleNamespace
import pytest
from paper_agent.memory_manager import MemoryManager
from paper_agent.long_term import MemoryStore
from paper_agent.memory_stores import RedisWorkingStore, ChromaEpisodes
from paper_agent.pipeline import PaperAgentState
from test_reliability import pipeline
from paper_agent.storage import digest


class Cache:
    def __init__(self): self.values = {}
    def get(self, key): return self.values.get(key)
    def set(self, key, value, ex=None): self.values[key] = value


class Episodes:
    min_similarity = .35
    def __init__(self): self.values = {}
    def write(self, episode): self.values[episode["episode_id"]] = episode
    def search(self, session_id, query, paper_ids=None, event_type=None, limit=6):
        return [{"episode_id":k, "relevance_score":.9, "score_kind":"cosine_similarity", "payload_hash":digest(e)}
                for k, e in self.values.items() if e["session_id"] == session_id][:limit]
    def fingerprints(self, session_id, ids):
        return {i:digest(self.values[i]) for i in ids if i in self.values and self.values[i]["session_id"] == session_id}


class Down:
    def __getattr__(self, key):
        def fail(*a, **kw): raise ConnectionError("secret connection string")
        return fail


def manager(tmp_path, **kwargs):
    return MemoryManager(tmp_path/"memory", kwargs.get("working", RedisWorkingStore("unused", client=Cache())),
                         kwargs.get("episodic", Episodes()), fallback=kwargs.get("fallback", True))


def test_fact_versions_conflicts_delete(tmp_path):
    store = MemoryStore(tmp_path/"facts.sqlite3")
    fact = dict(key="Model_Size", text="10 parameters", source_type="user_explicit", status="active")
    with store.transaction() as db:
        first = store.put(db, "s", "t1", fact)
        assert store.put(db, "s", "t2", {**fact, "key":"model size"})["action"] == "duplicate"
        conflict = store.put(db, "s", "t3", {**fact, "text":"20 parameters"})
        assert conflict["action"] == "conflict"
        changed = store.put(db, "s", "t4", {**fact, "text":"30 parameters", "replaces":first["memory_id"]})
        assert changed["action"] == "update"
    rows = store.memories("s")
    assert {r["status"] for r in rows} == {"active", "superseded", "conflict"}
    assert store.memories("other") == []
    store.delete("s", changed["memory_id"])
    assert not store.memories("s", "active")
    with pytest.raises(RuntimeError):
        with store.transaction() as db:
            store.put(db, "s", "t5", {**fact, "key":"other"})
            raise RuntimeError("rollback")
    assert len(store.memories("s")) == 3


def test_redis_isolated_and_expiry_recovery(tmp_path):
    m = manager(tmp_path)
    p = pipeline(tmp_path, mode="bm25", memory_manager=m)
    state = PaperAgentState("s")
    p.run("研究决策: alpha", state, "t1")
    assert m.working.load("s")["messages"] == state.messages
    assert m.working.load("other") is None
    m.working.client.values.clear()  # simulate TTL expiry
    restored = PaperAgentState("s")
    result = p.run("研究决策: alpha", restored, "t1")
    assert result["replayed_commit"] and len(restored.messages) == 2
    assert len(m.store.episodes("s")) == 1
    assert len(m.store.memories("s")) == 0
    assert m.working.load("s") is not None


def test_current_evidence_required_and_paraphrase_not_authoritative(tmp_path):
    m = manager(tmp_path)
    evidence = [{"chunk_id":"c", "paper_id":"p", "text":"Accuracy is 90%."}]
    claim = {"text":"Accuracy is 90%.", "citations":[{"paper_id":"p", "chunk_id":"c"}]}
    answer = {"decision":"answer", "claims":[claim]}
    facts, _ = m.extract_memories("accuracy", answer, evidence, "t")
    assert facts == []
    full = "Accuracy is 90%. Evaluation used 100 examples."
    facts, _ = m.extract_memories("q", {**answer, "claims":[{**claim, "text":full}]},
                                  [{**evidence[0], "text":full}], "t")
    assert facts == []
    for field, value in (("chunk_id", "fake"), ("paper_id", "wrong")):
        bad = {**claim, "citations":[{**claim["citations"][0], field:value}]}
        with pytest.raises(ValueError): m.extract_memories("q", {**answer, "claims":[bad]}, evidence, "t")
    for text in ("Accuracy is 99%.", "The system performs well."):
        facts, _ = m.extract_memories("q", {**answer, "claims":[{**claim, "text":text}]}, evidence, "t")
        assert facts == []
    with pytest.raises(ValueError): m.extract_memories("q", answer, [], "t")


def test_service_failure_outbox_recovery_and_trace(tmp_path):
    m = manager(tmp_path, working=Down(), episodic=Down())
    p = pipeline(tmp_path, mode="bm25", memory_manager=m)
    p.run("研究决策: alpha", PaperAgentState("s"), "t1")
    assert len(m.store.episodes("s", pending=True)) == 1
    trace = json.loads((tmp_path/"traces/t1.json").read_text())
    assert trace["degraded"] and trace["commit_status"] == "committed"
    assert "secret connection" not in json.dumps(trace)
    assert m.retrieve_memories(PaperAgentState("s"), "alpha")
    m.working = RedisWorkingStore("unused", client=Cache())
    m.episodic = Episodes()
    assert p.run("研究决策: alpha", PaperAgentState("s"), "t1")["replayed_commit"]
    assert not m.store.episodes("s", pending=True)
    assert len(m.episodic.values) == 1


def test_strict_failure_has_trace_and_no_commit(tmp_path):
    m = manager(tmp_path, working=Down(), fallback=False)
    p = pipeline(tmp_path, memory_manager=m)
    with pytest.raises(RuntimeError, match="redis_read"):
        p.run("alpha", PaperAgentState("s"), "t1")
    assert m.store.snapshot("s") is None
    assert json.loads((tmp_path/"traces/t1.json").read_text())["status"] == "error"


def test_lost_index_recalled_from_original_and_rebuilt(tmp_path):
    m = manager(tmp_path)
    p = pipeline(tmp_path, mode="bm25", memory_manager=m)
    state = PaperAgentState("s")
    p.run("研究决策: alpha", state, "t1")
    m.episodic.values.clear()
    assert m.retrieve_memories(state, "alpha")
    assert m.store.episodes("s", pending=True)
    m.load_working_memory(state)
    assert len(m.episodic.values) == 1 and not m.store.episodes("s", pending=True)


def test_long_conversation_recall_and_legacy_import(tmp_path):
    m = manager(tmp_path)
    state = PaperAgentState("s")
    state.memory.summary["legacy_context_unverified"] = True
    state.save(tmp_path/"memory")
    p = pipeline(tmp_path, mode="bm25", memory_manager=m)
    for i in range(6): p.run(f"研究决策: alpha discussion {i}", state, f"t{i}")
    assert len(state.messages) == 8
    assert not m.store.memories("s")
    assert m.retrieve_memories(state, "alpha")
    assert not m.retrieve_memories(PaperAgentState("other"), "alpha")
    assert len(m.store.episodes("s")) == 6
    assert list((tmp_path/"memory/archives").glob("*.json"))


def test_chroma_explicit_vectors_and_filter(tmp_path):
    # Real Chroma, deterministic tiny vectors: validates storage, not BGE quality.
    pytest.importorskip("chromadb")
    import numpy as np
    encoder = SimpleNamespace(model_id="test_vectors", max_length=10,
                              encode=lambda texts: np.array([[1.,0.,0.] for _ in texts]))
    store = ChromaEpisodes(tmp_path/"chroma", encoder)
    for sid, paper in (("s", "p"), ("s", "other"), ("other", "p")):
        store.write(dict(episode_id=sid+paper, session_id=sid, turn_id="t", paper_ids=[paper],
                         evidence_ids=["c"], event_type="decision", created_at=1., source_type="conversation", content="alpha"))
    assert [h["episode_id"] for h in store.search("s", "alpha", ["p"], "decision")] == ["sp"]
    assert store.search("s", "alpha", ["p"], "different") == []
    store.write(dict(episode_id="sp", session_id="s", turn_id="t", paper_ids=["p"], evidence_ids=["c"],
                     event_type="decision", created_at=1., source_type="conversation", content="alpha"))
    assert store.collection.count() == 3


def test_explicit_user_update_and_similar_key_conflict(tmp_path):
    m = manager(tmp_path)
    p = pipeline(tmp_path, mode="bm25", memory_manager=m)
    state = PaperAgentState("s")
    p.run("记住:输出语言=中文", state, "t1")
    p.run("记住:输出语言=英文", state, "t2")
    rows = [f for f in m.store.memories("s") if f["source_type"] == "user_explicit"]
    assert {f["status"] for f in rows} == {"active", "conflict"}
    p.run("更新记忆:输出语言=英文", state, "t3")
    active = [f for f in m.store.memories("s", "active") if f["source_type"] == "user_explicit"]
    assert len(active) == 1 and active[0]["text"] == "英文" and active[0]["replaces"]
    with m.store.transaction() as db:
        result = m.store.put(db, "s", "t4", dict(key="different", text="英文", source_type="user_explicit", status="active"))
        assert result["action"] == "active"  # Different attributes no longer conflict by text overlap.


def test_archive_failure_rolls_back_all_durable_memory(tmp_path, monkeypatch):
    from paper_agent.memory import ContextArchive
    m = manager(tmp_path)
    p = pipeline(tmp_path, mode="bm25", memory_manager=m)
    state = PaperAgentState("s", messages=[{"role":"user", "content":"old"}]*8)
    def fail(*a, **kw): raise OSError("archive offline")
    monkeypatch.setattr(ContextArchive, "store", fail)
    with pytest.raises(OSError): p.run("alpha", state, "t1")
    assert m.store.snapshot("s") is None
    assert not m.store.memories("s") and not m.store.episodes("s")
    assert len(state.messages) == 8 and not state.committed_turns


def test_concurrent_commit_rejected_and_summary_fallback(tmp_path):
    m = manager(tmp_path)
    p = pipeline(tmp_path, mode="bm25", memory_manager=m, memory_strategy="semantic")
    state = PaperAgentState("s", messages=[{"role":"user", "content":"old"}]*8)
    p.run("alpha", state, "t1")
    # Scripted reasoner has no summary schema: bounded compaction must preserve archives.
    assert len(state.messages) == 8
    assert any(e["operation"] == "semantic_compaction" for e in m.events if "operation" in e)
    before = m.store.snapshot("s")
    with pytest.raises(RuntimeError, match="Concurrent"):
        m.commit(state, "q", {"decision":"refuse", "claims":[]}, [], "stale", 0, True, "bounded", p.reasoner)
    assert m.store.snapshot("s") == before
