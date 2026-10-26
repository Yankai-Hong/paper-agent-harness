"""Targeted memory regressions. No external LLM calls."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from paper_agent.memory import ContextArchive
from paper_agent.memory_stores import ChromaEpisodes, paper_ids
from paper_agent.pipeline import PaperAgentState
from test_memory_manager import manager
from test_reliability import pipeline


def episode(identifier, text="alpha research decision"):
    return dict(episode_id=identifier, session_id="s", turn_id=identifier, paper_ids=["p"],
                evidence_ids=["c"], event_type="research_decision", created_at=1.,
                source_type="conversation", content=text)


def add_episode(m, e):
    with m.facts.transaction() as db:
        db.execute("INSERT INTO episodes VALUES (?,?,?,1)", (e["episode_id"], "s", json.dumps(e)))
    m.episodic.write(e)


def test_unified_ranking_budget_and_dedup(tmp_path, monkeypatch):
    m = manager(tmp_path)
    with m.facts.transaction() as db:
        for i in range(12):
            m.facts.put_fact(db, "s", f"f{i}", dict(key=f"alpha {i}", text=f"alpha record {i}",
                            source_type="user_explicit", status="active"))
    e = episode("decision")
    add_episode(m, e)
    monkeypatch.setattr(ContextArchive, "search", lambda *a, **k: [dict(kind="message", turn_id="decision",
                        archive_id="archive", content={"role":"user", "content":e["content"]}, evidence_ids=[])])
    for budget in (5000, 1800):
        records = m.retrieve_memories(PaperAgentState("s"), "alpha", limit=4, max_chars=budget)
        assert any(r["kind"] == "episode" for r in records)
        assert len({r["record_id"] for r in records}) == len(records)
        assert not any(r["retrieval_source"] == "archive" for r in records)
        assert sum(len(json.dumps(r, ensure_ascii=False)) for r in records) <= budget
    assert m.retrieve_memories(PaperAgentState("s"), "") == []


@pytest.mark.parametrize("stored", ["paper_a", '["paper_b", "paper_a"]', ["paper_a", "paper_b"]])
def test_multi_paper_and_legacy_scope(tmp_path, stored):
    m = manager(tmp_path)
    with m.facts.transaction() as db:
        m.facts.put_fact(db, "s", "t", dict(key="alpha", text="alpha finding", source_type="paper_evidence",
                        paper_id=stored, status="active", evidence_ids=["c"]))
    state = PaperAgentState("s")
    assert m.retrieve_memories(state, "alpha", ["paper_a"])
    assert bool(m.retrieve_memories(state, "alpha", ["paper_b"])) == (len(paper_ids(stored)) == 2)
    assert m.retrieve_memories(state, "alpha", ["paper_c"]) == []
    assert m.retrieve_memories(state, "alpha")
    with pytest.raises(ValueError): paper_ids('["paper_a", 12]')
    with pytest.raises(ValueError): paper_ids('[broken')


def test_conflict_history_and_generate_boundary(tmp_path):
    m = manager(tmp_path)
    with m.facts.transaction() as db:
        base = dict(key="alpha", text="alpha old", source_type="user_explicit", status="active")
        old = m.facts.put_fact(db, "s", "old", base)
        m.facts.put_fact(db, "s", "conflict", {**base, "text":"alpha disputed"})
        m.facts.put_fact(db, "s", "new", {**base, "text":"alpha current", "replaces":old["fact_id"]})
    state = PaperAgentState("s")
    records = m.retrieve_memories(state, "alpha")
    assert {r["status"] for r in records} == {"active", "conflict"}
    conflict = next(r for r in records if r["status"] == "conflict")
    assert conflict["kind"] == "conflict" and conflict["requires_resolution"] and not conflict["is_current_fact"]
    assert conflict["historical_context_not_evidence"] and not conflict["is_authoritative"]
    assert any(r["status"] == "superseded" for r in m.retrieve_memories(state, "alpha history"))
    p = pipeline(tmp_path, mode="bm25", memory_manager=m)
    original, payloads = p.reasoner.structured, {}
    def recording(stage, payload):
        payloads[stage] = payload
        return original(stage, payload)
    p.reasoner.structured = recording
    p.run("alpha", state, "target")
    for stage in ("nlu", "reflect", "generate"):
        assert any(r["status"] == "conflict" and not r["is_authoritative"]
                   for r in payloads[stage]["historical_context_not_evidence"])
    assert payloads["generate"]["evidence"] == [{"paper_id":"p", "chunk_id":"c", "text":"alpha evidence"}]
    assert any(f["text"] == "alpha disputed" for f in m.facts.facts("s", "conflict"))


def test_retention_and_multi_paper_extraction(tmp_path):
    m = manager(tmp_path)
    evidence = [dict(chunk_id=p, paper_id=p, text="Exact finding.") for p in ("a", "b")]
    answer = dict(decision="answer", claims=[dict(text="Exact finding.", citations=[
        dict(chunk_id=p, paper_id=p) for p in ("a", "b")])])
    facts, ordinary = m.extract_memories("What is the finding?", answer, evidence, "t")
    assert ordinary is None and facts[0]["paper_ids"] == ["a", "b"] and facts[0]["status"] == "active"
    for query in ("研究决策:采用此方法", "实验结果:下降 10%", "后续待办:复现实验", "偏好:语言=中文"):
        _, retained = m.extract_memories(query, answer, evidence, "t")
        assert retained and retained["retention_reason"] == "explicit_user_marker"
    p = pipeline(tmp_path, mode="bm25", memory_manager=m)
    state = PaperAgentState("s")
    p.run("alpha", state, "ordinary")
    assert not m.facts.episodes("s") and len(state.messages) == 2
    p.run("研究决策: alpha", state, "decision")
    p.run("研究决策: alpha", state, "decision")
    assert len(m.facts.episodes("s")) == 1 and len(state.messages) == 4


def test_batched_partial_index_and_changed_payload(tmp_path):
    m = manager(tmp_path)
    for i in range(40): add_episode(m, episode(f"e{i:02}"))
    del m.episodic.values["e35"]
    m.episodic.values["e02"] = {**m.episodic.values["e02"], "content":"stale text"}
    batches, original = [], m.episodic.fingerprints
    def checking(sid, ids):
        batches.append(ids)
        return original(sid, ids)
    m.episodic.fingerprints = checking
    state = PaperAgentState("s")
    m.repair_projections(state)
    assert "e35" not in m.episodic.values
    m.repair_projections(state)
    assert len(m.episodic.values) == 40 and all(len(b) <= 32 for b in batches)
    assert m.episodic.values["e02"]["content"] == "alpha research decision"
    assert not m.facts.episodes("s", pending=True)


def test_real_chroma_distance_and_partial_recovery(tmp_path):
    pytest.importorskip("chromadb")
    encoder = SimpleNamespace(model_id="test_unit_vectors", max_length=20,
        encode=lambda texts: np.array([[-1.,0.] if t == "no match" else [1.,0.] if "alpha" in t else [0.,1.] for t in texts]))
    store = ChromaEpisodes(tmp_path/"chroma", encoder, min_similarity=.5)
    m = manager(tmp_path, episodic=store)
    add_episode(m, episode("near"))
    add_episode(m, episode("far", text="unrelated biology"))
    hits = store.search("s", "alpha")
    assert [h["episode_id"] for h in hits] == ["near"]
    assert hits[0]["metric"] == "l2" and hits[0]["distance"] == pytest.approx(0)
    assert hits[0]["relevance_score"] == pytest.approx(1)
    assert m.retrieve_memories(PaperAgentState("s"), "no match") == []
    assert not m.facts.episodes("s", pending=True)
    store.client.create_collection("cosine_test", embedding_function=None,
        metadata={"encoder_signature":store.collection.metadata["encoder_signature"], "hnsw:space":"cosine"})
    cosine = ChromaEpisodes(tmp_path/"chroma", encoder, collection="cosine_test", client=store.client, min_similarity=.5)
    cosine.write(episode("near"))
    cosine.write(episode("far", text="unrelated biology"))
    assert [h["episode_id"] for h in cosine.search("s", "alpha")] == ["near"]
    assert cosine.search("s", "alpha")[0]["metric"] == "cosine"
    assert m.retrieve_memories(PaperAgentState("other"), "alpha") == []
    store.collection.delete(ids=["near"])
    assert store.collection.count() == 1
    m.repair_projections(PaperAgentState("s"))
    assert store.collection.count() == 2 and store.search("s", "alpha")[0]["episode_id"] == "near"
    changed = {**next(e for e in m.facts.episodes("s") if e["episode_id"] == "near"), "paper_ids":["new"]}
    with m.facts.transaction() as db:
        db.execute("UPDATE episodes SET payload=? WHERE episode_id=?", (json.dumps(changed), "near"))
    m.repair_projections(PaperAgentState("s"))
    assert store.search("s", "alpha", ["p"]) == []
    assert store.search("s", "alpha", ["new"])


@pytest.mark.parametrize("distance", [None, float('nan'), float('inf'), -1, 6])
def test_invalid_distance_is_not_invented(tmp_path, distance):
    encoder = SimpleNamespace(model_id="test", max_length=2, encode=lambda t:np.array([[1.,0.]]))
    store = ChromaEpisodes(tmp_path, encoder)
    store._collection = SimpleNamespace(metadata={"hnsw:space":"l2"}, count=lambda:1,
        query=lambda **kw: {"ids":[["a"]], "distances":[[distance]], "metadatas":[[{}]]})
    with pytest.raises(ValueError): store.search("s", "q")
    assert store.search("s", "") == []


def test_missing_distance_and_invalid_vector(tmp_path):
    encoder = SimpleNamespace(model_id="test", max_length=2, encode=lambda t:np.array([[1.,0.]]))
    store = ChromaEpisodes(tmp_path, encoder)
    store._collection = SimpleNamespace(metadata={"hnsw:space":"l2"}, count=lambda:1,
        query=lambda **kw: {"ids":[["a"]], "metadatas":[[{}]]})
    with pytest.raises(ValueError, match="Missing"): store.search("s", "q")
    encoder.encode = lambda texts: np.array([[float('nan'),0.]])
    with pytest.raises(ValueError, match="embedding"): store.search("s", "q")
    encoder.encode = lambda texts: np.array([[0.,0.]])
    with pytest.raises(ValueError, match="norm"): store.search("s", "q")
