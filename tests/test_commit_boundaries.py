"""Archive bytes can survive rollback; only durable authorization permits recall."""
from contextlib import contextmanager
from dataclasses import asdict
import pytest
from paper_agent.memory import ContextArchive
from paper_agent.pipeline import PaperAgentState
from test_memory_manager import manager
from test_reliability import pipeline


@pytest.mark.parametrize("backend", ["legacy", "three_layer"])
@pytest.mark.parametrize("strategy", ["bounded", "semantic"])
def test_archive_written_commit_failed_restart_and_retry(tmp_path, monkeypatch, backend, strategy):
    m = manager(tmp_path) if backend == "three_layer" else None
    p = pipeline(tmp_path, mode="bm25", memory_manager=m, memory_strategy=strategy)
    m = p.memory_manager
    original_structured = p.reasoner.structured
    p.reasoner.structured = lambda stage, payload: {"overview":"alpha"} if stage == "memory_summary" else original_structured(stage, payload)
    state = PaperAgentState("s")
    for i in range(4):
        p.run("alpha", state, f"t{i}")
    with monkeypatch.context() as patch:
        if m:
            transaction = m.store.transaction
            @contextmanager
            def fail_commit():
                with transaction() as db:
                    yield db
                    row = db.execute("SELECT version FROM sessions WHERE session_id='s'").fetchone()
                    if row and row[0] == 5:
                        raise RuntimeError("injected after archive before commit")
            patch.setattr(m.store, "transaction", fail_commit)
        else:
            patch.setattr(PaperAgentState, "save", lambda *a: (_ for _ in ()).throw(RuntimeError("commit failed")))
        with pytest.raises(RuntimeError):
            p.run("orphansecret alpha", state, "retry")
    archive = ContextArchive(tmp_path/"memory/archives")
    assert list(archive.root.glob("*.json"))  # Original bytes preserved, not cleaned up.
    restored = PaperAgentState.load(tmp_path/"memory", "s")
    restarted = manager(tmp_path) if backend == 'three_layer' else __import__('paper_agent.memory_manager',fromlist=['MemoryManager']).MemoryManager.local(tmp_path/'memory')
    durable = restarted.store.snapshot("s") if m else asdict(restored)
    assert not archive.search("s", "orphansecret", committed_state=durable)
    if restarted:
        assert not restarted.retrieve_memories(restored, "orphansecret")
    p.memory_manager = restarted
    p.run("committedsecret alpha", restored, "retry")
    durable = restarted.store.snapshot("s") if m else asdict(PaperAgentState.load(tmp_path/"memory", "s"))
    assert archive.search("s", "committedsecret", committed_state=durable)
    assert not archive.search("s", "orphansecret", committed_state=durable)
    assert p.run("committedsecret alpha", restored, "retry")["replayed_commit"]
    assert restored.turn_count == 5


def test_old_archive_requires_matching_committed_content(tmp_path):
    state = PaperAgentState("s", turn_count=1)
    answer = {"decision":"refuse", "claims":[]}
    state.committed_turns["old"] = {"question":"oldhistory", "result":{"answer":answer}}
    original = {"memory":asdict(state.memory), "turn_count":1,
                "messages":[{"role":"user", "content":"oldhistory", "turn_id":"old"}]}
    archive = ContextArchive(tmp_path/"archives")
    archive.store(original)
    archive.store({**original, "messages":[{"role":"user", "content":"orphansecret", "turn_id":"old"}]})
    state.save(tmp_path)
    assert archive.search("s", "oldhistory")
    assert not archive.search("s", "orphansecret")


@pytest.mark.parametrize("backend", ["legacy", "three_layer"])
def test_abrupt_process_exit_after_archive(tmp_path, backend):
    import subprocess
    import sys
    m = manager(tmp_path) if backend == "three_layer" else None
    p = pipeline(tmp_path, mode="bm25", memory_manager=m)
    m = p.memory_manager
    state = PaperAgentState("s")
    for i in range(4): p.run("alpha", state, f"t{i}")
    code = '''
import os, sys
from pathlib import Path
sys.path.insert(0, 'tests')
from test_reliability import pipeline
from test_memory_manager import manager
from paper_agent.memory import ContextArchive
from paper_agent.pipeline import PaperAgentState
root = Path(sys.argv[1])
m = manager(root) if sys.argv[2] == 'three_layer' else None
p = pipeline(root, mode='bm25', memory_manager=m)
store = ContextArchive.store
def crash(self, value):
    store(self, value)
    os._exit(79)
ContextArchive.store = crash
p.run('crashsecret alpha', PaperAgentState.load(root/'memory', 's'), 'crash')
'''
    result = subprocess.run([sys.executable,'-c',code,str(tmp_path),backend],timeout=30)
    assert result.returncode == 79
    assert list((tmp_path/'memory/archives').glob('*.json'))
    durable = manager(tmp_path).store.snapshot('s') if m else asdict(PaperAgentState.load(tmp_path/'memory','s'))
    assert durable['turn_count'] == 4
    assert not ContextArchive(tmp_path/'memory/archives').search('s','crashsecret',committed_state=durable)


def test_paper_scope_attribute_versions_and_conflict_recall(tmp_path):
    m = manager(tmp_path)
    declaration = dict(key='accuracy',text='Use strict criteria',source_type='user_explicit',status='active')
    with m.store.transaction() as db:
        first=m.store.put(db,'s','t1',declaration)
        assert m.store.put(db,'s','t2',{**declaration,'text':'Use broad criteria'})['action']=='conflict'
        assert m.store.put(db,'s','t3',{**declaration,'text':'Use broad criteria','replaces':first['memory_id']})['action']=='update'
        with pytest.raises(ValueError): m.store.put(db,'s','t4',{**declaration,'source_type':'paper_evidence'})
    recalled=m.retrieve_memories(PaperAgentState('s'),'accuracy',['p1'],limit=20)
    assert any(r['status']=='conflict' and not r['is_authoritative'] for r in recalled)


def test_chroma_first_access_concurrent_instances(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from types import SimpleNamespace
    from paper_agent.memory_stores import ChromaEpisodes
    pytest.importorskip('chromadb')
    barrier = Barrier(3)
    stores = [ChromaEpisodes(tmp_path/'new-chroma',SimpleNamespace(model_id='test',max_length=1)) for _ in range(3)]
    def read(i):
        barrier.wait(timeout=10)
        return stores[i].search('s'+str(i),'alpha')
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(read,range(3))) == [[],[],[]]
