import json
from dataclasses import asdict
from types import SimpleNamespace
import pytest
from paper_agent.providers import APIConfig, CompatibleReasoner, ProviderTransportError
from paper_agent.pipeline import PaperPipeline, PaperAgentState, compact_state
from paper_agent.memory import ContextArchive


def response(value):
    return {"choices":[{"finish_reason":"stop", "message":{"content":json.dumps(value)}}]}


@pytest.mark.parametrize("failure", [ProviderTransportError(429), ProviderTransportError(503), ProviderTransportError(kind="TimeoutError")])
def test_transient_retry(failure):
    attempts = []
    def transport(*args):
        attempts.append(1)
        if len(attempts) < 3: raise failure
        return response({"decision":"refuse", "reason":"insufficient"})
    r = CompatibleReasoner(APIConfig("secret", "test", retry_base_seconds=0), "s", transport)
    assert r.structured("reflect", {})["decision"] == "refuse"
    assert len(r.calls) == 3
    assert all("elapsed_ms" in c for c in r.calls)
    assert "secret" not in json.dumps(r.calls)


@pytest.mark.parametrize("status", [400,401,403])
def test_nonretryable(status):
    def transport(*args): raise ProviderTransportError(status)
    r = CompatibleReasoner(APIConfig("secret", "test", retry_base_seconds=0), "s", transport)
    with pytest.raises(RuntimeError): r.structured("reflect", {})
    assert len(r.calls) == 1


@pytest.mark.parametrize("value", [{"decision":"invent"}, None])
def test_single_schema_repair(value):
    r = CompatibleReasoner(APIConfig("secret", "test"), "s", lambda *a: response(value))
    with pytest.raises(RuntimeError): r.structured("reflect", {})
    assert len(r.calls) == 2


def pipeline(tmp_path, failure=None, **options):
    chunk = {"paper_id":"p", "chunk_id":"c", "text":"alpha evidence"}
    class Reasoner:
        def structured(self, stage, payload):
            if stage == "nlu": return {"intent":"fact", "standalone_query":"alpha", "resolved_entities":{"it":"p"}}
            if stage == "reflect":
                if failure == "reflect": return {"decision":"invalid"}
                return {"decision":"answer"}
            return {"claims":[{"key":"alpha", "text":"alpha", "citations":[{"paper_id":"p", "chunk_id":"wrong" if failure == "citation" else "c"}]}]}
    def search(*a, **kw):
        if failure == "route" and kw["mode"] == "hybrid": raise RuntimeError("route offline")
        return {"hits":[chunk], "candidate_ids":["c"]}
    retriever = SimpleNamespace(metadata={}, by_id={"c":chunk}, search=search, reranker=None)
    return PaperPipeline(retriever, Reasoner(), tmp_path/"traces", tmp_path/"memory", **options)


def test_idempotent_commit_and_restart(tmp_path):
    p = pipeline(tmp_path, mode="bm25")
    state = PaperAgentState("s")
    first = p.run("alpha", state, "t1")
    assert p.run("alpha", state, "t1")["replayed_commit"]
    restored = PaperAgentState.load(tmp_path/"memory", "s")
    assert p.run("alpha", restored, "t1")["answer"] == first["answer"]
    assert len(restored.messages) == 2 and restored.turn_count == 1
    with pytest.raises(ValueError): p.run("different", restored, "t1")


@pytest.mark.parametrize("failure", ["route", "citation", None])
def test_degraded_trace(tmp_path, failure):
    p = pipeline(tmp_path, failure=failure)
    p.run("alpha", PaperAgentState("s"), "t1")
    trace = json.loads((tmp_path/"traces/t1.json").read_text())
    assert trace["degraded"] and trace["commit_status"] == "committed"


def test_invalid_reflect_rolls_back(tmp_path):
    p = pipeline(tmp_path, failure="reflect")
    state = PaperAgentState("s")
    with pytest.raises(ValueError): p.run("alpha", state, "t1")
    assert not state.messages and state.turn_count == 0 and not state.committed_turns
    assert json.loads((tmp_path/"traces/t1.json").read_text())["status"] == "error"


def test_archive_write_failure_rolls_back(tmp_path, monkeypatch):
    p = pipeline(tmp_path)
    state = PaperAgentState("s", messages=[{"role":"user", "content":"old"}]*8)
    def fail(*args): raise OSError("archive unavailable")
    monkeypatch.setattr(ContextArchive, "store", fail)
    with pytest.raises(OSError): p.run("alpha", state, "t1")
    assert len(state.messages) == 8 and not state.committed_turns


def test_archive_recall_corrections_and_isolation(tmp_path):
    state = PaperAgentState("s")
    state.memory.update({"it":"p"}, [{"key":"alpha", "text":"old alpha", "evidence_ids":["c"]}], [], ["c"])
    state.memory.facts["alpha"]["turn_id"] = "t1"
    state.messages = [{"role":"user", "content":"alpha", "turn_id":"t1"}]*10
    compact_state(state, tmp_path, max_facts=0)
    archive = ContextArchive(tmp_path)
    state.memory.update({}, [{"key":"alpha", "text":"new alpha", "evidence_ids":["c"]}], [], ["c"])
    rows = archive.search("s", "alpha", current_memory=state.memory)
    assert any(r["superseded"] and r["turn_id"] == "t1" for r in rows)
    assert archive.search("other", "alpha") == []
    assert archive.read(rows[0]["archive_id"])


def test_long_pipeline_recalls_archived_early_message(tmp_path):
    p = pipeline(tmp_path, mode="bm25")
    state = PaperAgentState("s")
    p.run("alpha early unique question", state, "early")
    for i in range(5): p.run("alpha follow up", state, f"t{i}")
    assert not any(m.get("turn_id") == "early" for m in state.messages)
    p.run("alpha early unique question", state, "recall")
    trace = json.loads((tmp_path/"traces/recall.json").read_text())
    assert any(r["turn_id"] == "early" for r in trace["archive_recall"])
    assert trace["answer"]["claims"][0]["citations"][0]["chunk_id"] == "c"


def test_connect_budget_does_not_limit_header_wait():
    import time
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from paper_agent.providers import post_json
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == '/trickle':
                try:
                    for byte in b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}':
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(.01)
                except OSError: pass
                return
            time.sleep(.08)
            self.send_response(200)
            self.end_headers()
            try: self.wfile.write(b'{"ok":true}')
            except OSError: pass
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        assert post_json(url, {}, {}, 1, connect_timeout=.03) == {"ok":True}
        with pytest.raises(ProviderTransportError): post_json(url, {}, {}, .03, connect_timeout=.01)
        started=time.monotonic()
        with pytest.raises(ProviderTransportError): post_json(url+'/trickle', {}, {}, .06, connect_timeout=.03)
        assert time.monotonic()-started < .3
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
