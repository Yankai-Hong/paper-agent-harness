import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from paper_agent.providers import APIConfig, CompatibleReasoner
from paper_agent.pipeline import PaperPipeline, PaperAgentState, compact_state
from paper_agent.memory import ContextArchive
from paper_agent.corpus import Chunker
from paper_agent.retrieval import build_index, Retriever
from paper_agent.storage import write_jsonl


@pytest.fixture
def server():
    records = []
    answers = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            records.append((self.path, dict(self.headers), body))
            status, response = answers.pop(0)
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        def log_message(self, *args):
            pass
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    yield f'http://127.0.0.1:{httpd.server_port}/v1', records, answers
    httpd.shutdown()
    httpd.server_close()
    worker.join()


def response(value, finish='stop'):
    return {"model": "fixture-only", "choices": [{"finish_reason": finish,
            "message": {"content": json.dumps(value)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 60}}


def test_http_contract_full_pipeline_and_session_resume(server, tmp_path):
    url, requests, answers = server
    chunk = Chunker().split({"paper_id": "p", "title": "Example",
        "paragraphs": [{"paragraph_id": "a", "section": "Methods", "text": "retrieval uses sparse matching"}]})[0]
    write_jsonl(tmp_path/'chunks.jsonl', [chunk])
    build_index(tmp_path/'chunks.jsonl', tmp_path/'index')
    nlu = {"intent": "fact", "standalone_query": "retrieval", "needs_clarification": False,
           "paper_ids": ["p"], "resolved_entities": {"it": "p"}}
    claims = {"claims": [{"key": "method", "text": "uses sparse matching",
        "citations": [{"paper_id": "p", "chunk_id": chunk['chunk_id']}]}]}
    answers.extend([(200, response(v)) for v in [nlu, {"decision": "answer", "reason": "fixture"}, claims]])
    reasoner = CompatibleReasoner(APIConfig('test-secret', 'fixture-only', url), 'conversation-1')
    pipeline = PaperPipeline(Retriever(tmp_path/'index'), reasoner, tmp_path/'traces', tmp_path/'memory', 'bm25')
    result = pipeline.run('What retrieval?', PaperAgentState('conversation-1'))
    assert result['answer']['decision'] == 'answer'
    assert len(requests) == 3
    assert requests[0][0] == '/v1/chat/completions'
    assert requests[0][1]['X-Opencode-Session'] == 'conversation-1'
    assert requests[0][2]['response_format'] == {'type': 'json_object'}
    assert reasoner.calls[0]['usage']['input_tokens'] == 40
    assert sum(reasoner.calls[0]['usage'].values()) == 120
    restored = PaperAgentState.load(tmp_path/'memory', 'conversation-1')
    assert restored.turn_count == 1 and restored.memory.entities == {'it': 'p'}
    assert len(restored.messages) == 2
    trace_text = next((tmp_path/'traces').glob('*.json')).read_text(encoding='utf-8')
    assert 'test-secret' not in trace_text
    assert len(json.loads(trace_text)['api_calls']) == 3


@pytest.mark.parametrize('failure', ['truncated', 'schema', 'http', 'empty'])
def test_api_errors_fail_closed_without_retry(server, failure):
    url, requests, answers = server
    value = response({'decision': 'answer', 'reason': 'ok'})
    if failure == 'truncated':
        value['choices'][0]['finish_reason'] = 'length'
    elif failure == 'schema':
        value = response({'decision': 'invent'})
    elif failure == 'empty':
        value['choices'][0]['message']['content'] = ''
    answers.append((401 if failure == 'http' else 200, value))
    reasoner = CompatibleReasoner(APIConfig('secret', 'fixture', url, json_mode=False, repair_retries=0), 's')
    with pytest.raises(RuntimeError):
        reasoner.structured('reflect', {})
    assert len(requests) == 1 and reasoner.calls[0]['status'] == 'error'
    assert 'response_format' not in requests[0][2]


def test_limits_stop_before_network_and_key_not_in_repr():
    config = APIConfig('SECRET', 'fixture', max_calls=1)
    assert 'SECRET' not in repr(config)
    reasoner = CompatibleReasoner(config, 's', lambda *args: response({'decision': 'refuse', 'reason': 'no evidence'}))
    reasoner.structured('reflect', {})
    with pytest.raises(RuntimeError, match='API_MAX_CALLS'):
        reasoner.structured('reflect', {})
    with pytest.raises(ValueError):
        APIConfig('SECRET', 'fixture', 'http://example.com')


def test_explicit_thinking_setting_is_sent(server):
    url, requests, answers = server
    answers.append((200, response({'decision': 'refuse', 'reason': 'fixture'})))
    reasoner = CompatibleReasoner(APIConfig('test', 'fixture', url, thinking='disabled'), 's')
    reasoner.structured('reflect', {})
    assert requests[0][2]['thinking'] == {'type': 'disabled'}
    with pytest.raises(ValueError):
        APIConfig('test', 'fixture', thinking='invalid')


def test_working_memory_compaction_archives_evicted_facts(tmp_path):
    state = PaperAgentState('s')
    for i in range(45):
        state.memory.update({'it': str(i)}, [])
    state.messages = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': str(i)} for i in range(12)]
    result = compact_state(state, tmp_path)
    assert len(state.messages) == 8
    archived = ContextArchive(tmp_path).read(result['archive_id'])
    assert 'facts' not in archived['memory'] and len(archived['messages']) == 12
    assert state.memory.entities['it'] == '44'
