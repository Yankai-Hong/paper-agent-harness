"""Readback checks use synthetic data and a fake API; never contact a server."""
import importlib.util
import json
import time
from pathlib import Path

import pytest
from paper_agent.observability import metadata, score_id, trace_mapping, trace_summary, stage_summary


spec = importlib.util.spec_from_file_location('langfuse_readback', Path('scripts/check_langfuse_readback.py'))
readback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readback)


def fixture(run):
    trace = {'session_id':'s', 'turn_id':'t', 'case_id':'c', 'run_id':run,
             'status':'completed', 'commit_status':'committed',
             'stages':[{'stage':'Retrieve','invocation_id':'i','attempt':1,'status':'ok','elapsed_ms':2.,
                        'output':{'candidate_ids':['synthetic-evidence']}},
                       {'stage':'Memory','invocation_id':'m','status':'ok','elapsed_ms':1.}],
             'api_calls':[{'invocation_id':'i','request_id':'q','status':'ok','elapsed_ms':1.,
                           'model_requested':'fixture-model','usage_complete':True,
                           'usage':{'input_tokens':2,'output_tokens':3}}],
             'scores':[{'name':'execution_success','source':'rule','value':1}]}
    trace['observability_mapping'] = trace_mapping(trace)
    tid = trace_mapping(trace)['langfuse_trace_id']
    root = {'id':'root','trace_id':tid,'name':'Turn','parent_observation_id':None,
            'metadata':trace_summary(trace)}
    ret = {'id':'ret','trace_id':tid,'name':'Retrieve','parent_observation_id':'root',
           'level':'DEFAULT','metadata':{**metadata(trace['stages'][0]),**stage_summary(trace['stages'][0])}}
    mem = {'id':'mem','trace_id':tid,'name':'Memory','parent_observation_id':'root',
           'level':'DEFAULT','metadata':metadata(trace['stages'][1])}
    gen = {'id':'gen','trace_id':tid,'name':'LLM request','parent_observation_id':'ret',
           'provided_model_name':'fixture-model','level':'DEFAULT','metadata':metadata(trace['api_calls'][0]),
           'usage_details':{'input':2,'output':3}}
    score = {'id':score_id(tid,'rule','execution_success'),'name':'execution_success','value':1,
             'metadata':{'source':'rule'},'subject':{'kind':'trace','id':tid}}
    return trace, [root, ret, mem, gen], [score]


def test_correct_snapshot_checks_tree_metadata_usage_and_rule():
    trace, obs, scores = fixture('a')
    assert all(v == 'passed' for v in readback.verify_trace(trace, obs, scores).values())


def test_missing_and_wrong_usage_never_pass():
    trace, obs, scores = fixture('a')
    obs[-1]['usage_details'] = None
    assert readback.verify_trace(trace, obs, scores)['usage'] == 'unknown'
    obs[-1]['usage_details'] = {'input':2,'output':8}
    assert readback.verify_trace(trace, obs, scores)['usage'] == 'failed'
    trace['api_calls'][0]['usage_complete'] = False
    assert readback.verify_trace(trace, obs, scores)['usage'] == 'unknown'


def test_bad_parent_and_score_assignment_fail():
    trace, obs, scores = fixture('a')
    obs[-1]['parent_observation_id'] = 'wrong'
    scores[0]['subject']['id'] = 'wrong'
    result = readback.verify_trace(trace, obs, scores)
    assert result['tree'] == result['rule_score'] == 'failed'


def test_unresolved_root_parent_unknown_and_cycle_fails():
    trace, obs, scores = fixture('a')
    obs[0]['parent_observation_id'] = 'external-unresolved-parent'
    assert readback.verify_trace(trace, obs, scores)['tree'] == 'unknown'
    obs[0]['parent_observation_id'] = obs[1]['id']
    assert readback.verify_trace(trace, obs, scores)['tree'] == 'failed'


def test_model_field_fallback_checks_conflicting_values():
    trace, obs, scores = fixture('a')
    generation = obs[-1]
    generation['model'] = generation.pop('provided_model_name')
    assert readback.verify_trace(trace, obs, scores)['request'] == 'passed'
    generation['provided_model_name'] = 'different-model'
    assert readback.verify_trace(trace, obs, scores)['request'] == 'failed'
    generation.pop('provided_model_name')
    generation['metadata']['model_requested'] = 'different-model'
    assert readback.verify_trace(trace, obs, scores)['request'] == 'failed'


def test_missing_response_and_score_are_unknown():
    trace, obs, _ = fixture('a')
    assert readback.verify_trace(trace, [], [])['tree'] == 'unknown'
    assert readback.verify_trace(trace, obs, [])['rule_score'] == 'unknown'


def test_isolation_requires_actual_server_observations():
    first, second = fixture('a'), fixture('b')
    class API:
        class observations:
            @staticmethod
            def get_many(**kwargs): return {'data':[], 'meta':{}}
        class scores_v3:
            @staticmethod
            def get_many_v3(**kwargs): return {'data':[], 'meta':{}}
    result=readback.readback(API(), [first[0],second[0]], time.monotonic()+5)
    assert result['trace_isolation'] == 'unknown' and result['status'] == 'not_verified'
    class WrongAPI(API):
        class observations:
            @staticmethod
            def get_many(**kwargs): return {'data':[{'trace_id':'wrong'}], 'meta':{}}
    result=readback.readback(WrongAPI(), [first[0],second[0]], time.monotonic()+5)
    assert result['trace_isolation'] == 'failed' and result['status'] == 'not_verified'


def test_pagination_two_runs_and_non_sensitive_output():
    first = fixture('a'); second = fixture('b')
    data = {trace_mapping(x[0])['langfuse_trace_id']:x for x in (first,second)}
    calls = []
    class Endpoint:
        def __init__(self, which): self.which = which
        def get(self, **kwargs):
            calls.append(kwargs)
            rows = data[kwargs['trace_id']][self.which]
            if self.which == 1 and kwargs.get('cursor') is None:
                return {'data':rows[:2], 'meta':{'cursor':'next'}}
            return {'data':rows[2:] if self.which == 1 else rows, 'meta':{}}
    class API: pass
    api = API()
    api.observations = type('Observations', (), {'get_many':Endpoint(1).get})()
    api.scores_v3 = type('Scores', (), {'get_many_v3':Endpoint(2).get})()
    result = readback.readback(api, [first[0],second[0]], time.monotonic()+10)
    assert result['status'] == 'readback_checked'
    assert result['complete_end_to_end_acceptance'] == 'not_verified'
    assert result['replay'] == result['ui'] == 'not_checked'
    assert len(calls) == 6 and all(c['request_options']['max_retries'] == 0 for c in calls)
    assert 'synthetic-evidence' not in json.dumps(result)
    assert 'fixture-model' not in json.dumps(result)


def test_reused_run_and_cursor_cycle_rejected():
    trace, _, _ = fixture('a')
    assert readback.readback(object(), [trace,trace], time.monotonic()+5)['status'] == 'not_verified'
    with pytest.raises(ValueError):
        readback._pages(lambda **kwargs: {'data':[], 'meta':{'cursor':'same'}},
                        deadline=time.monotonic()+5, kwargs={})


def test_stuck_sdk_call_is_bounded():
    import threading
    trace_a, trace_b = fixture('a')[0], fixture('b')[0]
    release = threading.Event()
    class Observations:
        def get_many(self, **kwargs):
            release.wait(3)
            return {'data':[], 'meta':{}}
    api = type('API', (), {'observations':Observations()})()
    started = time.monotonic()
    result = readback.bounded_readback(api, [trace_a,trace_b], timeout=.02)
    release.set()
    assert result == {'status':'not_verified','reason':'readback_timeout'}
    assert time.monotonic()-started < .5


def test_missing_config_never_constructs_client(monkeypatch, tmp_path, capsys):
    a,b=fixture('a')[0],fixture('b')[0]
    for name in ('LANGFUSE_BASE_URL','LANGFUSE_PUBLIC_KEY','LANGFUSE_SECRET_KEY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    for path,trace in ((tmp_path/'a.json',a),(tmp_path/'b.json',b)):
        path.write_text(json.dumps(trace),encoding='utf-8')
    monkeypatch.setattr('sys.argv',['check','a.json','b.json'])
    assert readback.main() == 2
    assert json.loads(capsys.readouterr().out)['reason'] == 'missing_configuration'


def test_basic_auth_endpoint_requires_https_or_loopback():
    assert readback._valid_endpoint('https://example.org')
    assert readback._valid_endpoint('http://localhost:3000')
    assert readback._valid_endpoint('http://127.0.0.1:3000')
    assert readback._valid_endpoint('http://[::1]:3000')
    assert not readback._valid_endpoint('http://example.org')
    assert not readback._valid_endpoint('https://user:password@example.org')
    assert not readback._valid_endpoint('https://example.org/?secret=value')


def test_sdk_v4_mock_transport_contract(monkeypatch, tmp_path, capsys):
    """Exercise real SDK response decoding without any socket or Langfuse server."""
    langfuse = pytest.importorskip('langfuse')
    httpx = pytest.importorskip('httpx')
    first, second = fixture('a'), fixture('b')
    data = {trace_mapping(item[0])['langfuse_trace_id']:item for item in (first,second)}

    def transport(request):
        tid = request.url.params['traceId']
        trace, obs, scores = data[tid]
        if request.url.path == '/api/public/v2/observations':
            rows = []
            for obj in obs:
                row = {'id':obj['id'],'traceId':tid,'name':obj['name'],
                       'parentObservationId':obj['parent_observation_id'],
                       'type':'GENERATION' if obj['name']=='LLM request' else 'SPAN',
                       'startTime':'2026-10-30T00:00:00Z','projectId':'fixture',
                       'metadata':obj.get('metadata'), 'level':obj.get('level'),
                       'providedModelName':obj.get('provided_model_name'),
                       'usageDetails':obj.get('usage_details')}
                rows.append(row)
            return httpx.Response(200,json={'data':rows,'meta':{}})
        assert request.url.path == '/api/public/v3/scores'
        obj=scores[0]
        row={'id':obj['id'],'projectId':'fixture','name':obj['name'],'value':obj['value'],
             'source':'API','dataType':'NUMERIC','environment':'paper-agent',
             'timestamp':'2026-10-30T00:00:00Z','createdAt':'2026-10-30T00:00:00Z',
             'updatedAt':'2026-10-30T00:00:00Z','subject':obj['subject'],'metadata':obj['metadata']}
        return httpx.Response(200,json={'data':[row],'meta':{'limit':100}})

    client = httpx.Client(transport=httpx.MockTransport(transport),follow_redirects=False)
    from langfuse.api import LangfuseAPI
    api = LangfuseAPI(base_url='https://fixture',username='public',password='secret',httpx_client=client)
    result = readback.readback(api, [first[0],second[0]], time.monotonic()+10)
    assert result['status'] == 'readback_checked'
    assert result['complete_end_to_end_acceptance'] == 'not_verified'

    # Exercise the production entry point, including its base_url construction.
    # A duplicated /api/public prefix must fail the strict transport paths above.
    import langfuse.api
    real_api = langfuse.api.LangfuseAPI
    monkeypatch.setattr(langfuse.api, 'LangfuseAPI', lambda **kwargs: real_api(**kwargs, httpx_client=client))
    monkeypatch.setattr(readback, 'version', lambda _: '4.8.1')
    monkeypatch.setenv('LANGFUSE_BASE_URL', 'https://fixture')
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'public')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'secret')
    monkeypatch.chdir(tmp_path)
    for name, case in (('a.json', first), ('b.json', second)):
        Path(name).write_text(json.dumps(case[0]), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['check', 'a.json', 'b.json'])
    assert readback.main() == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'readback_checked'
