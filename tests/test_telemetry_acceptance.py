"""Transport-independent checks: these do not prove remote delivery."""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from paper_agent.observability import Exporter, LangfuseSink
from paper_agent.pipeline import PaperAgentState
from test_reliability import pipeline


def _trace():
    return {'session_id':'s', 'turn_id':'t', 'run_id':'r', 'case_id':'c',
            'status':'completed', 'commit_status':'committed',
            'dataset_hash':'a'*64, 'evaluation_config_hash':'b'*64,
            'evaluation_schema':'paper_agent_e2e_rule_v1',
            'archive_recall':[{'retrieval_source':'archive','kind':'episode'}],
            'memory_events':[{'operation':'chroma_outbox','degraded':True,'error_type':'SECRET'},
                             {'operation':'chroma_read','degraded':True},
                             {'operation':'chroma_repair_queued','episode_ids':['sensitive']}],
            'stages':[{'stage':'Memory','status':'ok','compaction':{'compressed':True}},
                      {'stage':'Retrieve','attempt':2,'status':'error','invocation_id':'i'}],
            'api_calls':[{'invocation_id':'i','status':'error','model_requested':'fixture',
                          'usage_complete':True,'usage':{'input_tokens':10,'output_tokens':3,
                          'cache_read_input_tokens':5,'cache_creation_input_tokens':0}},
                         {'invocation_id':'i','status':'ok','usage_complete':False}],
            'scores':[{'source':'rule','name':'behavior_match','value':1},
                      {'source':'rule','name':'task_success','value':None}]}


def test_context_and_sdk_projection_preserve_unknown_and_exclusive_usage(tmp_path):
    sent=[]
    exporter=Exporter(True,sent.append,flush_seconds=.5)
    exporter.export(_trace()); assert exporter.flush()
    payload=sent[0]; meta=payload['metadata']
    assert meta['memory_archive_recall_count']==meta['memory_compaction_count']==meta['memory_repair_count']==1
    assert meta['memory_degradation_operations']==['chroma_outbox','chroma_read']
    assert payload['scores'][0]['metadata']['dataset_hash']=='a'*64
    assert payload['scores'][0]['metadata']['evaluation_config_hash']=='b'*64
    assert len(payload['scores'])==1
    assert 'sensitive' not in json.dumps(payload) and 'SECRET' not in json.dumps(payload)
    payload['metadata']['dataset_hash']='local-only-invalid'

    calls=[]
    class Span:
        def start_observation(self,**kw):
            calls.append(kw)
            return Span()
        def end(self): pass
    class Client(Span):
        def create_score(self,**kw): pass
        def flush(self): pass
    sink=LangfuseSink(tmp_path); sink.client=Client(); sink(payload)
    stage=next(c for c in calls if c['name']=='Retrieve #2')
    assert stage['level']=='ERROR'
    generations=[c for c in calls if c.get('as_type')=='generation']
    assert generations[0]['usage_details']=={'input':10,'output':3,'cache_read_input_tokens':5,'cache_creation_input_tokens':0}
    assert generations[0]['level']=='ERROR' and generations[1]['usage_details'] is None


def test_invalid_context_fields_never_exported():
    sent=[]; exporter=Exporter(True,sent.append,flush_seconds=.5)
    trace=_trace(); trace.update(dataset_hash='private text',evaluation_config_hash='C'*64,evaluation_schema='private text')
    exporter.export(trace); assert exporter.flush()
    for key in ('dataset_hash','evaluation_config_hash','evaluation_schema'):
        assert key not in sent[0]['metadata'] and key not in sent[0]['scores'][0]['metadata']


def test_missing_sdk_is_lazy_and_failure_isolated(monkeypatch,tmp_path):
    import importlib.metadata
    def missing(_): raise importlib.metadata.PackageNotFoundError('langfuse')
    monkeypatch.setattr(importlib.metadata,'version',missing)
    disabled=Exporter(False,LangfuseSink(tmp_path/'disabled'),flush_seconds=.1)
    disabled.export(_trace())
    assert disabled.worker is None
    exporter=Exporter(True,LangfuseSink(tmp_path/'enabled'),flush_seconds=.5)
    agent=pipeline(tmp_path,mode='bm25',exporter=exporter)
    state=PaperAgentState('s')
    result=agent.run('alpha',state,'t')
    assert result['answer']['decision']=='answer' and exporter.status=='export_failed'
    assert agent.memory_manager.store.snapshot('s')['turn_count']==1
    assert agent.run('alpha',state,'t')['replayed_commit']


def test_flush_configuration_is_bounded(monkeypatch):
    monkeypatch.setenv('PAPER_TELEMETRY_ENABLED','true')
    monkeypatch.setenv('PAPER_TELEMETRY_FLUSH_SECONDS','0.25')
    assert Exporter.from_env().flush_seconds==.25
    monkeypatch.setenv('PAPER_TELEMETRY_FLUSH_SECONDS','3')
    assert Exporter.from_env().status=='disabled_invalid_configuration'


@pytest.mark.parametrize('wrong_provider,wrong_endpoint',[(True,False),(False,True)])
def test_shared_sdk_resources_fail_closed(tmp_path,monkeypatch,wrong_provider,wrong_endpoint):
    import importlib.metadata
    import types
    calls=[]
    class Client:
        def __init__(self,**kwargs):
            self._resources=types.SimpleNamespace(
                tracer_provider=object() if wrong_provider else kwargs['tracer_provider'],
                base_url='https://other.invalid' if wrong_endpoint else kwargs['base_url'],shutdown=lambda:None)
        def start_observation(self,**kwargs): calls.append(kwargs)
    monkeypatch.setitem(sys.modules,'langfuse',types.SimpleNamespace(Langfuse=Client))
    monkeypatch.setattr(importlib.metadata,'version',lambda _:'4.8.1')
    monkeypatch.setattr('paper_agent.observability._new_private_provider',lambda:object())
    monkeypatch.setattr('paper_agent.observability._client_registry',{})
    for key in ('LANGFUSE_PUBLIC_KEY','LANGFUSE_SECRET_KEY','LANGFUSE_BASE_URL'):
        monkeypatch.setenv(key,'fixture')
    exporter=Exporter(True,LangfuseSink(tmp_path),flush_seconds=.5)
    exporter.export(_trace()); assert exporter.flush()
    assert exporter.status=='export_failed' and calls==[] and list(tmp_path.iterdir())==[]


def test_owned_client_is_reused_and_only_owned_spans_pass_filter(tmp_path,monkeypatch):
    import importlib.metadata
    import types
    created=[]; exported=[]; filters=[]
    class Span:
        sequence=0
        def __init__(self):
            Span.sequence+=1; self.id=f'{Span.sequence:016x}'
        def start_observation(self,**kwargs): return Span()
        def end(self):
            test_span=types.SimpleNamespace(context=types.SimpleNamespace(span_id=int(self.id,16)))
            unrelated=types.SimpleNamespace(context=types.SimpleNamespace(span_id=99999))
            assert filters[0](test_span) and not filters[0](unrelated)
            exported.append(self.id)
    class Client:
        def __init__(self,**kwargs):
            created.append(1); filters.append(kwargs['should_export_span'])
            self._resources=types.SimpleNamespace(tracer_provider=kwargs['tracer_provider'],
                                                  base_url=kwargs['base_url'],shutdown=lambda:None)
        def start_observation(self,**kwargs): return Span()
        def create_score(self,**kwargs): pass
        def flush(self): pass
    monkeypatch.setitem(sys.modules,'langfuse',types.SimpleNamespace(Langfuse=Client))
    monkeypatch.setattr(importlib.metadata,'version',lambda _:'4.8.1')
    monkeypatch.setattr('paper_agent.observability._new_private_provider',lambda:object())
    monkeypatch.setattr('paper_agent.observability._client_registry',{})
    for key in ('LANGFUSE_PUBLIC_KEY','LANGFUSE_SECRET_KEY','LANGFUSE_BASE_URL'):
        monkeypatch.setenv(key,'fixture')
    data={'trace_id':'a'*32,'metadata':{},'stages':[{'metadata':{'stage':'NLU'},'calls':[]}],'scores':[]}
    first=LangfuseSink(tmp_path/'a'); second=LangfuseSink(tmp_path/'b')
    first(data); second({**data,'trace_id':'b'*32})
    assert len(created)==1 and len(exported)==4 and first.client is second.client


def test_known_to_unknown_score_warns_that_remote_may_remain(tmp_path):
    from paper_agent.observability import attach_scores,trace_mapping
    trace=_trace(); trace['observability_mapping']=trace_mapping(trace)
    path=tmp_path/'trace.json'; path.write_text(json.dumps(trace),encoding='utf-8')
    sent=[]; exporter=Exporter(True,sent.append,flush_seconds=.5)
    attach_scores(path,{'behavior_match':None},'rule',exporter)
    score=next(s for s in json.loads(path.read_text())['scores'] if s['name']=='behavior_match')
    assert score['status']=='unknown' and score['remote_status']=='unknown_not_retracted'
    assert sent[0]['scores']==[]


def test_schema_repair_attempts_belong_to_same_stage(tmp_path):
    from paper_agent.providers import APIConfig, CompatibleReasoner
    calls=[]; sent=[]
    def transport(*_args):
        calls.append(1)
        content = 'not json' if len(calls)==1 else json.dumps({
            'intent':'out_of_scope','standalone_query':'q','paper_ids':None,
            'resolved_entities':{},'needs_clarification':False})
        return {'model':'test-model','usage':{'prompt_tokens':10,'completion_tokens':3,
                'prompt_tokens_details':{'cached_tokens':4}},
                'choices':[{'finish_reason':'stop','message':{'content':content}}]}
    exporter=Exporter(True,sent.append,flush_seconds=.5)
    agent=pipeline(tmp_path,mode='bm25',exporter=exporter)
    agent.reasoner=CompatibleReasoner(APIConfig('secret','test'), 's', transport)
    agent.run('alpha',PaperAgentState('s'),'t')
    nlu=next(s for s in sent[0]['stages'] if s['metadata']['stage']=='NLU')
    assert len(nlu['calls'])==2
    assert [c['metadata']['repair'] for c in nlu['calls']]==[False,True]
    assert [c['metadata']['status'] for c in nlu['calls']]==['error','ok']
    assert all(c['usage']['input_tokens']==6 and c['usage']['cache_read_input_tokens']==4 for c in nlu['calls'])


def test_pre_stage_memory_error_retains_local_and_remote_attempt(tmp_path,monkeypatch):
    sent=[]; exporter=Exporter(True,sent.append,flush_seconds=.5)
    agent=pipeline(tmp_path,mode='bm25',exporter=exporter)
    def fail(*_args,**_kwargs): raise RuntimeError('private body')
    monkeypatch.setattr(agent.memory_manager,'load_working_memory',fail)
    with pytest.raises(RuntimeError,match='private body'):
        agent.run('alpha',PaperAgentState('s'),'t')
    local=json.loads((tmp_path/'traces'/'t.json').read_text(encoding='utf-8'))
    assert local['status']=='error' and local['commit_status']=='uncommitted'
    assert sent[0]['metadata']['status']=='error'
    assert len(sent[0]['stages'])==7 and all(s['metadata']['status']=='skipped' for s in sent[0]['stages'])
    assert 'private body' not in json.dumps(sent)


@pytest.mark.parametrize('failure', [ModuleNotFoundError('SDK absent'), TimeoutError('unreachable')])
def test_export_failure_cannot_change_answer_or_idempotent_commit(tmp_path, failure):
    def fail(_): raise failure
    exporter=Exporter(True,fail,flush_seconds=.5)
    agent=pipeline(tmp_path,mode='bm25',exporter=exporter)
    state=PaperAgentState('s')
    first=agent.run('alpha',state,'t')
    assert first['answer']['decision']=='answer' and exporter.status=='export_failed'
    replay=agent.run('alpha',state,'t')
    assert replay['replayed_commit'] and replay['answer']==first['answer']
    durable=agent.memory_manager.store.snapshot('s')
    assert durable['turn_count']==1


def test_full_queue_and_flush_are_bounded():
    started=threading.Event(); release=threading.Event()
    def stalled(_):
        started.set(); release.wait(5)
    exporter=Exporter(True,stalled,flush_seconds=.02)
    exporter.export(_trace()); assert started.wait(1)
    exporter.export(_trace()); exporter.export(_trace())
    assert exporter.status=='dropped_queue_full'
    begin=time.monotonic(); assert not exporter.flush()
    assert time.monotonic()-begin < .5
    release.set(); assert exporter.flush() or exporter.queue.unfinished_tasks > 0


@pytest.mark.parametrize('abnormal',[False,True])
def test_short_process_exit_is_not_blocked_by_stalled_export(abnormal):
    script="""
import threading
from paper_agent.observability import Exporter
started=threading.Event()
def sink(_):
    started.set(); threading.Event().wait(10)
exporter=Exporter(True,sink,flush_seconds=.01)
exporter.export({'session_id':'s','turn_id':'t','stages':[]})
assert started.wait(2)
assert not exporter.flush()
""" + ("raise RuntimeError('synthetic exception')\n" if abnormal else '')
    env=dict(os.environ); env['PYTHONPATH']=os.path.abspath('src')
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,env=env,timeout=4)
    assert result.returncode==(1 if abnormal else 0)


@pytest.mark.parametrize('abnormal',[False,True])
def test_real_sdk_short_process_exit_with_blocked_flush(abnormal,tmp_path):
    # Optional separately installed SDK: no real service, no network export.
    from pathlib import Path
    sdk=Path('.cache/langfuse-sdk-v4').resolve()
    if not (sdk/'langfuse-4.8.1.dist-info').is_dir():
        pytest.skip('isolated optional Langfuse SDK 4.8.1 not installed')
    script="""
import threading
import langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from langfuse._client.resource_manager import LangfuseResourceManager
from paper_agent.observability import Exporter
blocked=threading.Event()
def blocked_flush(self):
    blocked.set(); threading.Event().wait(30)
LangfuseResourceManager.flush=blocked_flush
original=langfuse.Langfuse
def memory_client(**kwargs):
    return original(**kwargs,span_exporter=InMemorySpanExporter())
langfuse.Langfuse=memory_client
exporter=Exporter(True,flush_seconds=.01)
exporter.export({'session_id':'s','turn_id':'t','stages':[]})
assert blocked.wait(5), exporter.status
assert not exporter.flush()
""" + ("raise RuntimeError('synthetic exception')\n" if abnormal else '')
    env=dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join([str(sdk),os.path.abspath('src')]),
               LANGFUSE_PUBLIC_KEY='pk-lf-local-fixture',LANGFUSE_SECRET_KEY='sk-lf-local-fixture',
               LANGFUSE_BASE_URL='http://127.0.0.1:1', PAPER_TELEMETRY_DEDUP_DIR=str(tmp_path))
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,env=env,timeout=10)
    assert result.returncode==(1 if abnormal else 0),result.stderr
    if abnormal:
        assert 'synthetic exception' in result.stderr

