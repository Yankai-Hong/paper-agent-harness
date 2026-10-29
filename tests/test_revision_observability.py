import json
import threading
import time
from paper_agent.observability import Exporter
from paper_agent.providers import APIConfig, CompatibleReasoner, ProviderTransportError
from paper_agent.pipeline import PaperAgentState
from test_reliability import pipeline, response

def test_attempt_tree_redaction_and_disabled(tmp_path):
    sent = []
    exporter = Exporter(True, sent.append, flush_seconds=.2)
    p = pipeline(tmp_path, mode='bm25', exporter=exporter)
    attempts = []
    def transport(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise ProviderTransportError(429)
        return response({'intent':'out_of_scope','standalone_query':'q','paper_ids':None,
                         'resolved_entities':{},'needs_clarification':False})
    p.reasoner = CompatibleReasoner(APIConfig('secret','test',retry_base_seconds=0), 's', transport)
    result=p.run('private preference and paper text', PaperAgentState('s'), 't', case_id='case',run_id='run')
    local = json.loads(__import__('pathlib').Path(result['trace_path']).read_text())
    assert len(local['api_calls']) == 2
    assert len({c['request_id'] for c in local['api_calls']}) == 2
    nlu = next(s for s in sent[0]['stages'] if s['metadata']['stage']=='NLU')
    assert len(nlu['calls']) == 2
    assert all(c['invocation_id'] == local['api_calls'][0]['invocation_id'] for c in local['api_calls'])
    assert 'private' not in json.dumps(sent) and 'secret' not in json.dumps(sent)
    disabled = Exporter(False, lambda _: (_ for _ in ()).throw(AssertionError()))
    disabled.export(local)
    assert disabled.worker is None

def test_failed_and_stuck_export_bounded(tmp_path):
    def fail(_): raise ConnectionError('private')
    p = pipeline(tmp_path, mode='bm25', exporter=Exporter(True, fail, flush_seconds=.05))
    assert p.run('alpha', PaperAgentState('s'), 't')['answer']['decision']=='answer'
    assert p.exporter.status == 'export_failed'
    release = threading.Event()
    stuck = Exporter(True, lambda _: release.wait(5), flush_seconds=.01)
    local = json.loads((tmp_path/'traces/t.json').read_text())
    start = time.monotonic()
    for _ in range(10): stuck.export(local)
    assert not stuck.flush() and time.monotonic()-start < .5
    release.set()

def test_exception_trace_export(tmp_path):
    import pytest
    sent=[]
    p=pipeline(tmp_path, failure='reflect', exporter=Exporter(True,sent.append,flush_seconds=.1))
    with pytest.raises(ValueError): p.run('alpha',PaperAgentState('s'),'t')
    assert sent[0]['metadata']['commit_status']=='uncommitted'

def test_score_updates_do_not_duplicate_observations_or_export_unknown(tmp_path):
    from paper_agent.observability import attach_scores
    sent=[]
    exporter=Exporter(True,sent.append,flush_seconds=.2)
    p=pipeline(tmp_path,mode='bm25',exporter=exporter)
    p.run('alpha',PaperAgentState('s'),'t')
    path=tmp_path/'traces/t.json'
    attach_scores(path,{'behavior_match':True,'task_success':None,'secret message':1},'rule',exporter)
    exporter.flush()
    assert len(sent)==2 and sent[1]['scores_only'] and sent[1]['stages']==[]
    assert [s['name'] for s in sent[1]['scores']]==['behavior_match']
    trace=json.loads(path.read_text())
    assert next(s for s in trace['scores'] if s['name']=='task_success')['value'] is None

def test_malformed_optional_telemetry_configuration_is_disabled(monkeypatch,tmp_path):
    monkeypatch.setenv('PAPER_TELEMETRY_ENABLED','true')
    monkeypatch.setenv('PAPER_TELEMETRY_SAMPLE_RATE','not-a-number')
    p=pipeline(tmp_path,mode='bm25')
    assert p.exporter.status=='disabled_invalid_configuration'
    assert p.run('alpha',PaperAgentState('s'),'t')['answer']['decision']=='answer'

def test_export_usage_whitelist_and_reject_free_text():
    sent=[]
    exporter=Exporter(True,sent.append,flush_seconds=.2)
    trace={'session_id':'s','turn_id':'t','status':'finished', 'stages':[{'stage':'NLU','invocation_id':'i'}],
           'api_calls':[{'invocation_id':'i','model_requested':'my secret phrase','usage_complete':True,
                         'usage':{'input_tokens':2,'private':'secret','other':4}}]}
    exporter.export(trace);exporter.flush()
    call=sent[0]['stages'][0]['calls'][0]
    assert call['usage']=={'input_tokens':2}
    assert 'my secret phrase' not in json.dumps(sent)

def test_sdk_mapping_is_tree_and_score_updates_have_stable_ids(monkeypatch,tmp_path):
    import sys
    import types
    import importlib.metadata
    from paper_agent.observability import LangfuseSink
    operations=[]
    class Span:
        def __init__(self,name): self.name=name
        def start_observation(self,**kwargs):
            operations.append(('child',self.name,kwargs['name'],kwargs.get('as_type')))
            return Span(kwargs['name'])
        def end(self): operations.append(('end',self.name))
    class Client:
        def __init__(self,**kwargs): pass
        def start_observation(self,**kwargs):
            operations.append(('root',kwargs['name'],kwargs['trace_context']['trace_id']))
            return Span(kwargs['name'])
        def create_score(self,**kwargs): operations.append(('score',kwargs['score_id'],kwargs['value']))
        def flush(self): operations.append(('flush',))
    monkeypatch.setitem(sys.modules,'langfuse',types.SimpleNamespace(Langfuse=Client))
    monkeypatch.setattr(importlib.metadata,'version',lambda _: '4.7.0')
    for name in ('LANGFUSE_PUBLIC_KEY','LANGFUSE_SECRET_KEY','LANGFUSE_BASE_URL'):
        monkeypatch.setenv(name,'fixture')
    data={'trace_id':'a'*32,'metadata':{},'stages':[{'metadata':{'stage':'NLU'},'calls':[{'metadata':{'model_requested':'fixture'}}]}],
          'scores':[{'name':'behavior_match','value':1,'metadata':{'source':'rule'}}]}
    sink=LangfuseSink(dedup_dir=tmp_path)
    sink(data)
    sink({**data,'scores_only':True,'stages':[]})
    assert len([x for x in operations if x[0]=='root'])==1
    assert ('child','Turn','NLU',None) in operations
    assert ('child','NLU','LLM request','generation') in operations
    scores=[x for x in operations if x[0]=='score']
    assert len(scores)==2 and scores[0][1]==scores[1][1]

def test_duplicate_tree_suppressed_across_sink_instances_and_failed_send(monkeypatch,tmp_path):
    import sys,types,importlib.metadata
    from concurrent.futures import ThreadPoolExecutor
    from paper_agent.observability import LangfuseSink
    calls=[]
    class Client:
        def __init__(self,**kwargs): pass
        def start_observation(self,**kwargs):
            calls.append(kwargs['trace_context']['trace_id'])
            raise OSError('simulated ambiguous send')
        def create_score(self,**kwargs): pass
        def flush(self): pass
    monkeypatch.setitem(sys.modules,'langfuse',types.SimpleNamespace(Langfuse=Client))
    monkeypatch.setattr(importlib.metadata,'version',lambda _: '4.7.0')
    for name in ('LANGFUSE_PUBLIC_KEY','LANGFUSE_SECRET_KEY','LANGFUSE_BASE_URL'):
        monkeypatch.setenv(name,'fixture')
    data={'trace_id':'b'*32,'metadata':{},'stages':[],'scores':[]}
    def send(_):
        try: LangfuseSink(dedup_dir=tmp_path)(data)
        except OSError: pass
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(send,range(4)))
    send(None)  # Persisted claim survives a fresh sink after ambiguous failure.
    assert calls==['b'*32]
    names=[p.name for p in tmp_path.iterdir()]
    assert len(names)==1 and names[0].endswith('-'+'b'*32+'.claimed')
    assert 'fixture' not in names[0]

def test_dedup_is_scoped_to_destination(monkeypatch,tmp_path):
    from paper_agent.observability import LangfuseSink
    monkeypatch.setenv('LANGFUSE_BASE_URL','https://first.example')
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY','public-first')
    sink=LangfuseSink(dedup_dir=tmp_path)
    assert sink._claim_tree('c'*32)
    assert not sink._claim_tree('c'*32)
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY','public-second')
    assert sink._claim_tree('c'*32)
    assert len(list(tmp_path.iterdir()))==2

def test_run_identity_and_scores_are_isolated_and_stable(tmp_path):
    from paper_agent.observability import trace_mapping,score_id,attach_scores
    base={'session_id':'s','turn_id':'t','case_id':'case','stages':[]}
    a={**base,'run_id':'baseline'};b={**base,'run_id':'split'}
    ia=trace_mapping(a)['langfuse_trace_id']; ib=trace_mapping(b)['langfuse_trace_id']
    assert ia!=ib and len(ia)==32 and int(ia,16)>0
    assert ia==trace_mapping(dict(a))['langfuse_trace_id']
    assert score_id(ia,'rule','behavior_match')!=score_id(ib,'rule','behavior_match')
    assert score_id(ia,'rule','behavior_match')==score_id(ia,'rule','behavior_match')
    assert trace_mapping(base)==trace_mapping({**base,'run_id':None})
    assert trace_mapping(base)['langfuse_trace_id'] not in {ia,ib}
    a['observability_mapping']=trace_mapping(a)
    path=tmp_path/'trace.json';path.write_text(json.dumps(a),encoding='utf-8')
    sent=[];exporter=Exporter(True,sent.append,flush_seconds=.2)
    attach_scores(path,{'behavior_match':True},'rule',exporter);exporter.flush()
    saved=json.loads(path.read_text(encoding='utf-8'))
    assert saved['observability_mapping']['langfuse_trace_id']==ia==sent[0]['trace_id']
    attach_scores(path,{'behavior_match':False},'rule',exporter);exporter.flush()
    assert sent[1]['trace_id']==ia and sent[1]['scores'][0]['value']==0

def test_historical_trace_with_missing_mapping_keeps_local_score_without_export(tmp_path):
    from paper_agent.observability import attach_scores
    trace={'session_id':'s','turn_id':'t','run_id':'historic','stages':[]}
    path=tmp_path/'old.json';path.write_text(json.dumps(trace),encoding='utf-8')
    sent=[];exporter=Exporter(True,sent.append,flush_seconds=.2)
    attach_scores(path,{'behavior_match':True},'rule',exporter);exporter.flush()
    saved=json.loads(path.read_text(encoding='utf-8'))
    assert saved['scores'][0]['value']==1
    assert saved['score_export_status']=='withheld_missing_or_mismatched_mapping'
    assert sent==[]

def test_local_run_paths_are_separate_and_interactive_compatible(tmp_path):
    from paper_agent.observability import local_trace_path
    base={'session_id':'s','turn_id':'t'}
    assert local_trace_path(tmp_path,base)==tmp_path/'t.json'
    a=local_trace_path(tmp_path,{**base,'run_id':'baseline'})
    b=local_trace_path(tmp_path,{**base,'run_id':'split'})
    assert a!=b and a.name==b.name=='t.json'
    assert a==local_trace_path(tmp_path,{**base,'run_id':'baseline'})

def test_e2e_rule_score_attaches_to_run_trace(tmp_path):
    from pathlib import Path
    from paper_agent.eval_tasks import run_e2e_session
    def factory(session_id):
        p=pipeline(tmp_path/'factory',mode='bm25')
        p.trace_dir=tmp_path/'traces'/session_id
        return p
    rows=[{'session_id':'s','turn_id':'t','turn':1,'question':'alpha','success_conditions':{'decision':'answer'}}]
    prediction=run_e2e_session(factory,'s',rows,tmp_path)[0]
    assert prediction['assessment']['execution_success']
    path=Path(prediction['trace_path'])
    trace=json.loads(path.read_text(encoding='utf-8'))
    assert trace['observability_mapping']['scope']=='run'
    assert any(s['name']=='execution_success' and s['value']==1 for s in trace['scores'])

def test_safe_retrieval_and_memory_summary():
    sent=[]; exporter=Exporter(True,sent.append,flush_seconds=.2)
    trace={'session_id':'secret-user','turn_id':'turn','run_id':'run', 'status':'completed',
           'archive_recall':[{'kind':'episode','content':'private preference','retrieval_source':'chroma'},
                             {'kind':'long_term','content':'private preference','retrieval_source':'sqlite'}],
           'memory_events':[{'operation':'chroma_repair_queued','episode_ids':['secret1','secret2']}],
           'degradation_reasons':['reranker_unavailable','secret failure text'],
           'stages':[{'stage':'Retrieve','output':{'candidate_ids':['private-chunk']},
                      'candidate_coverage_not_semantic_support':{'g1':['private-chunk'],'g2':[]}},
                     {'stage':'Rerank','input':{'candidate_ids':['private-chunk']},
                      'output':{'evidence_ids':['private-chunk']},'reason':'reranker_unavailable'}]}
    exporter.export(trace);exporter.flush()
    payload=sent[0];meta=payload['metadata'];retrieve=payload['stages'][0]['metadata'];rerank=payload['stages'][1]['metadata']
    assert (meta['memory_recall_count'],meta['memory_repair_count'])==(2,2)
    assert meta['degradation_reasons']==['reranker_unavailable']
    assert (retrieve['candidate_count'],retrieve['subgoal_count'],retrieve['subgoals_with_candidates'])==(1,2,1)
    assert rerank['rerank_candidate_count']==1 and rerank['evidence_count']==1
    serialized=json.dumps(payload)
    for secret in ('private preference','private-chunk','secret-user','secret failure text','secret1'):
        assert secret not in serialized
