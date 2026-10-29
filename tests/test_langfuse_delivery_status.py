"""SDK doubles distinguish delivery attempts from server verification."""
import importlib.metadata
import sys
import types

from paper_agent.observability import Exporter, LangfuseSink


def test_failed_tree_then_replay_reports_unverified_claim_and_sends_score(monkeypatch,tmp_path):
    attempts=[]
    scores=[]
    class Client:
        def __init__(self,**kwargs): pass
        def start_observation(self,**kwargs):
            attempts.append(kwargs['trace_context']['trace_id'])
            raise OSError('simulated ambiguous delivery')
        def create_score(self,**kwargs): scores.append(kwargs)
        def flush(self): pass
    monkeypatch.setitem(sys.modules,'langfuse',types.SimpleNamespace(Langfuse=Client))
    monkeypatch.setattr(importlib.metadata,'version',lambda _: '4.7.0')
    for key in ('LANGFUSE_PUBLIC_KEY','LANGFUSE_SECRET_KEY','LANGFUSE_BASE_URL'):
        monkeypatch.setenv(key,'fixture')
    trace={'session_id':'s','turn_id':'t','run_id':'run','stages':[],
           'scores':[{'source':'rule','name':'behavior_match','value':1}]}
    exporter=Exporter(True,LangfuseSink(dedup_dir=tmp_path),flush_seconds=.5)
    exporter.export(trace)
    assert exporter.flush() and exporter.status=='export_failed'
    exporter.export(trace)
    assert exporter.flush() and exporter.status=='tree_already_claimed_unverified'
    assert len(attempts)==1 and len(scores)==1
    exporter.export_scores(trace)
    assert exporter.flush() and exporter.status=='scores_sent_unverified'
    assert len(attempts)==1 and len(scores)==2 and scores[0]['score_id']==scores[1]['score_id']


def test_successful_local_sink_is_not_server_confirmation(tmp_path):
    from test_reliability import pipeline
    from paper_agent.pipeline import PaperAgentState
    exporter=Exporter(True,lambda _:None,flush_seconds=.5)
    p=pipeline(tmp_path,mode='bm25',exporter=exporter)
    assert p.run('alpha',PaperAgentState('s'),'t')['answer']['decision']=='answer'
    assert exporter.status=='sent_unverified'
